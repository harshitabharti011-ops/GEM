// Word-level hardware macros carried through GEM without AIG decomposition.
//
// These are the in-memory counterpart of the blackbox cells in
// synth/gem_macros.v. A MacroBlock plays the same role for DSP48E2 / CARRY4 /
// SRLC32E that `RAMBlock` plays for `$__RAMGEM_SYNC_`.
//
// The one structural difference from RAMBlock, and the reason this file is not
// a copy of it: a RAM is purely sequential, so its outputs have no
// combinational fan-in and the scheduler can treat them as graph roots. That
// is NOT true here.
//
//   CARRY4  is entirely combinational. Its outputs must sit INSIDE the
//           combinational cone or every carry chain would force a stage
//           boundary, which is exactly the graph-depth explosion we are here
//           to avoid.
//   SRLC32E is mixed. Q = state[A] reads combinationally through A[4:0], while
//           D and CE are consumed at the clock edge.
//   DSP48E2 is cleanly sequential: only PREG is clocked, so P behaves like a
//           DFF output and has no combinational fan-in at all.
//
// So every macro input is classified as either combinational (outputs depend
// on it this instant) or sequential (consumed at the edge). The traversal in
// aig.rs recurses through the first set and stops at the second.

use crate::aigpdk::{GEMDSP_CELL, GEMCARRY4_CELL, GEMSRL32_CELL};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroKind { Dsp48e2, Carry4, Srlc32e }

impl MacroKind {
    pub fn from_celltype(celltype: &str) -> Option<MacroKind> {
        match celltype {
            GEMDSP_CELL    => Some(MacroKind::Dsp48e2),
            GEMCARRY4_CELL => Some(MacroKind::Carry4),
            GEMSRL32_CELL  => Some(MacroKind::Srlc32e),
            _ => None,
        }
    }

    /// Does this macro hold state across a clock edge?
    ///
    /// Only stateful macros become endpoint groups: a partition must be
    /// scheduled to commit their next-cycle value. CARRY4 holds nothing and is
    /// ordinary combinational logic as far as the scheduler is concerned.
    pub fn is_sequential(self) -> bool {
        !matches!(self, MacroKind::Carry4)
    }

    /// 64-bit words of device state this macro needs (DSP PREG, SRL shifter).
    pub fn state_words(self) -> usize {
        match self {
            MacroKind::Dsp48e2 => 1,   // P[47:0]
            MacroKind::Srlc32e => 1,   // 32-bit shift register
            MacroKind::Carry4  => 0,
        }
    }

    pub fn num_outputs(self) -> usize {
        match self {
            MacroKind::Dsp48e2 => 48,       // P[47:0]
            MacroKind::Carry4  => 8,        // CO[3:0] then O[3:0]
            MacroKind::Srlc32e => 2,        // Q then Q31
        }
    }

    /// Flat output slot for a pin name and bit index, or None if not an output.
    ///
    /// The slot numbering here is the canonical order that flatten.rs will use
    /// when it lays these out in the word-state region, so it must stay stable.
    pub fn output_slot(self, pin: &str, bit: Option<isize>) -> Option<usize> {
        match (self, pin) {
            (MacroKind::Dsp48e2, "P")   => bit.map(|b| b as usize),
            (MacroKind::Carry4,  "CO")  => bit.map(|b| b as usize),
            (MacroKind::Carry4,  "O")   => bit.map(|b| 4 + b as usize),
            (MacroKind::Srlc32e, "Q")   => Some(0),
            (MacroKind::Srlc32e, "Q31") => Some(1),
            _ => None,
        }
    }

    /// Is this input pin one the outputs depend on combinationally?
    pub fn is_comb_input(self, pin: &str) -> bool {
        match self {
            // every CARRY4 input feeds the chain directly
            MacroKind::Carry4  => matches!(pin, "S" | "DI" | "CIN" | "CYINIT"),
            // the read address is combinational; D and CE are edge-consumed
            MacroKind::Srlc32e => pin == "A",
            // A/B/C/D/MODE reach the outside world only through the clocked P
            MacroKind::Dsp48e2 => false,
        }
    }

    /// Is this input pin consumed at the clock edge?
    pub fn is_seq_input(self, pin: &str) -> bool {
        match self {
            MacroKind::Carry4  => false,
            MacroKind::Srlc32e => matches!(pin, "D" | "CE"),
            MacroKind::Dsp48e2 =>
                matches!(pin, "A" | "B" | "C" | "D" | "MODE" | "USE_PREADD"),
        }
    }
}

/// One macro instance, with its AIG-level connectivity resolved.
///
/// Inputs are stored iv-encoded -- `aigpin << 1 | invert` -- the same
/// convention `DFF::d_iv` and `RAMBlock::port_*_iv` use. Consumers must shift
/// right by one to recover the aigpin id.
#[derive(Debug, Clone)]
pub struct MacroBlock {
    pub kind: MacroKind,
    /// netlistdb cell id, so diagnostics can name the instance.
    pub cellid: usize,
    /// Inputs the outputs depend on this instant, iv-encoded.
    pub comb_in_iv: Vec<usize>,
    /// Inputs latched at the clock edge, iv-encoded.
    pub seq_in_iv: Vec<usize>,
    /// Clock enable, from `AIG::trace_clock_pin`. Zero for combinational
    /// macros, which have no clock pin.
    pub clk_en_iv: usize,
    /// Output aigpins indexed by `MacroKind::output_slot`. Zero means the
    /// output is unconnected in this netlist, which is legal.
    pub outputs: Vec<usize>,
}

impl MacroBlock {
    pub fn new(kind: MacroKind, cellid: usize) -> Self {
        MacroBlock {
            kind, cellid,
            comb_in_iv: Vec::new(),
            seq_in_iv: Vec::new(),
            clk_en_iv: 0,
            outputs: vec![0; kind.num_outputs()],
        }
    }

    pub fn set_output(&mut self, slot: usize, aigpin: usize) {
        self.outputs[slot] = aigpin;
    }

    /// Combinational fan-in as bare aigpin ids, skipping constants.
    ///
    /// Returned as a Vec rather than taking a callback so that callers can
    /// hold it while mutating other fields of the AIG -- the fanout CSR build
    /// needs exactly that.
    pub fn comb_inputs(&self) -> Vec<usize> {
        self.comb_in_iv.iter().map(|iv| iv >> 1).filter(|&i| i >= 1).collect()
    }

    /// Every input that must be realised for this macro to be clocked.
    pub fn for_each_seq_input(&self, mut f: impl FnMut(usize)) {
        for iv in &self.seq_in_iv {
            if (iv >> 1) >= 1 { f(iv >> 1); }
        }
        // A stateful macro also needs its combinational inputs realised: the
        // SRL read address is combinational but still has to be available in
        // the cycle the shift commits.
        for iv in &self.comb_in_iv {
            if (iv >> 1) >= 1 { f(iv >> 1); }
        }
        if (self.clk_en_iv >> 1) >= 1 { f(self.clk_en_iv >> 1); }
    }

    pub fn is_sequential(&self) -> bool { self.kind.is_sequential() }
}

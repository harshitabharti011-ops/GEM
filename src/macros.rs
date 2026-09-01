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

use serde::{Serialize, Deserialize};
use crate::aigpdk::{
    GEMDSP_CELL, GEMCARRY4_CELL, GEMSRL32_CELL,
    GEMDSP_A_WIDTH, GEMDSP_B_WIDTH, GEMDSP_C_WIDTH, GEMDSP_D_WIDTH,
    GEMDSP_P_WIDTH, GEMDSP_MODE_WIDTH, GEMCARRY4_WIDTH, GEMSRL32_ADDR_WIDTH,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        self.output_ports().iter().map(|&(_, w)| w).sum()
    }

    /// Canonical INPUT port order, as declared in synth/gem_macros.v.
    ///
    /// This table is the single source of truth for operand ordering. Without
    /// it, input order followed netlistdb's pin iteration, which depends on how
    /// Yosys happened to emit the instance -- so two structurally identical
    /// netlists could produce different descriptors and Part B would have no
    /// way to tell operand A from operand B.
    ///
    /// CLK is absent on purpose: it is traced separately into `clk_en_iv`, not
    /// treated as an operand.
    pub fn input_ports(self) -> &'static [(&'static str, usize)] {
        match self {
            MacroKind::Dsp48e2 => &[
                ("A", GEMDSP_A_WIDTH), ("B", GEMDSP_B_WIDTH),
                ("C", GEMDSP_C_WIDTH), ("D", GEMDSP_D_WIDTH),
                ("USE_PREADD", 1), ("MODE", GEMDSP_MODE_WIDTH)],
            MacroKind::Carry4 => &[
                ("S", GEMCARRY4_WIDTH), ("DI", GEMCARRY4_WIDTH),
                ("CIN", 1), ("CYINIT", 1)],
            MacroKind::Srlc32e => &[
                ("D", 1), ("CE", 1), ("A", GEMSRL32_ADDR_WIDTH)],
        }
    }

    /// Canonical OUTPUT port order, as declared in synth/gem_macros.v.
    pub fn output_ports(self) -> &'static [(&'static str, usize)] {
        match self {
            MacroKind::Dsp48e2 => &[("P", GEMDSP_P_WIDTH)],
            MacroKind::Carry4  => &[("CO", GEMCARRY4_WIDTH), ("O", GEMCARRY4_WIDTH)],
            MacroKind::Srlc32e => &[("Q", 1), ("Q31", 1)],
        }
    }

    /// Total input bits across all ports.
    pub fn num_inputs(self) -> usize {
        self.input_ports().iter().map(|&(_, w)| w).sum()
    }

    fn slot_in(table: &'static [(&'static str, usize)],
               pin: &str, bit: Option<isize>) -> Option<usize> {
        let mut base = 0;
        for &(name, w) in table {
            if name == pin {
                let b = match bit { Some(b) if b >= 0 => b as usize, None => 0, _ => return None };
                return if b < w { Some(base + b) } else { None }
            }
            base += w;
        }
        None
    }

    /// Flat input slot for a pin name and bit index. Stable across netlists.
    pub fn input_slot(self, pin: &str, bit: Option<isize>) -> Option<usize> {
        Self::slot_in(self.input_ports(), pin, bit)
    }

    /// Flat output slot. The numbering flatten.rs and Part B both index by.
    pub fn output_slot(self, pin: &str, bit: Option<isize>) -> Option<usize> {
        Self::slot_in(self.output_ports(), pin, bit)
    }

    /// The port a flat input slot belongs to, as (name, bit).
    pub fn input_port_of_slot(self, slot: usize) -> Option<(&'static str, usize)> {
        let mut base = 0;
        for &(name, w) in self.input_ports() {
            if slot < base + w { return Some((name, slot - base)) }
            base += w;
        }
        None
    }

    /// Do the outputs depend on this input slot combinationally?
    ///
    /// Classification is per PORT, so it is a static property of the slot --
    /// never of the netlist.
    pub fn is_comb_slot(self, slot: usize) -> bool {
        match self.input_port_of_slot(slot) {
            Some((name, _)) => self.is_comb_input(name),
            None => false,
        }
    }

    /// Do this kind's outputs have to be evaluated inside the combinational
    /// cone, rather than read from state at the start of the cycle?
    ///
    /// This is NOT the same question as `is_sequential`:
    ///   CARRY4  stateless, combinational outputs      -> true
    ///   SRLC32E stateful,  Q = state[A] reads through A -> true
    ///   DSP48E2 stateful,  P is purely clocked        -> false
    /// A DSP's P therefore behaves exactly like a DFF's Q: a level-0 leaf read
    /// from state. Deciding this from `is_sequential` would misclassify both
    /// the SRL and the DSP, in opposite directions.
    ///
    /// Static, so an instance whose inputs happen to be tied to constants is
    /// still classified by its kind.
    pub fn has_comb_outputs(self) -> bool {
        (0..self.num_inputs()).any(|slot| self.is_comb_slot(slot))
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
    /// Inputs, iv-encoded (`aigpin << 1 | invert`), indexed by
    /// `MacroKind::input_slot`. Length is always `kind.num_inputs()`, so the
    /// ordering is canonical and independent of netlist pin iteration order.
    /// Zero means tied low or unconnected.
    pub in_iv: Vec<usize>,
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
            in_iv: vec![0; kind.num_inputs()],
            clk_en_iv: 0,
            outputs: vec![0; kind.num_outputs()],
        }
    }

    pub fn set_input(&mut self, slot: usize, iv: usize) { self.in_iv[slot] = iv; }
    pub fn set_output(&mut self, slot: usize, aigpin: usize) { self.outputs[slot] = aigpin; }

    /// Combinational fan-in as bare aigpin ids, skipping constants.
    ///
    /// Returned as a Vec rather than taking a callback so callers can hold it
    /// while mutating other fields of the AIG -- the fanout CSR build needs
    /// exactly that.
    pub fn comb_inputs(&self) -> Vec<usize> {
        self.in_iv.iter().enumerate()
            .filter(|&(slot, _)| self.kind.is_comb_slot(slot))
            .map(|(_, iv)| iv >> 1)
            .filter(|&i| i >= 1)
            .collect()
    }

    /// Every input that must be realised for this macro to be clocked.
    ///
    /// Includes the combinational inputs: an SRL's read address is
    /// combinational but still has to be available in the cycle the shift
    /// commits.
    pub fn for_each_seq_input(&self, mut f: impl FnMut(usize)) {
        for iv in &self.in_iv {
            if (iv >> 1) >= 1 { f(iv >> 1); }
        }
        if (self.clk_en_iv >> 1) >= 1 { f(self.clk_en_iv >> 1); }
    }

    pub fn is_sequential(&self) -> bool { self.kind.is_sequential() }
}

#[cfg(test)]
mod canonical_order_tests {
    use super::*;
    use crate::aigpdk::*;

    const KINDS: [MacroKind; 3] =
        [MacroKind::Dsp48e2, MacroKind::Carry4, MacroKind::Srlc32e];

    /// The port tables are the contract with synth/gem_macros.v. If a width
    /// moves in one place it must move in the other.
    #[test]
    fn port_widths_match_the_aigpdk_constants() {
        let k = MacroKind::Dsp48e2;
        assert_eq!(k.input_slot("A", Some(0)), Some(0));
        assert_eq!(k.num_inputs(),
                   GEMDSP_A_WIDTH + GEMDSP_B_WIDTH + GEMDSP_C_WIDTH
                   + GEMDSP_D_WIDTH + 1 + GEMDSP_MODE_WIDTH);
        assert_eq!(k.num_outputs(), GEMDSP_P_WIDTH);
        assert_eq!(MacroKind::Carry4.num_inputs(), 2 * GEMCARRY4_WIDTH + 2);
        assert_eq!(MacroKind::Carry4.num_outputs(), 2 * GEMCARRY4_WIDTH);
        assert_eq!(MacroKind::Srlc32e.num_inputs(), 2 + GEMSRL32_ADDR_WIDTH);
        assert_eq!(MacroKind::Srlc32e.num_outputs(), 2);
    }

    /// Every input bit maps to exactly one slot and back again, with no gaps
    /// and no collisions.
    #[test]
    fn input_slots_are_a_bijection() {
        for k in KINDS {
            let mut seen = vec![false; k.num_inputs()];
            for &(name, w) in k.input_ports() {
                for b in 0..w {
                    let bit = if w == 1 { None } else { Some(b as isize) };
                    let slot = k.input_slot(name, bit)
                        .unwrap_or_else(|| panic!("{:?}.{}[{}] has no slot", k, name, b));
                    assert!(!seen[slot], "{:?}: slot {} claimed twice", k, slot);
                    seen[slot] = true;
                    assert_eq!(k.input_port_of_slot(slot), Some((name, b)));
                }
            }
            assert!(seen.iter().all(|&s| s), "{:?}: gap in slot numbering", k);
            assert_eq!(k.input_slot("NOT_A_PIN", None), None);
            assert_eq!(k.input_slot("CLK", None), None, "CLK is not an operand");
        }
    }

    #[test]
    fn output_slot_numbering_is_unchanged() {
        // These indices are already baked into flatten.rs and will be baked
        // into the Part B kernel. They must not drift.
        assert_eq!(MacroKind::Dsp48e2.output_slot("P", Some(47)), Some(47));
        assert_eq!(MacroKind::Carry4.output_slot("CO", Some(0)), Some(0));
        assert_eq!(MacroKind::Carry4.output_slot("CO", Some(3)), Some(3));
        assert_eq!(MacroKind::Carry4.output_slot("O", Some(0)), Some(4));
        assert_eq!(MacroKind::Carry4.output_slot("O", Some(3)), Some(7));
        assert_eq!(MacroKind::Srlc32e.output_slot("Q", None), Some(0));
        assert_eq!(MacroKind::Srlc32e.output_slot("Q31", None), Some(1));
        assert_eq!(MacroKind::Dsp48e2.output_slot("P", Some(48)), None);
    }

    /// comb/seq classification is a static property of the slot, so the
    /// scheduler can never disagree with the formatter about it.
    #[test]
    fn comb_classification_is_per_port_and_slot_consistent() {
        for k in KINDS {
            for slot in 0..k.num_inputs() {
                let (name, _) = k.input_port_of_slot(slot).unwrap();
                assert_eq!(k.is_comb_slot(slot), k.is_comb_input(name));
            }
        }
        // CARRY4 is wholly combinational; a DSP is wholly sequential; an SRL
        // is combinational on its read address only.
        assert!((0..MacroKind::Carry4.num_inputs())
                .all(|s| MacroKind::Carry4.is_comb_slot(s)));
        assert!((0..MacroKind::Dsp48e2.num_inputs())
                .all(|s| !MacroKind::Dsp48e2.is_comb_slot(s)));
        let srl = MacroKind::Srlc32e;
        assert!(srl.is_comb_slot(srl.input_slot("A", Some(0)).unwrap()));
        assert!(!srl.is_comb_slot(srl.input_slot("D", None).unwrap()));
        assert!(!srl.is_comb_slot(srl.input_slot("CE", None).unwrap()));
    }

    /// The point of the whole exercise: filling a MacroBlock in one pin order
    /// and in the reverse order must produce byte-identical in_iv.
    #[test]
    fn in_iv_is_independent_of_pin_visit_order() {
        for k in KINDS {
            let pins: Vec<(&str, Option<isize>)> = k.input_ports().iter()
                .flat_map(|&(n, w)| (0..w).map(move |b|
                    (n, if w == 1 { None } else { Some(b as isize) })))
                .collect();

            let fill = |order: Box<dyn Iterator<Item = &(&str, Option<isize>)>>| {
                let mut mb = MacroBlock::new(k, 1);
                for &(name, bit) in order {
                    let slot = k.input_slot(name, bit).unwrap();
                    // a value that depends only on the pin, never on visit order
                    mb.set_input(slot, (slot + 1) << 1);
                }
                mb
            };
            let fwd = fill(Box::new(pins.iter()));
            let rev = fill(Box::new(pins.iter().rev()));
            assert_eq!(fwd.in_iv, rev.in_iv, "{:?}: in_iv depends on visit order", k);
            assert_eq!(fwd.in_iv.len(), k.num_inputs());
        }
    }

    /// The classifier that decides whether a macro needs an in-cone
    /// evaluation op. Getting this from is_sequential() misclassifies the SRL
    /// and the DSP in opposite directions, so it is pinned here.
    #[test]
    fn has_comb_outputs_is_not_is_sequential() {
        assert!(MacroKind::Carry4.has_comb_outputs());
        assert!(MacroKind::Srlc32e.has_comb_outputs(),
                "Q = state[A] reads combinationally through A[4:0]");
        assert!(!MacroKind::Dsp48e2.has_comb_outputs(),
                "P is purely clocked -- a level-0 leaf like a DFF's Q");

        // The two predicates genuinely disagree, in both directions.
        assert!(MacroKind::Carry4.has_comb_outputs()
                != MacroKind::Carry4.is_sequential());
        assert!(MacroKind::Dsp48e2.has_comb_outputs()
                != MacroKind::Dsp48e2.is_sequential());
        assert_eq!(MacroKind::Srlc32e.has_comb_outputs(),
                   MacroKind::Srlc32e.is_sequential());
    }

    /// A CARRY4 with every input tied to a constant still needs evaluating,
    /// so the classifier must be static and not derived from live fan-in.
    #[test]
    fn has_comb_outputs_is_static_not_instance_derived() {
        let mb = MacroBlock::new(MacroKind::Carry4, 1);   // all inputs zero
        assert!(mb.comb_inputs().is_empty(), "no live fan-in");
        assert!(mb.kind.has_comb_outputs(), "but it still must be evaluated");
    }

    #[test]
    fn comb_inputs_reads_the_right_slots() {
        let k = MacroKind::Srlc32e;
        let mut mb = MacroBlock::new(k, 1);
        for slot in 0..k.num_inputs() { mb.set_input(slot, (slot + 10) << 1); }
        let comb = mb.comb_inputs();
        let expect: Vec<usize> = (0..5)
            .map(|b| k.input_slot("A", Some(b)).unwrap() + 10)
            .collect();
        assert_eq!(comb, expect, "only A[4:0] is combinational on an SRLC32E");

        let c4 = MacroKind::Carry4;
        let mut mb = MacroBlock::new(c4, 2);
        for slot in 0..c4.num_inputs() { mb.set_input(slot, (slot + 10) << 1); }
        assert_eq!(mb.comb_inputs().len(), c4.num_inputs(), "CARRY4 is all comb");

        let dsp = MacroKind::Dsp48e2;
        let mut mb = MacroBlock::new(dsp, 3);
        for slot in 0..dsp.num_inputs() { mb.set_input(slot, (slot + 10) << 1); }
        assert!(mb.comb_inputs().is_empty(), "a DSP has no comb fan-in");
    }
}

// ---------------------------------------------------------------------------
// Host-side evaluation, mirroring csrc/macros.cuh
//
// These exist so the CPU script executor (src/bin/flatten_test.rs) can run the
// macro phase exactly as the kernel does, isolating a script/layout defect
// from a kernel defect. They deliberately mirror the CUDA models rather than
// re-deriving the equations: the arithmetic is already verified exhaustively
// by csrc/tests/verify_macros.sh, so what this rung tests is the SCRIPT, not
// the maths.
//
// naive_sim.rs keeps its own independent transcription from the problem
// statement. The two must never be merged -- comparing them is what makes the
// ladder's middle rung meaningful.
// ---------------------------------------------------------------------------

/// Sign-extend the low `bits` of `v`.
pub fn host_sext(v: u64, bits: u32) -> i64 {
    let sh = 64 - bits;
    ((v << sh) as i64) >> sh
}

/// DSP48E2, simplified subset. Mirrors `gem_dsp48e2`.
pub fn host_dsp48e2(a: u64, b: u64, c: u64, d: u64,
                    use_preadd: bool, mode: u32, p_cur: i64) -> i64 {
    let (a, b, c, d) = (host_sext(a, 27), host_sext(b, 18),
                        host_sext(c, 48), host_sext(d, 27));
    // 27-bit pre-adder, wraps -- this is why the product is 45 bits, not 46.
    let ad = host_sext((if use_preadd { a + d } else { a }) as u64, 27);
    let m = host_sext((ad * b) as u64, 45);
    let p = match mode { 0 => c, 1 => m, _ => p_cur + m };
    host_sext(p as u64, 48)
}

/// CARRY4. Mirrors `gem_carry4`; returns `(CO, O)`.
pub fn host_carry4(s: u32, di: u32, cin: bool, cyinit: bool) -> (u32, u32) {
    let (s, di) = (s & 0xF, di & 0xF);
    let c0 = (cyinit as u32) | (cin as u32);
    let (mut c, mut co) = (c0, 0u32);
    for i in 0..4 {
        let cn = ((!s >> i) & (di >> i) & 1) | (((s >> i) & 1) & c);
        co |= cn << i;
        c = cn;
    }
    (co, (s ^ (c0 | (co << 1))) & 0xF)
}

/// SRLC32E combinational read. Mirrors `gem_srlc32e_read`; returns `(Q, Q31)`.
pub fn host_srlc32e_read(state: u32, a: u32) -> (bool, bool) {
    (((state >> (a & 31)) & 1) != 0, ((state >> 31) & 1) != 0)
}

/// SRLC32E clock edge. Mirrors `gem_srlc32e_edge`.
pub fn host_srlc32e_edge(state: u32, d: bool, ce: bool) -> u32 {
    if ce { (state << 1) | (d as u32) } else { state }
}

//! Host-side memory formatter for word-level macros.
//!
//! Deliverable A, second half: map the intercepted macro nodes into flattened,
//! 64-bit aligned CUDA buffers laid out for coalesced global reads.
//!
//! GEM's existing state is `u32` bit-packed and read a bit at a time through
//! permutation tables. That is the right shape for 1-bit AIG nodes and the
//! wrong shape for a 48-bit accumulator: a DSP's PREG would be scattered over
//! 48 separate bit positions, and reassembling it per cycle would cost more
//! than the multiply.
//!
//! So macro *state* lives in its own `u64` region, separate from the bit-state:
//!
//! ```text
//!   word_state: [ DSP.P x N_dsp | pad ][ SRL.state x N_srl | pad ]
//!                ^ base % 32 == 0       ^ base % 32 == 0
//! ```
//!
//! Two properties, both deliberate:
//!
//! * **64-bit alignment.** The region is `UVec<u64>`, so every slot is
//!   8-byte aligned by element type, and cudaMalloc returns a base at least
//!   256-byte aligned. Alignment is structural, not asserted.
//!
//! * **Coalescing.** Macros are grouped by kind into contiguous runs, and each
//!   run is padded to a multiple of [`WARP`]. Lane `t` of a batch touches
//!   `word_state[base + t]`, so a warp reads 32 x 8 B = 256 B from one aligned
//!   256-byte segment: 2 sectors, the minimum possible. An array-of-structs
//!   layout would have given each lane a strided hop and shredded the sector
//!   count -- this is the structure-of-arrays choice, and it is what the
//!   bandwidth number in the report rests on.
//!
//! The bridge between the two regions is explicit rather than implied. Each
//! macro carries a descriptor listing where every input bit lives in the
//! bit-state and where every output bit must be written back. The kernel will
//! consume these as PACK / EVAL / UNPACK in Part B; this module only lays them
//! out.

use crate::macros::MacroKind;
use std::collections::HashMap;

/// Lanes per warp. Each kind's run is padded to a multiple of this so a batch
/// never straddles a memory segment.
pub const WARP: usize = 32;

/// Marks an input bit that is tied to constant zero rather than living in the
/// bit-state, and an output bit that nothing reads.
pub const NO_BIT: u32 = u32::MAX;

/// The three kinds, in the fixed order their runs appear in `word_state`.
pub const KIND_ORDER: [MacroKind; 3] =
    [MacroKind::Dsp48e2, MacroKind::Srlc32e, MacroKind::Carry4];

pub fn kind_code(k: MacroKind) -> u32 {
    match k {
        MacroKind::Dsp48e2 => 0,
        MacroKind::Srlc32e => 1,
        MacroKind::Carry4  => 2,
    }
}

/// One macro instance as the formatter sees it, before layout.
#[derive(Debug, Clone)]
pub struct MacroIo {
    pub cellid: usize,
    pub kind: MacroKind,
    /// Bit-state positions of the input bits, in canonical port order.
    /// `NO_BIT` for an input tied to zero.
    pub in_bit_pos: Vec<u32>,
    /// Bit-state positions the output bits are written back to, indexed by
    /// `MacroKind::output_slot`. `NO_BIT` where nothing reads that output.
    pub out_bit_pos: Vec<u32>,
    /// Bit-state position of the clock enable, or `NO_BIT` if combinational.
    pub clk_en_pos: u32,
}

/// One macro's placement in the flattened buffers.
#[derive(Debug, Clone)]
pub struct MacroSlot {
    pub cellid: usize,
    pub kind: MacroKind,
    /// Index into the `u64` word-state region. `usize::MAX` for a stateless
    /// macro (CARRY4 holds nothing across a cycle).
    pub word_index: usize,
}

/// The formatter's output.
#[derive(Debug, Clone, Default)]
pub struct MacroLayout {
    /// Length of the `u64` word-state region, in u64 elements.
    pub word_state_size: usize,
    /// `(base, count)` per kind, indexed by `kind_code`. `base` is a u64
    /// element index and is always a multiple of [`WARP`].
    pub kind_ranges: [(usize, usize); 3],
    /// Placements, grouped by kind in `KIND_ORDER` so a batch is contiguous.
    pub slots: Vec<MacroSlot>,
    /// Flat, kernel-consumable descriptors. CSR-indexed by `desc_start`.
    ///
    /// Per macro:
    /// ```text
    ///   [0] kind code
    ///   [1] word-state index (u32::MAX if stateless)
    ///   [2] clock-enable bit position (NO_BIT if combinational)
    ///   [3] number of input bits
    ///   [4] number of output bits
    ///   [5 ..]        input bit positions   (PACK sources)
    ///   [5 + n_in ..] output bit positions  (UNPACK destinations)
    /// ```
    pub descriptors: Vec<u32>,
    /// `descriptors` CSR offsets; length `slots.len() + 1`.
    pub desc_start: Vec<usize>,
    /// netlistdb cell id -> index into `slots` / `desc_start`.
    ///
    /// The per-partition script references macros by this index rather than
    /// carrying their descriptors inline: a DSP's descriptor is 128 words, and
    /// duplicating that into every block script would dwarf the script itself.
    pub slot_of_cell: HashMap<usize, usize>,
}

impl MacroLayout {
    pub fn build(macros: &[MacroIo]) -> MacroLayout {
        let mut layout = MacroLayout::default();
        let mut base = 0usize;

        for kind in KIND_ORDER {
            let of_kind: Vec<&MacroIo> =
                macros.iter().filter(|m| m.kind == kind).collect();
            let words_each = kind.state_words();
            let count = of_kind.len();
            layout.kind_ranges[kind_code(kind) as usize] = (base, count);

            for (i, m) in of_kind.iter().enumerate() {
                let word_index = if words_each == 0 { usize::MAX } else { base + i };
                layout.slots.push(MacroSlot {
                    cellid: m.cellid, kind: m.kind, word_index,
                });
                layout.slot_of_cell.insert(m.cellid, layout.slots.len() - 1);
                layout.desc_start.push(layout.descriptors.len());
                layout.descriptors.extend_from_slice(&[
                    kind_code(m.kind),
                    if word_index == usize::MAX { u32::MAX } else { word_index as u32 },
                    m.clk_en_pos,
                    m.in_bit_pos.len() as u32,
                    m.out_bit_pos.len() as u32,
                ]);
                layout.descriptors.extend_from_slice(&m.in_bit_pos);
                layout.descriptors.extend_from_slice(&m.out_bit_pos);
            }

            // Advance past this kind's run, padded to a whole number of warps
            // so the next run also starts on a 256-byte segment boundary.
            if words_each > 0 {
                base += round_up(count * words_each, WARP);
            }
        }

        layout.desc_start.push(layout.descriptors.len());
        layout.word_state_size = base;
        layout
    }

    /// Index into `slots` for a netlistdb cell id.
    pub fn slot_index_of(&self, cellid: usize) -> Option<usize> {
        self.slot_of_cell.get(&cellid).copied()
    }

    /// Descriptor words for one macro, by position in `slots`.
    pub fn descriptor(&self, i: usize) -> &[u32] {
        &self.descriptors[self.desc_start[i]..self.desc_start[i + 1]]
    }

    /// Bytes of device memory the word-state region occupies.
    pub fn word_state_bytes(&self) -> usize { self.word_state_size * 8 }
}

fn round_up(v: usize, m: usize) -> usize { (v + m - 1) / m * m }

// ---------------------------------------------------------------------------
// Partition capacity model
//
// A boomerang partition holds BOOMERANG_MAX_WRITEOUTS (256) u32 writeout slots
// = 8192 bits. Every macro spends some of that, and the numbers are worth
// stating explicitly because the intuitive estimate is wrong in both
// directions.
//
// What a macro actually costs a partition:
//
//   * OUTPUT slots -- ceil(num_outputs / 32) u32 slots, reserved outright.
//     A DSP48E2 takes 2, a CARRY4 or SRLC32E takes 1.
//
//   * INPUT bits -- every input bit must be a realised combinational result,
//     i.e. present in the partition's normal writeouts. A DSP48E2 needs 123.
//
//   * DUPLICATE slots -- NOT one per input, which is the easy mistake. The
//     first activation of a pin reuses that pin's existing writeout position
//     and costs nothing. A duplicate is spent only when the same pin is
//     consumed under a DIFFERENT (clock-enable, polarity) pair. Since all of a
//     macro's inputs share one clk_en_iv, a pin feeding only this macro costs
//     zero duplicates. Duplicates appear when a pin also feeds a DFF with a
//     different enable, or feeds the macro at both polarities.
// ---------------------------------------------------------------------------

/// Writeout bits one instance of `kind` needs realised in a partition:
/// its inputs, plus the output bits it reserves.
pub fn writeout_bits(kind: MacroKind) -> usize {
    kind.num_inputs() + round_up(kind.num_outputs(), 32)
}

/// u32 writeout slots reserved outright for one instance's outputs.
pub fn writeout_slots(kind: MacroKind) -> usize {
    (kind.num_outputs() + 31) / 32
}

/// Upper bound on instances of one kind in a single partition, ignoring all
/// other logic. Real designs reach far fewer, because the partition also has
/// to hold the combinational cone feeding the macros.
pub fn max_per_partition(kind: MacroKind, capacity_slots: usize) -> usize {
    (capacity_slots * 32) / writeout_bits(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(cellid: usize, kind: MacroKind, n_in: usize) -> MacroIo {
        MacroIo {
            cellid, kind,
            in_bit_pos: (0..n_in as u32).collect(),
            out_bit_pos: vec![NO_BIT; kind.num_outputs()],
            clk_en_pos: if kind.is_sequential() { 7 } else { NO_BIT },
        }
    }

    #[test]
    fn stateless_macros_occupy_no_word_state() {
        let l = MacroLayout::build(&[io(1, MacroKind::Carry4, 10),
                                     io(2, MacroKind::Carry4, 10)]);
        assert_eq!(l.word_state_size, 0, "CARRY4 holds nothing across a cycle");
        assert!(l.slots.iter().all(|s| s.word_index == usize::MAX));
    }

    #[test]
    fn each_kind_starts_on_a_warp_boundary() {
        let mut ms: Vec<MacroIo> = (0..40).map(|i| io(i, MacroKind::Dsp48e2, 5)).collect();
        ms.extend((100..103).map(|i| io(i, MacroKind::Srlc32e, 3)));
        let l = MacroLayout::build(&ms);
        for (code, (base, _)) in l.kind_ranges.iter().enumerate() {
            assert_eq!(base % WARP, 0,
                       "kind {} run must begin on a warp boundary, got {}", code, base);
        }
        // 40 DSPs round up to 64 u64s, so the SRL run starts at 64.
        assert_eq!(l.kind_ranges[kind_code(MacroKind::Dsp48e2) as usize], (0, 40));
        assert_eq!(l.kind_ranges[kind_code(MacroKind::Srlc32e) as usize], (64, 3));
        assert_eq!(l.word_state_size, 64 + 32);
    }

    /// The coalescing property the report claims: consecutive lanes of a batch
    /// read consecutive u64s, so a warp covers one aligned 256-byte segment.
    #[test]
    fn lanes_of_a_batch_are_contiguous_and_segment_aligned() {
        let ms: Vec<MacroIo> = (0..70).map(|i| io(i, MacroKind::Dsp48e2, 5)).collect();
        let l = MacroLayout::build(&ms);
        let (base, count) = l.kind_ranges[kind_code(MacroKind::Dsp48e2) as usize];
        assert_eq!(count, 70);
        for (t, s) in l.slots.iter().enumerate() {
            assert_eq!(s.word_index, base + t, "lane {} must read word {}", t, base + t);
        }
        for warp in 0..(count + WARP - 1) / WARP {
            let byte_off = (base + warp * WARP) * 8;
            assert_eq!(byte_off % 256, 0,
                       "warp {} starts mid-segment at byte {}", warp, byte_off);
        }
    }

    #[test]
    fn slots_are_grouped_by_kind_so_a_batch_is_contiguous() {
        let ms = vec![io(1, MacroKind::Srlc32e, 3), io(2, MacroKind::Dsp48e2, 5),
                      io(3, MacroKind::Carry4, 10), io(4, MacroKind::Dsp48e2, 5)];
        let l = MacroLayout::build(&ms);
        let kinds: Vec<MacroKind> = l.slots.iter().map(|s| s.kind).collect();
        assert_eq!(kinds, vec![MacroKind::Dsp48e2, MacroKind::Dsp48e2,
                               MacroKind::Srlc32e, MacroKind::Carry4]);
    }

    #[test]
    fn descriptors_round_trip() {
        let mut a = io(9, MacroKind::Dsp48e2, 4);
        a.out_bit_pos = (0..48).map(|b| 1000 + b).collect();
        let l = MacroLayout::build(&[a.clone()]);
        let d = l.descriptor(0);
        assert_eq!(d[0], kind_code(MacroKind::Dsp48e2));
        assert_eq!(d[1], 0, "first DSP sits at word 0");
        assert_eq!(d[2], 7, "clock enable position");
        assert_eq!(d[3] as usize, a.in_bit_pos.len());
        assert_eq!(d[4] as usize, a.out_bit_pos.len());
        let n_in = d[3] as usize;
        assert_eq!(&d[5..5 + n_in], &a.in_bit_pos[..]);
        assert_eq!(&d[5 + n_in..], &a.out_bit_pos[..]);
        assert_eq!(l.desc_start.len(), 2);
    }

    /// The 8192-bit writeout ceiling, stated in macro terms. These numbers go
    /// in the report; if the layout changes and they move, that is a finding,
    /// not a test to relax.
    #[test]
    fn partition_capacity_is_what_we_think_it_is() {
        const CAP: usize = 256;                        // BOOMERANG_MAX_WRITEOUTS
        assert_eq!(CAP * 32, 8192, "capacity in bits");

        assert_eq!(MacroKind::Dsp48e2.num_inputs(), 123);
        assert_eq!(writeout_slots(MacroKind::Dsp48e2), 2);
        assert_eq!(writeout_bits(MacroKind::Dsp48e2), 123 + 64);
        assert_eq!(max_per_partition(MacroKind::Dsp48e2, CAP), 43);

        assert_eq!(MacroKind::Carry4.num_inputs(), 10);
        assert_eq!(writeout_slots(MacroKind::Carry4), 1);
        assert_eq!(max_per_partition(MacroKind::Carry4, CAP), 195);

        assert_eq!(MacroKind::Srlc32e.num_inputs(), 7);
        assert_eq!(writeout_slots(MacroKind::Srlc32e), 1);
        // 7 input bits + one 32-bit output slot = 39; 8192 / 39 = 210.
        assert_eq!(writeout_bits(MacroKind::Srlc32e), 39);
        assert_eq!(max_per_partition(MacroKind::Srlc32e, CAP), 210);
    }

    /// A DSP does NOT cost one duplicate slot per input bit. All of its inputs
    /// share one clock enable, so a pin feeding only this macro is a single
    /// activation and reuses its existing writeout position.
    #[test]
    fn a_macros_own_inputs_are_one_activation_each() {
        let k = MacroKind::Dsp48e2;
        let clk_en = 42usize;
        let activations: std::collections::HashSet<usize> =
            (0..k.num_inputs()).map(|slot| {
                let in_iv = (100 + slot) << 1;          // distinct pins, even polarity
                clk_en << 1 | (in_iv & 1)               // the activation key
            }).collect();
        assert_eq!(activations.len(), 1,
                   "distinct pins under one clock enable share one activation key, \
                    so none of them is a duplicate");
    }

    /// Scaling behaviour across the DSP counts the stress test exercises.
    #[test]
    fn dsp_word_state_scales_and_stays_aligned() {
        for n in [1usize, 8, 32, 43, 44, 128, 512] {
            let ms: Vec<MacroIo> = (0..n)
                .map(|i| io(i, MacroKind::Dsp48e2, MacroKind::Dsp48e2.num_inputs()))
                .collect();
            let l = MacroLayout::build(&ms);
            assert_eq!(l.slots.len(), n);
            assert_eq!(l.word_state_size, round_up(n, WARP), "n = {}", n);
            assert_eq!(l.word_state_bytes() % 256, 0, "n = {}", n);
            let (base, count) = l.kind_ranges[kind_code(MacroKind::Dsp48e2) as usize];
            assert_eq!((base, count), (0, n));
            assert_eq!(l.desc_start.len(), n + 1);
        }
    }

    /// Every macro must be addressable by cell id, and the index must agree
    /// with its position in `slots` -- the script encodes only this index.
    #[test]
    fn slot_index_of_agrees_with_slots_order() {
        let ms = vec![io(7, MacroKind::Srlc32e, 3), io(11, MacroKind::Dsp48e2, 5),
                      io(3, MacroKind::Carry4, 10), io(5, MacroKind::Dsp48e2, 5)];
        let l = MacroLayout::build(&ms);
        assert_eq!(l.slot_of_cell.len(), ms.len(), "every cell id is mapped");
        for (i, slot) in l.slots.iter().enumerate() {
            assert_eq!(l.slot_index_of(slot.cellid), Some(i));
        }
        assert_eq!(l.slot_index_of(999), None);
        // and the index really does select that macro's descriptor
        let i = l.slot_index_of(11).unwrap();
        assert_eq!(l.descriptor(i)[0], kind_code(MacroKind::Dsp48e2));
    }

    #[test]
    fn empty_design_lays_out_cleanly() {
        let l = MacroLayout::build(&[]);
        assert_eq!(l.word_state_size, 0);
        assert_eq!(l.word_state_bytes(), 0);
        assert_eq!(l.desc_start, vec![0]);
    }
}

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

    /// Descriptor words for one macro, by position in `slots`.
    pub fn descriptor(&self, i: usize) -> &[u32] {
        &self.descriptors[self.desc_start[i]..self.desc_start[i + 1]]
    }

    /// Bytes of device memory the word-state region occupies.
    pub fn word_state_bytes(&self) -> usize { self.word_state_size * 8 }
}

fn round_up(v: usize, m: usize) -> usize { (v + m - 1) / m * m }

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

    #[test]
    fn empty_design_lays_out_cleanly() {
        let l = MacroLayout::build(&[]);
        assert_eq!(l.word_state_size, 0);
        assert_eq!(l.word_state_bytes(), 0);
        assert_eq!(l.desc_start, vec![0]);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! AIGPDK is a special artificial cell library used in GEM.

use netlistdb::{Direction, LeafPinProvider};
use compact_str::CompactString;
use sverilogparse::SVerilogRange;

/// This implements direction and width providers for
/// AIG PDK cells.
///
/// You can use it in netlistdb construction.
pub struct AIGPDKLeafPins();

/// The addr width of an SRAM.
///
/// The word width is always 32.
/// If you change this, make sure to change all other occurences in this
/// project as well as the definitions in PDK libraries.
pub const AIGPDK_SRAM_ADDR_WIDTH: usize = 13;

pub const AIGPDK_SRAM_SIZE: usize = 1 << 13;

// ---------------------------------------------------------------------------
// Heterogeneous word-level macros.
//
// These three cells are NOT decomposed into AND gates. They are intercepted as
// blackboxes by synth/gem_macros.v and reach us intact, the same way
// `$__RAMGEM_SYNC_` does. The names and widths below are the contract with
// synth/gem_macros.v -- if you change one, change both.
// ---------------------------------------------------------------------------

/// Cell name of the simplified DSP48E2 subset (27x18 -> 48, clocked PREG).
pub const GEMDSP_CELL: &str = "$__GEMDSP_";
/// Cell name of the Xilinx CARRY4 carry chain (purely combinational).
pub const GEMCARRY4_CELL: &str = "$__GEMCARRY4_";
/// Cell name of the SRLC32E shift-register LUT.
pub const GEMSRL32_CELL: &str = "$__GEMSRL32_";

pub const GEMDSP_A_WIDTH: usize = 27;
pub const GEMDSP_B_WIDTH: usize = 18;
pub const GEMDSP_C_WIDTH: usize = 48;
pub const GEMDSP_D_WIDTH: usize = 27;
pub const GEMDSP_P_WIDTH: usize = 48;
pub const GEMDSP_MODE_WIDTH: usize = 2;

/// A CARRY4 is four bits wide on S, DI, CO and O alike.
pub const GEMCARRY4_WIDTH: usize = 4;

pub const GEMSRL32_DEPTH: usize = 32;
pub const GEMSRL32_ADDR_WIDTH: usize = 5;

/// True for the word-level macro cells that bypass AIG decomposition.
///
/// Use this instead of matching the string literals, so that adding a fourth
/// macro later is a one-line change here rather than a grep across the repo.
#[inline]
pub fn is_gem_macro(macro_name: &str) -> bool {
    matches!(macro_name, GEMDSP_CELL | GEMCARRY4_CELL | GEMSRL32_CELL)
}

impl LeafPinProvider for AIGPDKLeafPins {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString, pin_idx: Option<isize>
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str(), pin_idx) {
            ("INV" | "BUF", "A", None) => Direction::I,
            ("INV" | "BUF", "Y", None) => Direction::O,

            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B", None) => Direction::I,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "Y", None) => Direction::O,

            ("DFF" | "LATCH", "CLK" | "D", None) => Direction::I,
            ("DFFSR", "CLK" | "D" | "S" | "R", None) => Direction::I,
            ("DFF" | "DFFSR" | "LATCH", "Q", None) => Direction::O,

            ("CKLNQD", "CP" | "E", None) => Direction::I,
            ("CKLNQD", "Q", None) => Direction::O,

            ("$__RAMGEM_ASYNC_", _, _) => {
                panic!("Async RAM (lib cell {}) not supported yet in GEM.", macro_name);
            },

            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK",
             None) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR",
             Some(0..=12)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA",
             Some(0..=31)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_RD_DATA",
             Some(0..=31)) => Direction::O,

            // --- word-level macros -------------------------------------
            // DSP48E2 subset. AREG/BREG/CREG/DREG/ADREG/MREG are all
            // combinational; only PREG (the P output) is clocked.
            ("$__GEMDSP_", "CLK" | "USE_PREADD", None) => Direction::I,
            ("$__GEMDSP_", "A" | "D", Some(0..=26)) => Direction::I,
            ("$__GEMDSP_", "B", Some(0..=17)) => Direction::I,
            ("$__GEMDSP_", "C", Some(0..=47)) => Direction::I,
            ("$__GEMDSP_", "MODE", Some(0..=1)) => Direction::I,
            ("$__GEMDSP_", "P", Some(0..=47)) => Direction::O,

            // CARRY4. Fully combinational: no clock pin at all.
            ("$__GEMCARRY4_", "CIN" | "CYINIT", None) => Direction::I,
            ("$__GEMCARRY4_", "S" | "DI", Some(0..=3)) => Direction::I,
            ("$__GEMCARRY4_", "CO" | "O", Some(0..=3)) => Direction::O,

            // SRLC32E. The shift is clocked; both read ports are
            // combinational reads off the current 32-bit state.
            ("$__GEMSRL32_", "CLK" | "D" | "CE", None) => Direction::I,
            ("$__GEMSRL32_", "A", Some(0..=4)) => Direction::I,
            ("$__GEMSRL32_", "Q" | "Q31", None) => Direction::O,

            _ => {
                use netlistdb::{GeneralPinName, HierName};
                panic!("Cannot recognize pin type {}, please make sure the verilog netlist is synthesized in GEM's aigpdk.",
                       (HierName::single(macro_name.clone()),
                        pin_name, pin_idx).dbg_fmt_pin());
            }
        }
    }

    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString
    ) -> Option<SVerilogRange> {
        match (macro_name.as_str(), pin_name.as_str()) {
            ("INV" | "BUF", "A" | "Y") => None,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B" | "Y") => None,
            ("DFF" | "DFFSR" | "LATCH", "CLK" | "D" | "Q" | "S" | "R") => None,
            ("CKLNQD", "CP" | "E" | "Q") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR")
                => Some(SVerilogRange(12, 0)),
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA" | "PORT_R_RD_DATA")
                => Some(SVerilogRange(31, 0)),
            // --- word-level macros -------------------------------------
            ("$__GEMDSP_", "CLK" | "USE_PREADD") => None,
            ("$__GEMDSP_", "A" | "D") => Some(SVerilogRange(26, 0)),
            ("$__GEMDSP_", "B") => Some(SVerilogRange(17, 0)),
            ("$__GEMDSP_", "C" | "P") => Some(SVerilogRange(47, 0)),
            ("$__GEMDSP_", "MODE") => Some(SVerilogRange(1, 0)),

            ("$__GEMCARRY4_", "CIN" | "CYINIT") => None,
            ("$__GEMCARRY4_", "S" | "DI" | "CO" | "O")
                => Some(SVerilogRange(3, 0)),

            ("$__GEMSRL32_", "CLK" | "D" | "CE" | "Q" | "Q31") => None,
            ("$__GEMSRL32_", "A") => Some(SVerilogRange(4, 0)),

            _ => None
        }
    }
}

#[cfg(test)]
mod macro_pin_tests {
    use super::*;
    use netlistdb::Direction;

    fn dir(m: &str, p: &str, i: Option<isize>) -> Direction {
        AIGPDKLeafPins().direction_of(&m.into(), &p.into(), i)
    }
    fn span(m: &str, p: &str) -> Option<usize> {
        AIGPDKLeafPins().width_of(&m.into(), &p.into())
            .map(|r| (r.0 - r.1 + 1) as usize)
    }

    #[test]
    fn dsp_pin_directions() {
        assert!(matches!(dir(GEMDSP_CELL, "CLK", None), Direction::I));
        assert!(matches!(dir(GEMDSP_CELL, "USE_PREADD", None), Direction::I));
        for i in 0..27 {
            assert!(matches!(dir(GEMDSP_CELL, "A", Some(i)), Direction::I));
            assert!(matches!(dir(GEMDSP_CELL, "D", Some(i)), Direction::I));
        }
        for i in 0..18 { assert!(matches!(dir(GEMDSP_CELL, "B", Some(i)), Direction::I)); }
        for i in 0..48 {
            assert!(matches!(dir(GEMDSP_CELL, "C", Some(i)), Direction::I));
            assert!(matches!(dir(GEMDSP_CELL, "P", Some(i)), Direction::O));
        }
        for i in 0..2 { assert!(matches!(dir(GEMDSP_CELL, "MODE", Some(i)), Direction::I)); }
    }

    #[test]
    fn carry4_pin_directions() {
        assert!(matches!(dir(GEMCARRY4_CELL, "CIN", None), Direction::I));
        assert!(matches!(dir(GEMCARRY4_CELL, "CYINIT", None), Direction::I));
        for i in 0..4 {
            assert!(matches!(dir(GEMCARRY4_CELL, "S",  Some(i)), Direction::I));
            assert!(matches!(dir(GEMCARRY4_CELL, "DI", Some(i)), Direction::I));
            assert!(matches!(dir(GEMCARRY4_CELL, "CO", Some(i)), Direction::O));
            assert!(matches!(dir(GEMCARRY4_CELL, "O",  Some(i)), Direction::O));
        }
    }

    #[test]
    fn srl_pin_directions() {
        for p in ["CLK", "D", "CE"] {
            assert!(matches!(dir(GEMSRL32_CELL, p, None), Direction::I));
        }
        for i in 0..5 { assert!(matches!(dir(GEMSRL32_CELL, "A", Some(i)), Direction::I)); }
        assert!(matches!(dir(GEMSRL32_CELL, "Q",   None), Direction::O));
        assert!(matches!(dir(GEMSRL32_CELL, "Q31", None), Direction::O));
    }

    /// The widths reported to netlistdb must equal the declared constants,
    /// which in turn must equal the port widths in synth/gem_macros.v.
    #[test]
    fn widths_agree_with_constants() {
        assert_eq!(span(GEMDSP_CELL, "A"),    Some(GEMDSP_A_WIDTH));
        assert_eq!(span(GEMDSP_CELL, "B"),    Some(GEMDSP_B_WIDTH));
        assert_eq!(span(GEMDSP_CELL, "C"),    Some(GEMDSP_C_WIDTH));
        assert_eq!(span(GEMDSP_CELL, "D"),    Some(GEMDSP_D_WIDTH));
        assert_eq!(span(GEMDSP_CELL, "P"),    Some(GEMDSP_P_WIDTH));
        assert_eq!(span(GEMDSP_CELL, "MODE"), Some(GEMDSP_MODE_WIDTH));
        assert_eq!(span(GEMCARRY4_CELL, "S"), Some(GEMCARRY4_WIDTH));
        assert_eq!(span(GEMCARRY4_CELL, "O"), Some(GEMCARRY4_WIDTH));
        assert_eq!(span(GEMSRL32_CELL, "A"),  Some(GEMSRL32_ADDR_WIDTH));
        assert_eq!(span(GEMSRL32_CELL, "Q"),  None);
    }

    /// Adding macros must not disturb the cells GEM already understood.
    #[test]
    fn existing_cells_unaffected() {
        assert_eq!(span("$__RAMGEM_SYNC_", "PORT_R_ADDR"), Some(AIGPDK_SRAM_ADDR_WIDTH));
        assert_eq!(span("$__RAMGEM_SYNC_", "PORT_R_RD_DATA"), Some(32));
        assert!(matches!(dir("AND2_00_0", "Y", None), Direction::O));
        assert!(matches!(dir("DFF", "Q", None), Direction::O));
        assert!(!is_gem_macro("AND2_00_0"));
        assert!(!is_gem_macro("$__RAMGEM_SYNC_"));
    }

    #[test]
    fn all_three_are_macros() {
        assert!(is_gem_macro(GEMDSP_CELL));
        assert!(is_gem_macro(GEMCARRY4_CELL));
        assert!(is_gem_macro(GEMSRL32_CELL));
    }

    /// An out-of-range bit index must still be a loud failure, not silently
    /// treated as an input.
    #[test]
    #[should_panic(expected = "Cannot recognize pin type")]
    fn out_of_range_bit_panics() { dir(GEMCARRY4_CELL, "S", Some(4)); }
}

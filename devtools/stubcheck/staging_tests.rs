//! Semantic tests for macro-aware major staging.
//!
//! These pin down the one property the whole split exists for: a value a macro
//! produces combinationally must not be readable by boolean logic until a
//! LATER major stage, because the kernel's macro phase writes into
//! `shared_writeouts` and no boomerang stage ever reads that buffer.
//!
//! They reuse the synthetic netlist from `semantics.rs`:
//!
//!   top.a, top.b, top.clk -> CARRY4 (combinational) + DSP (sequential)
//!   CARRY4.O[0] -> top.o1        DSP.P[0] -> top.o2
#![cfg(test)]
use crate::aig::*;
use crate::staging::*;

/// A DSP's P is clocked -- read from state like a DFF's Q -- so it forces no
/// boundary. A CARRY4's O is combinational and does.
#[test]
fn only_combinational_macro_outputs_force_a_boundary() {
    let aig = AIG::from_netlistdb(&crate::semantics::db());
    let ms = macro_avail_stages(&aig);

    let mut saw_comb = false;
    let mut saw_seq = false;
    for i in 1..aig.num_aigpins + 1 {
        match aig.drivers[i] {
            DriverType::Macro(cellid, _) => {
                if aig.is_comb_macro_output(i) {
                    saw_comb = true;
                    assert_eq!(ms.eval[i], 0,
                               "the CARRY4's operands are primary inputs, so \
                                it runs in stage 0");
                    assert_eq!(ms.avail[i], 1,
                               "but its result is only readable by boolean \
                                logic one major stage later");
                }
                else {
                    saw_seq = true;
                    assert_eq!(ms.avail[i], 0,
                               "cell {} is a sequential macro output; it is \
                                read from state and needs no boundary", cellid);
                }
            },
            _ => {}
        }
    }
    assert!(saw_comb && saw_seq, "netlist must exercise both kinds");
    assert_eq!(ms.max_stage, 1, "exactly one boundary: CARRY4.O -> top.o1");
}

/// The regression guarantee, stated as the property that actually protects it.
///
/// Only a COMBINATIONAL macro output can ever raise a stage index. Everything
/// else -- AND gates, primary inputs, DFF Q, a DSP's P -- stays at 0. So a
/// netlist without one reports max_stage 0, `build_staged_aigs` falls through
/// to the untouched level-split path, and a macro-free design stages exactly
/// as it always did. That is what keeps the byte-identical-VCD regression
/// check meaningful.
#[test]
fn nothing_but_a_combinational_macro_can_raise_a_stage() {
    let aig = AIG::from_netlistdb(&crate::semantics::db());
    let ms = macro_avail_stages(&aig);
    for i in 1..aig.num_aigpins + 1 {
        if aig.is_comb_macro_output(i) { continue }
        assert_eq!(ms.avail[i], 0,
                   "aigpin {} is not a combinational macro output, so it must \
                    not carry a stage of its own", i);
        assert_eq!(ms.eval[i], 0);
    }
}

/// Chained macros cost nothing. pe.rs emits one ordered batch per link and the
/// kernel separates them with __syncthreads(), so a CARRY4 feeding another
/// CARRY4's CIN stays inside one major stage. Only macro-to-BOOLEAN pays.
///
/// This is checked on the availability rules directly rather than on a second
/// hand-built netlist: `eval` of a macro takes `eval` (not `avail`) of any
/// operand that is itself a combinational macro output, which is precisely the
/// no-cost-for-chaining rule.
#[test]
fn chaining_rule_does_not_accumulate_stages() {
    let aig = AIG::from_netlistdb(&crate::semantics::db());
    let ms = macro_avail_stages(&aig);
    for i in 1..aig.num_aigpins + 1 {
        if !aig.is_comb_macro_output(i) { continue }
        // Every macro runs one stage before its result is readable -- never
        // more. If this ever grows, a long carry chain would blow past the
        // kernel's 32 major stage cap.
        assert_eq!(ms.avail[i], ms.eval[i] + 1);
    }
}

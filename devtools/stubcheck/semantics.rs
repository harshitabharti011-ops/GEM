//! Semantic tests for the macro path, on a synthetic netlist.
//!
//! Types compiling proves nothing about whether a CARRY4 landed INSIDE the
//! combinational cone. These build a tiny netlist by hand and run the real
//! `AIG::from_netlistdb`, traversal and fanout-CSR code over it:
//!
//!   top.a, top.b, top.clk -> CARRY4 (combinational) + DSP (sequential)
//!   CARRY4.O[0] -> top.o1        DSP.P[0] -> top.o2
#![cfg(test)]
use crate::stubs::netlistdb::*;
use crate::aig::*;
use crate::macros::MacroKind;

fn h(s: &str) -> HierName { HierName(s.to_string()) }
pub fn db() -> NetlistDB {
    let pn = |n: &str, b: Option<isize>| (h(n), n.to_string(), b);
    NetlistDB {
        num_pins: 14, num_cells: 3,
        pin2cell: vec![0,0,0,0,0, 1,1,1,1,1, 2,2,2,2],
        pin2net:  vec![0,1,2,3,4, 0,1,5,5,3, 0,1,2,4],
        celltypes: vec!["".into(), "$__GEMCARRY4_".into(), "$__GEMDSP_".into()],
        cellnames: vec![h("top"), h("u_c4"), h("u_dsp")],
        pindirect: vec![
            Direction::O, Direction::O, Direction::O, Direction::I, Direction::I,
            Direction::I, Direction::I, Direction::I, Direction::I, Direction::O,
            Direction::I, Direction::I, Direction::I, Direction::O],
        pinnames: vec![
            pn("a",None), pn("b",None), pn("clk",None), pn("o1",None), pn("o2",None),
            pn("S",Some(0)), pn("DI",Some(0)), pn("CIN",None), pn("CYINIT",None), pn("O",Some(0)),
            pn("A",Some(0)), pn("B",Some(0)), pn("CLK",None), pn("P",Some(0))],
        // net2pin lists the DRIVER first for each net -- from_netlistdb relies on it
        net2pin: Csr { start: vec![0,3,6,8,10,12,14],
                       items: vec![0,5,10, 1,6,11, 2,12, 9,3, 13,4, 7,8] },
        cell2pin: Csr { start: vec![0,5,10,14], items: (0..14).collect() },
        net_zero: Some(5), net_one: None,
    }
}

/// The same netlist, but the DSP's pins are visited in reverse order. Yosys is
/// free to emit an instance's pins in any order, so nothing downstream may
/// depend on it.
fn db_reversed_macro_pins() -> NetlistDB {
    let mut d = db();
    let mut items: Vec<usize> = (0..14).collect();
    items[10..14].reverse();          // cell 2 (the DSP): A, B, CLK, P
    items[5..10].reverse();           // cell 1 (the CARRY4)
    d.cell2pin = Csr { start: vec![0,5,10,14], items };
    d
}

#[test]
fn carry4_is_combinational_dsp_is_sequential() {
    let aig = AIG::from_netlistdb(&db());
    assert_eq!(aig.macros.len(), 2, "both macros recorded");
    let c4  = aig.macros.get(&1).expect("carry4 present");
    let dsp = aig.macros.get(&2).expect("dsp present");
    assert_eq!(c4.kind,  MacroKind::Carry4);
    assert_eq!(dsp.kind, MacroKind::Dsp48e2);
    // in_iv is now slot-length for every instance; connectivity shows up as
    // non-zero entries, and comb/seq is a static property of the slot.
    assert_eq!(c4.in_iv.len(), MacroKind::Carry4.num_inputs());
    assert!((0..MacroKind::Carry4.num_inputs())
            .all(|s| MacroKind::Carry4.is_comb_slot(s)),
            "every CARRY4 input is combinational");
    // S[0] and DI[0] are driven; CIN and CYINIT are tied to the zero net and
    // so are filtered out of the fan-in.
    assert_eq!(c4.comb_inputs().len(), 2, "S[0] and DI[0] are the live fan-in");
    assert!(!c4.is_sequential());

    assert_eq!(dsp.in_iv.len(), MacroKind::Dsp48e2.num_inputs());
    assert!(dsp.comb_inputs().is_empty(), "DSP outputs have no comb fan-in");
    assert_eq!(dsp.in_iv.iter().filter(|&&iv| iv > 1).count(), 2, "A[0], B[0]");
    assert!(dsp.is_sequential());
}

#[test]
fn only_stateful_macros_are_endpoints() {
    let aig = AIG::from_netlistdb(&db());
    assert_eq!(aig.seq_macro_ids, vec![2],
               "DSP only -- a CARRY4 must never become an endpoint, or every \
                carry chain forces a stage boundary");
    let mut macro_endpoints = 0;
    for i in 0..aig.num_endpoint_groups() {
        if let EndpointGroup::Macro(mb) = aig.get_endpoint_group(i) {
            macro_endpoints += 1;
            assert!(mb.is_sequential());
        }
    }
    assert_eq!(macro_endpoints, 1);
}

/// The invariant everything downstream depends on.
#[test]
fn carry4_output_is_topologically_after_its_inputs() {
    let aig = AIG::from_netlistdb(&db());
    let order = aig.topo_traverse_generic(None, None);
    let pos = |p: usize| order.iter().position(|&x| x == p)
        .unwrap_or_else(|| panic!("aigpin {} missing from traversal", p));
    let c4 = aig.macros.get(&1).unwrap();
    let out = c4.outputs[4];                      // slot 4 == O[0]
    assert_ne!(out, 0, "O[0] should be driven");
    for inp in c4.comb_inputs() {
        assert!(pos(inp) < pos(out),
                "input aigpin {} must precede CARRY4 output {}", inp, out);
    }
}

/// Without these edges the partitioner cannot see through a CARRY4 at all.
#[test]
fn macro_comb_fanin_appears_in_the_fanout_csr() {
    let aig = AIG::from_netlistdb(&db());
    let c4 = aig.macros.get(&1).unwrap();
    let out = c4.outputs[4];
    for inp in c4.comb_inputs() {
        let fo = &aig.fanouts[aig.fanouts_start[inp]..aig.fanouts_start[inp + 1]];
        assert!(fo.contains(&out),
                "aigpin {} fanout {:?} should contain CARRY4 output {}", inp, fo, out);
    }
    assert!(aig.macros.get(&2).unwrap().comb_inputs().is_empty(),
            "a DSP contributes no combinational fanout edges");
}

/// Canonical operand ordering: the descriptors Part B will consume must not
/// depend on the order Yosys happened to emit the instance's pins.
#[test]
fn macro_inputs_are_canonically_ordered_regardless_of_pin_order() {
    let a = AIG::from_netlistdb(&db());
    let b = AIG::from_netlistdb(&db_reversed_macro_pins());

    for cellid in [1usize, 2usize] {
        let ma = a.macros.get(&cellid).expect("macro present in a");
        let mb = b.macros.get(&cellid).expect("macro present in b");
        assert_eq!(ma.kind, mb.kind);
        assert_eq!(ma.in_iv, mb.in_iv,
                   "cell {} in_iv changed with pin visit order", cellid);
        assert_eq!(ma.outputs, mb.outputs,
                   "cell {} outputs changed with pin visit order", cellid);
        assert_eq!(ma.in_iv.len(), ma.kind.num_inputs(),
                   "in_iv must always be slot-length");
    }
}

/// And the slots hold what their names say: the DSP's A[0] and B[0] land in the
/// A and B slots, not wherever iteration happened to put them.
#[test]
fn macro_input_slots_hold_the_named_operands() {
    let aig = AIG::from_netlistdb(&db());
    let dsp = aig.macros.get(&2).unwrap();
    let k = dsp.kind;
    let a0 = dsp.in_iv[k.input_slot("A", Some(0)).unwrap()];
    let b0 = dsp.in_iv[k.input_slot("B", Some(0)).unwrap()];
    assert_ne!(a0, 0, "A[0] is driven by top.a");
    assert_ne!(b0, 0, "B[0] is driven by top.b");
    assert_ne!(a0, b0, "A and B must not alias");
    // C, D, MODE and USE_PREADD are unconnected in this netlist.
    assert_eq!(dsp.in_iv[k.input_slot("C", Some(0)).unwrap()], 0);
    assert_eq!(dsp.in_iv[k.input_slot("MODE", Some(0)).unwrap()], 0);
}

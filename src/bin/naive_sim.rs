// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::hash::Hash;
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use compact_str::CompactString;
use netlistdb::{Direction, GeneralHierName, GeneralPinName, NetlistDB};
use sverilogparse::SVerilogRange;
use itertools::Itertools;
use gem::macros::MacroKind;

// ---------------------------------------------------------------------------
// Word-level macro evaluation for the CPU reference simulator.
//
// naive_sim is the bottom rung of the verification ladder: it evaluates the
// NETLIST directly, with no script, no partitioning and no boomerang
// scheduling, so it shares essentially no machinery with the GPU path. Without
// macro support it can only validate the shredded baseline, which leaves the
// macro-preserving path with no CPU-side ground truth at all.
//
// The equations below are transcribed from the problem statement, in Rust, and
// are deliberately NOT a port of csrc/macros.cuh -- an independent
// transcription is the whole point of a reference model.
// ---------------------------------------------------------------------------

fn sext(v: u64, bits: u32) -> i64 {
    let sh = 64 - bits;
    ((v << sh) as i64) >> sh
}

fn eval_dsp48e2(a: u64, b: u64, c: u64, d: u64,
                use_preadd: bool, mode: u64, p_cur: i64) -> i64 {
    let (a, b, c, d) = (sext(a, 27), sext(b, 18), sext(c, 48), sext(d, 27));
    // The pre-adder is 27 bits wide and WRAPS; that is why the product is 45
    // bits rather than 46.
    let ad = sext((if use_preadd { a + d } else { a }) as u64, 27);
    let m = sext((ad * b) as u64, 45);
    let p = match mode { 0 => c, 1 => m, _ => p_cur + m };
    sext(p as u64, 48)
}

fn eval_carry4(s: u64, di: u64, cin: u64, cyinit: u64) -> (u64, u64) {
    let mut c = (cyinit | cin) & 1;
    let (mut co, mut o) = (0u64, 0u64);
    for i in 0..4 {
        let (si, dii) = ((s >> i) & 1, (di >> i) & 1);
        o |= (si ^ c) << i;
        c = (si & c) | ((si ^ 1) & dii);
        co |= c << i;
    }
    (co, o)
}

/// Per-instance macro state: a DSP's PREG, an SRLC32E's 32-bit shift register.
#[derive(Default, Clone, Copy)]
struct MacroState { p: i64, sr: u32 }

/// Assemble a multi-bit input port of `cellid` from the pin-level state.
fn gather_port(netlistdb: &NetlistDB, circ_state: &[u8],
               cellid: usize, port: &str) -> u64 {
    let mut v = 0u64;
    for pinid in netlistdb.cell2pin.iter_set(cellid) {
        if netlistdb.pinnames[pinid].1.as_str() == port {
            let bit = netlistdb.pinnames[pinid].2.unwrap_or(0).max(0) as u32;
            v |= (circ_state[pinid] as u64) << bit;
        }
    }
    v
}
use vcd_ng::{Parser, ScopeItem, Var, Scope, FastFlow, FastFlowToken, FFValueChange, Writer, SimulationCommand};
use gem::aigpdk::AIGPDKLeafPins;

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// VCD input signal path
    input_vcd: String,
    /// The scope path of top module in the input VCD.
    ///
    /// If not specified, we will use a flat view.
    /// (this view is often incorrect..)
    #[clap(long)]
    input_vcd_scope: Option<String>,
    /// Output VCD path (must be writable)
    output_vcd: String,
    /// The scope path of top module in the output VCD.
    ///
    /// If not specified, we will use `gem_top_module`.
    #[clap(long)]
    output_vcd_scope: Option<String>,
    /// Whether to output wire states as well (for more verbose debugging)
    #[clap(long)]
    include_wires: bool,
}

/// Hierarchical name representation in VCD.
#[derive(PartialEq, Eq, Clone, Debug)]
struct VCDHier {
    cur: CompactString,
    prev: Option<Rc<VCDHier>>
}

/// Reverse iterator of a [`VCDHier`], yielding cell names
/// from the bottom to the top module.
struct VCDHierRevIter<'i>(Option<&'i VCDHier>);

impl<'i> Iterator for VCDHierRevIter<'i> {
    type Item = &'i CompactString;

    #[inline]
    fn next(&mut self) -> Option<&'i CompactString> {
        let name = self.0?;
        if name.cur.is_empty() {
            return None
        }
        let ret = &name.cur;
        self.0 = name.prev.as_ref().map(|a| a.as_ref());
        Some(ret)
    }
}

impl<'i> IntoIterator for &'i VCDHier {
    type Item = &'i CompactString;
    type IntoIter = VCDHierRevIter<'i>;

    #[inline]
    fn into_iter(self) -> VCDHierRevIter<'i> {
        VCDHierRevIter(Some(self))
    }
}

impl Hash for VCDHier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for s in self.iter() {
            s.hash(state);
        }
    }
}

#[allow(dead_code)]
impl VCDHier {
    #[inline]
    fn single(cur: CompactString) -> Self {
        VCDHier { cur, prev: None }
    }

    #[inline]
    fn empty() -> Self {
        VCDHier { cur: "".into(), prev: None }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.cur.as_str() == "" && self.prev.is_none()
    }

    #[inline]
    fn iter(&self) -> VCDHierRevIter {
        (&self).into_iter()
    }
}

/// Try to match one component in a scope.
/// If succeed, returns the remaining scope (can be None itself indicating
/// all paths matched).
/// If fails, return None.
fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
    if scope.len() == 0 { return Some("") }
    if scope.starts_with('/') {
        scope = &scope[1..];
    }
    if scope.len() == 0 { Some("") }
    else if scope.starts_with(cur) {
        if scope.len() == cur.len() { Some("") }
        else if scope.as_bytes()[cur.len()] == b'/' {
            Some(&scope[cur.len() + 1..])
        }
        else { None }
    }
    else { None }
}

fn find_top_scope<'i>(
    items: &'i [ScopeItem], top_scope: &'_ str
) -> Option<&'i Scope> {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            if let Some(s1) = match_scope_path(
                top_scope, scope.identifier.as_str()
            ) {
                return match s1 {
                    "" => Some(scope),
                    _ => find_top_scope(&scope.children[..], s1)
                };
            }
        }
    }
    None
}

fn main() {
    clilog::init_stderr_color_debug();
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

    let mut posedge_monitor = HashSet::new();
    for cellid in 1..netlistdb.num_cells {
        if matches!(netlistdb.celltypes[cellid].as_str(),
                    "DFF" | "$__RAMGEM_SYNC_") {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "CLK" | "PORT_R_CLK" | "PORT_W_CLK") {
                    let netid = netlistdb.pin2net[pinid];
                    if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                        continue
                    }
                    let root = netlistdb.net2pin.items[
                        netlistdb.net2pin.start[netid]
                    ];
                    if netlistdb.pin2cell[root] != 0 {
                        panic!("DFF {} driven by non-port pin {}: this pattern is not yet supported. please disable clock gating.",
                               netlistdb.cellnames[cellid],
                               netlistdb.pinnames[root].dbg_fmt_pin());
                    }
                    posedge_monitor.insert(root);
                }
            }
        }
    }
    clilog::info!(
        "clock ports detected: {}",
        posedge_monitor.iter()
            .map(|&i| netlistdb.pinnames[i].dbg_fmt_pin())
            .format(", "));

    let input_vcd = File::open(&args.input_vcd).unwrap();
    let mut bufrd = BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().unwrap();
    drop(vcd_parser);
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = FastFlow::new(vcd_file, 65536);

    let top_scope = find_top_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref().unwrap_or("")
    ).expect("Specified top scope not found in VCD.");

    let mut vcd2inp = HashMap::new();
    let mut inp_port_given = HashSet::new();

    let mut match_one_input = |var: &Var, i: Option<isize>, vcd_pos: usize| {
        let key = (VCDHier::empty(), var.reference.as_str(), i);
        if let Some(&id) = netlistdb.pinname2id.get(
            &key as &dyn GeneralPinName
        ) {
            if netlistdb.pindirect[id] != Direction::O { return }
            vcd2inp.insert((var.code.0, vcd_pos), id);
            inp_port_given.insert(id);
        }
    };
    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            use vcd_ng::ReferenceIndex::*;
            match var.index {
                None => match var.size {
                    1 => match_one_input(var, None, 0),
                    w @ _ => {
                        for (pos, i) in (0..w).rev()
                            .enumerate()
                        {
                            match_one_input(
                                var, Some(i as isize), pos)
                        }
                    }
                },
                Some(BitSelect(i)) => match_one_input(
                    var, Some(i as isize), 0),
                Some(Range(a, b)) => {
                    for (pos, i) in SVerilogRange(
                        a as isize, b as isize).enumerate()
                    {
                        match_one_input(var, Some(i), pos);
                    }
                }
            }
        }
    }
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I &&
            !inp_port_given.contains(&i)
        {
            clilog::warn!(
                GATESIM_VCDI_MISSING_PI,
                "Primary input port {:?} not present in \
                 the VCD input",
                netlistdb.pinnames[i]);
        }
    }

    let mut circ_state = vec![0u8; netlistdb.num_pins];
    let mut macro_states = HashMap::<usize, MacroState>::new();
    let mut srams = HashMap::new();
    if let Some(netid) = netlistdb.net_one {
        for pinid in netlistdb.net2pin.iter_set(netid) {
            circ_state[pinid] = 1u8;
        }
    }
    let mut topo_vis = vec![false; netlistdb.num_pins];
    let mut topo_instack = vec![false; netlistdb.num_pins];
    let mut topo = Vec::new();
    // mark all combinational circuit inputs
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I && !posedge_monitor.contains(&i) {
            topo_vis[i] = true;
        }
    }
    for cellid in 1..netlistdb.num_cells {
        if matches!(netlistdb.celltypes[cellid].as_str(),
                    "DFF" | "$__RAMGEM_SYNC_") {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "Q" | "PORT_R_RD_DATA") {
                    topo_vis[pinid] = true;
                    // do not add them to topo, but treat them separately before prop.
                }
            }
        }
        if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
            srams.insert(cellid, vec![0u32; 1 << 13]);
        }
        // A macro with no combinational outputs -- a DSP, whose only output P
        // is clocked -- is a topological leaf exactly like a DFF's Q: its
        // value is read from state and committed at the edge.
        if let Some(k) = MacroKind::from_celltype(netlistdb.celltypes[cellid].as_str()) {
            macro_states.insert(cellid, MacroState::default());
            if !k.has_comb_outputs() {
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    if netlistdb.pindirect[pinid] != Direction::I {
                        topo_vis[pinid] = true;
                    }
                }
            }
        }
    }
    fn dfs_topo(netlistdb: &NetlistDB, topo_vis: &mut Vec<bool>, topo_instack: &mut Vec<bool>, topo: &mut Vec<usize>, pinid: usize) {
        if topo_instack[pinid] {
            panic!("circuit has loop!");
        }
        if topo_vis[pinid] { return }
        topo_vis[pinid] = true;
        topo_instack[pinid] = true;
        if netlistdb.pindirect[pinid] == Direction::I {
            let netid = netlistdb.pin2net[pinid];
            if Some(netid) != netlistdb.net_zero && Some(netid) != netlistdb.net_one {
                let root = netlistdb.net2pin.items[
                    netlistdb.net2pin.start[netid]
                ];
                dfs_topo(netlistdb, topo_vis, topo_instack, topo, root);
            }
        }
        else {
            let cellid = netlistdb.pin2cell[pinid];
            let mk = MacroKind::from_celltype(netlistdb.celltypes[cellid].as_str());
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pn = netlistdb.pinnames[pinid].1.as_str();
                let take = match mk {
                    // Only the COMBINATIONAL inputs create a topological edge.
                    // Treating a DSP's A/B/C/D or an SRL's D/CE as
                    // combinational fan-in would manufacture false loops in
                    // netlists that legitimately feed a macro's output back
                    // into another macro's clocked input.
                    Some(k) => netlistdb.pindirect[pinid] == Direction::I
                               && k.is_comb_input(pn),
                    None => matches!(pn, "A" | "B"),
                };
                if take {
                    dfs_topo(netlistdb, topo_vis, topo_instack, topo, pinid);
                }
            }
        }
        topo.push(pinid);
        topo_instack[pinid] = false;
    }
    // start from all comb. circuit outputs
    for pinid in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[pinid] == Direction::I {
            dfs_topo(&netlistdb, &mut topo_vis, &mut topo_instack, &mut topo, pinid);
        }
    }
    for cellid in 1..netlistdb.num_cells {
        if matches!(netlistdb.celltypes[cellid].as_str(),
                    "DFF" | "$__RAMGEM_SYNC_") {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "D" | "PORT_R_ADDR" | "PORT_W_WR_EN" | "PORT_W_ADDR" | "PORT_W_WR_DATA") {
                    dfs_topo(&netlistdb, &mut topo_vis, &mut topo_instack, &mut topo, pinid);
                }
            }
        }
        // Every macro operand must be resolved before the edge commits, so
        // seed the traversal from all of them -- clocked ones included.
        if MacroKind::from_celltype(netlistdb.celltypes[cellid].as_str()).is_some() {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if netlistdb.pindirect[pinid] == Direction::I
                    && netlistdb.pinnames[pinid].1.as_str() != "CLK" {
                    dfs_topo(&netlistdb, &mut topo_vis, &mut topo_instack, &mut topo, pinid);
                }
            }
        }
    }
    for &clk in &posedge_monitor {
        if topo_vis[clk] {
            clilog::error!("Clock {} is also used in combinational logic. This is unsupported and might lead to error.",
                           netlistdb.pinnames[clk].dbg_fmt_pin());
        }
    }
    // clilog::info!("topo size: {} / {}", topo.len(), netlistdb.num_pins);

    // for i in 0..netlistdb.num_pins {
    //     if netlistdb.pinnames[i].0.cur.as_str() == "_46841_" {
    //         println!("pin _01039_ ({}) id {} net {}",
    //                  netlistdb.pinnames[i].dbg_fmt_pin(), i,
    //                  netlistdb.pin2net[i]);
    //     }
    // }

    // open out
    let write_buf = File::create(&args.output_vcd).unwrap();
    let write_buf = BufWriter::new(write_buf);
    let mut writer = Writer::new(write_buf);
    if let Some((ratio, unit)) = header.timescale {
        writer.timescale(ratio, unit).unwrap();
    }
    let output_vcd_scope = args.output_vcd_scope.as_deref().unwrap_or("gem_top_module");
    let output_vcd_scope = output_vcd_scope.split('/').collect::<Vec<_>>();
    for &scope in &output_vcd_scope {
        writer.add_module(scope).unwrap();
    }
    let mut out2vcd = netlistdb.cell2pin.iter_set(0).filter_map(|i| {
        if netlistdb.pindirect[i] == Direction::I {
            Some((i, writer.add_wire(
                1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin())).unwrap()))
        }
        else { None }
    }).collect::<Vec<_>>();
    if args.include_wires {
        out2vcd.extend((0..netlistdb.num_nets).filter_map(|i| {
            if Some(i) == netlistdb.net_zero || Some(i) == netlistdb.net_one {
                return None
            }
            let root = netlistdb.net2pin.items[netlistdb.net2pin.start[i]];
            if netlistdb.pindirect[root] != Direction::O {
                return None
            }
            Some((root, writer.add_wire(
                1, &format!("{}", netlistdb.netnames[i].dbg_fmt_pin())
            ).unwrap()))
        }));
    }
    let mut last_val = vec![2; out2vcd.len()];
    for _ in 0..output_vcd_scope.len() {
        writer.upscope().unwrap();
    }
    writer.enddefinitions().unwrap();
    writer.begin(SimulationCommand::Dumpvars).unwrap();

    // do simulation.
    let mut vcd_time = u64::MAX;
    let mut last_vcd_time_rising_edge = false;
    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time { continue }
                if last_vcd_time_rising_edge {
                    clilog::debug!("simulating t={}", vcd_time);
                    // latch the regs and srams.
                    for cellid in 1..netlistdb.num_cells {
                        if netlistdb.celltypes[cellid].as_str() == "DFF" {
                            let mut pinid_d = usize::MAX;
                            let mut pinid_q = usize::MAX;
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                match netlistdb.pinnames[pinid].1.as_str() {
                                    "D" => pinid_d = pinid,
                                    "Q" => pinid_q = pinid,
                                    _ => {}
                                }
                            }
                            circ_state[pinid_q] = circ_state[pinid_d];
                        }
                        else if let Some(k) = MacroKind::from_celltype(
                            netlistdb.celltypes[cellid].as_str()) {
                            let g = |port: &str| gather_port(&netlistdb, &circ_state, cellid, port);
                            let st = macro_states.get_mut(&cellid).unwrap();
                            match k {
                                MacroKind::Dsp48e2 => {
                                    st.p = eval_dsp48e2(
                                        g("A"), g("B"), g("C"), g("D"),
                                        g("USE_PREADD") != 0, g("MODE"), st.p);
                                },
                                MacroKind::Srlc32e => {
                                    if g("CE") != 0 {
                                        st.sr = (st.sr << 1) | (g("D") as u32 & 1);
                                    }
                                },
                                MacroKind::Carry4 => {},
                            }
                            let p = st.p;
                            if k == MacroKind::Dsp48e2 {
                                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                    if netlistdb.pinnames[pinid].1.as_str() == "P" {
                                        let bit = netlistdb.pinnames[pinid].2
                                            .unwrap_or(0).max(0) as u32;
                                        circ_state[pinid] = ((p as u64 >> bit) & 1) as u8;
                                    }
                                }
                            }
                        }
                        else if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
                            let sram = srams.get_mut(&cellid).unwrap();
                            let mut port_r_addr = 0usize;
                            let mut port_w_addr = 0usize;
                            let mut port_w_wr_en = 0u32;
                            let mut port_w_wr_data = 0u32;
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                macro_rules! load_var {
                                    ($($pin_name:literal => $var_name:ident),+) => {
                                        match netlistdb.pinnames[pinid].1.as_str() {
                                            $($pin_name => {
                                                $var_name = ($var_name as u64 | ((circ_state[pinid] as u64) << netlistdb.pinnames[pinid].2.unwrap())).try_into().unwrap();
                                            }),+,
                                            _ => {}
                                        }
                                    }
                                }
                                load_var! {
                                    "PORT_R_ADDR" => port_r_addr,
                                    "PORT_W_ADDR" => port_w_addr,
                                    "PORT_W_WR_EN" => port_w_wr_en,
                                    "PORT_W_WR_DATA" => port_w_wr_data
                                }
                            }
                            let port_r_rd_data = sram[port_r_addr];
                            let port_w_old_data = sram[port_w_addr];
                            let port_w_data = (port_w_old_data & (!port_w_wr_en)) | (port_w_wr_data & port_w_wr_en);
                            sram[port_w_addr] = port_w_data;
                            if netlistdb.cellnames[cellid].dbg_fmt_hier().as_str() == "cpu.instruction_unit.icache.memories[0].way_0_data_ram.mem.0.0" {
                                println!("our sram at time {vcd_time} port_r_addr {port_r_addr} port_w_addr {port_w_addr} port_w_wr_data {port_w_wr_data} -> port_r_rd_data {port_r_rd_data}");
                            }
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                macro_rules! save_var {
                                    ($($pin_name:literal <= $var_name:ident),+) => {
                                        match netlistdb.pinnames[pinid].1.as_str() {
                                            $($pin_name => {
                                                circ_state[pinid] = ($var_name >> netlistdb.pinnames[pinid].2.unwrap() & 1) as u8;
                                            }),+,
                                            _ => {}
                                        }
                                    }
                                }
                                save_var! {
                                    "PORT_R_RD_DATA" <= port_r_rd_data
                                }
                            }
                        }
                    }
                    // propagate
                    for &pinid in &topo {
                        // if netlistdb.pin2cell[pinid] == 0 {
                        //     println!("trying to visit port {}", netlistdb.pinnames[pinid].dbg_fmt_pin());
                        // }
                        if netlistdb.pindirect[pinid] == Direction::I {
                            let netid = netlistdb.pin2net[pinid];
                            if Some(netid) != netlistdb.net_zero && Some(netid) != netlistdb.net_one {
                                let root = netlistdb.net2pin.items[
                                    netlistdb.net2pin.start[netid]
                                ];
                                circ_state[pinid] = circ_state[root];
                                // if netlistdb.pin2cell[pinid] == 0 {
                                //     println!("changing output for pin {} to {}", netlistdb.pinnames[pinid].dbg_fmt_pin(), circ_state[pinid]);
                                // }
                            }
                        }
                        else {
                            let cellid = netlistdb.pin2cell[pinid];
                            // Combinational macro outputs: a CARRY4's CO/O,
                            // an SRLC32E's Q/Q31. A DSP never reaches here --
                            // its P was marked a topological leaf.
                            if let Some(k) = MacroKind::from_celltype(
                                netlistdb.celltypes[cellid].as_str()) {
                                let st = *macro_states.get(&cellid).unwrap();
                                let pn = netlistdb.pinnames[pinid].1.clone();
                                let bit = netlistdb.pinnames[pinid].2
                                    .unwrap_or(0).max(0) as u32;
                                let v = match k {
                                    MacroKind::Carry4 => {
                                        let (co, o) = eval_carry4(
                                            gather_port(&netlistdb, &circ_state, cellid, "S"),
                                            gather_port(&netlistdb, &circ_state, cellid, "DI"),
                                            gather_port(&netlistdb, &circ_state, cellid, "CIN"),
                                            gather_port(&netlistdb, &circ_state, cellid, "CYINIT"));
                                        match pn.as_str() {
                                            "CO" => (co >> bit) & 1,
                                            "O"  => (o  >> bit) & 1,
                                            _ => 0,
                                        }
                                    },
                                    MacroKind::Srlc32e => {
                                        // Read the CURRENT state: Q is a
                                        // combinational read of the register
                                        // as it stands before this edge.
                                        let a = gather_port(&netlistdb, &circ_state, cellid, "A");
                                        match pn.as_str() {
                                            "Q"   => ((st.sr >> (a & 31)) & 1) as u64,
                                            "Q31" => ((st.sr >> 31) & 1) as u64,
                                            _ => 0,
                                        }
                                    },
                                    MacroKind::Dsp48e2 => 0,
                                };
                                circ_state[pinid] = v as u8;
                                continue
                            }
                            let mut vala = 0;
                            let mut valb = 0;
                            for pinid_inp in netlistdb.cell2pin.iter_set(cellid) {
                                match netlistdb.pinnames[pinid_inp].1.as_str() {
                                    "A" => vala = circ_state[pinid_inp],
                                    "B" => valb = circ_state[pinid_inp],
                                    "Y" => {},
                                    _ => unreachable!()
                                }
                            }
                            circ_state[pinid] = match netlistdb.celltypes[cellid].as_str() {
                                "AND2_00_0" => vala & valb,
                                "AND2_01_0" => vala & (valb ^ 1),
                                "AND2_10_0" => (vala ^ 1) & valb,
                                "AND2_11_0" => (vala | valb) ^ 1,
                                "AND2_11_1" => vala | valb,
                                "INV" => vala ^ 1,
                                "BUF" => vala,
                                _ => unreachable!()
                            };
                            // if netlistdb.pin2net[pinid] == 1039 {
                            //     println!("d_we_o input gate: {} {} type {}", vala, valb, netlistdb.celltypes[cellid].as_str());
                            // }
                        }
                    }
                    // write vcd vars out
                    writer.timestamp(vcd_time).unwrap();
                    for (i, &(pinid, vid)) in out2vcd.iter().enumerate() {
                        use vcd_ng::Value;
                        let value_new = circ_state[pinid];
                        if value_new == last_val[i] {
                            continue
                        }
                        last_val[i] = value_new;
                        writer.change_scalar(vid, match value_new {
                            1 => Value::V1,
                            _ => Value::V0
                        }).unwrap();
                    }
                }
                // reset for next timestamp
                vcd_time = t;
                last_vcd_time_rising_edge = false;
                for &clk in &posedge_monitor {
                    circ_state[clk] = 0;
                }
            },
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                for (pos, &b) in bits.iter().enumerate() {
                    if let Some(&pin) = vcd2inp.get(
                        &(id.0, pos)
                    ) {
                        if b == b'1' && posedge_monitor.contains(&pin) {
                            last_vcd_time_rising_edge = true;
                        }
                        circ_state[pin] = match b {
                            b'1' => 1, _ => 0
                        };
                    }
                }
            }
        }
    }
}

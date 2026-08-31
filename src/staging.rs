// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Splitting deep circuit into major stages at global level indices.
//!
//! This is crucial in efficiently handling large and deep circuits
//! with a limited processing element width.
//!
//! Major stages also carry the ONLY ordering primitive strong enough to make a
//! combinational macro result visible to boolean logic. Inside one partition
//! the boomerang stages hand values to each other through `shared_state`,
//! while the macro phase writes into `shared_writeouts` -- a buffer no
//! boomerang stage ever reads. So a CARRY4 feeding an AND gate can only work
//! if the AND gate lives in a LATER major stage, where it reads the carry back
//! out of global state. [`macro_avail_stages`] and [`StagedAIG::from_macro_stage`]
//! build exactly that split.

use indexmap::IndexSet;
use crate::aig::{AIG, EndpointGroup, DriverType};

/// Major-stage assignment forced by combinational macros.
///
/// Two numbers per aigpin, because for a macro output they differ:
///
/// * `avail[i]` -- the stage in which boolean logic may READ the value.
/// * `eval[i]` -- for a combinational macro output, the stage in which the
///   macro that drives it RUNS. Equal to `avail[i]` for everything else.
///
/// The gap of one exists only for combinational macro outputs, and it is the
/// whole reason this module grew a second splitter: the macro phase writes its
/// results into `shared_writeouts`, which no boomerang stage reads, so the
/// value does not become visible to AND gates until the commit has carried it
/// to global state and a later major stage reads it back.
pub struct MacroStaging {
    pub avail: Vec<u32>,
    pub eval: Vec<u32>,
    /// Highest major stage index. Zero means no combinational macro feeds
    /// anything, and the caller must use the ordinary level-split path so
    /// macro-free designs stage exactly as they always did.
    pub max_stage: u32,
}

/// Compute [`MacroStaging`] for a netlist.
///
/// The rules, in one place:
///
/// * A leaf -- primary input, DFF `Q`, SRAM read data, a DSP's `P` -- is 0.
///   Every one of those is read from state and is available immediately.
/// * An AND gate takes the max over its fan-in. Boolean depth is free: a whole
///   cone of AND gates evaluates inside one major stage.
/// * A macro runs in the latest stage among its operands -- but an operand
///   that is ITSELF a combinational macro output contributes that macro's
///   `eval`, not its `avail`. Macro-to-macro needs no boundary: pe.rs emits
///   ordered batches and the kernel separates them with `__syncthreads()`, so
///   a CARRY4 chain of any length costs zero extra major stages. Only
///   macro-to-boolean does.
/// * A combinational macro output becomes readable one stage after it runs.
pub fn macro_avail_stages(aig: &AIG) -> MacroStaging {
    let order = aig.topo_traverse_generic(None, None);
    let mut avail = vec![0u32; aig.num_aigpins + 1];
    let mut eval = vec![0u32; aig.num_aigpins + 1];
    for &i in &order {
        match aig.drivers[i] {
            DriverType::AndGate(a, b) => {
                let va = if (a >> 1) != 0 { avail[a >> 1] } else { 0 };
                let vb = if (b >> 1) != 0 { avail[b >> 1] } else { 0 };
                avail[i] = va.max(vb);
                eval[i] = avail[i];
            },
            DriverType::Macro(cellid, _) => {
                match aig.macros.get(&cellid) {
                    Some(mb) if mb.kind.has_comb_outputs() => {
                        let mut m = 0;
                        for j in mb.comb_inputs() {
                            m = m.max(if aig.is_comb_macro_output(j) {
                                eval[j]
                            } else {
                                avail[j]
                            });
                        }
                        eval[i] = m;
                        avail[i] = m + 1;
                    },
                    // A DSP's P is clocked and read from state, exactly like a
                    // DFF's Q: stage 0, no boundary needed.
                    _ => { avail[i] = 0; eval[i] = 0; },
                }
            },
            _ => { avail[i] = 0; eval[i] = 0; },
        }
    }
    // The highest stage anything actually has to run in. A macro output whose
    // only consumers are other macros never needs its `avail` to exist, so
    // counting it would manufacture an empty trailing major stage.
    let mut max_stage = 0u32;
    for i in 1..aig.num_aigpins + 1 {
        max_stage = max_stage.max(eval[i]);
        if !aig.is_comb_macro_output(i) { continue }
        let st = aig.fanouts_start[i];
        let en = aig.fanouts_start[i + 1];
        if aig.fanouts[st..en].iter().any(|&c| !aig.is_comb_macro_output(c)) {
            max_stage = max_stage.max(avail[i]);
        }
    }
    for e in 0..aig.num_endpoint_groups() {
        aig.get_endpoint_group(e).for_each_input(|i| {
            max_stage = max_stage.max(avail[i]);
        });
    }
    MacroStaging { avail, eval, max_stage }
}

/// A struct representing the boundaries of a staged AIG.
pub struct StagedAIG {
    /// the staged primary inputs from previous levels.
    pub primary_inputs: Option<IndexSet<usize>>,
    /// the staged primary output pins for next levels.
    ///
    /// these pins are active nodes at the level split.
    pub primary_output_pins: Vec<usize>,
    /// the endpoint indices of original AIG fulfilled by current level.
    pub endpoints: Vec<usize>,
}

impl StagedAIG {
    /// Get the number of endpoint groups that should be fulfilled
    /// with this staged AIG.
    ///
    /// This mimics the interface given by a raw AIG.
    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_output_pins.len() + self.endpoints.len()
    }

    /// Get the virtual endpoint group with an index.
    ///
    /// This mimics the interface given by a raw AIG.
    pub fn get_endpoint_group<'aig>(&self, aig: &'aig AIG, endpt_id: usize) -> EndpointGroup<'aig> {
        if endpt_id < self.primary_output_pins.len() {
            EndpointGroup::StagedIOPin(self.primary_output_pins[endpt_id])
        }
        else {
            aig.get_endpoint_group(self.endpoints[endpt_id - self.primary_output_pins.len()])
        }
    }

    /// build a staged AIG that consists of all levels.
    pub fn from_full_aig(aig: &AIG) -> Self {
        StagedAIG {
            primary_inputs: None,
            primary_output_pins: vec![],
            endpoints: (0..aig.num_endpoint_groups()).collect()
        }
    }

    /// build a staged AIG by horizontal splitting given a subset
    /// of endpoints.
    ///
    /// return built StagedAIG.
    /// the endpoints are given as a slice of endpoint group indices,
    /// that must have all staged primary output groups at the front
    /// and original endpoints following. otherwise we panic.
    ///
    /// the result guarantees that the endpoint `i` corresponds to
    /// the original staged's endpoint `endpoint_subset[i]`.
    pub fn to_endpoint_subset(
        &self,
        endpoint_subset: &[usize]
    ) -> StagedAIG {
        let mut staged_sub = StagedAIG {
            primary_inputs: self.primary_inputs.clone(),
            primary_output_pins: vec![],
            endpoints: vec![],
        };
        for &endpt_i in endpoint_subset {
            if endpt_i < self.primary_output_pins.len() {
                staged_sub.primary_output_pins.push(
                    self.primary_output_pins[endpt_i]
                );
                assert!(staged_sub.endpoints.is_empty(),
                        "endpoint subset must be in order!");
            }
            else {
                staged_sub.endpoints.push(
                    self.endpoints[endpt_i - self.primary_output_pins.len()]
                );
            }
        }
        staged_sub
    }

    /// build a staged AIG by vertical splitting at the given level id.
    ///
    /// return built StagedAIG.
    /// the active middle nodes at split can be obtained from the
    /// StagedAIG::primary_output_pins.
    /// if this is empty, it means all endpoints are already satisfied
    /// after this stage.
    pub fn from_split(
        aig: &AIG,
        unrealized_orig_endpoints: &IndexSet<usize>,
        primary_inputs: Option<&IndexSet<usize>>,
        split_at_level: usize,
    ) -> Self {
        let mut unrealized_endpoint_nodes = Vec::new();
        for &endpt in unrealized_orig_endpoints {
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                unrealized_endpoint_nodes.push(i);
            });
        }
        assert!(!unrealized_endpoint_nodes.is_empty());
        let order = aig.topo_traverse_generic(
            Some(&unrealized_endpoint_nodes),
            primary_inputs
        );
        let mut num_fanouts = vec![0; aig.num_aigpins + 1];
        let mut level_id = vec![0; aig.num_aigpins + 1];
        for &i in &order {
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) {
                continue
            }
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    num_fanouts[a >> 1] += 1;
                    level_id[i] = level_id[i].max(level_id[a >> 1] + 1);
                }
                if b >= 2 {
                    num_fanouts[b >> 1] += 1;
                    level_id[i] = level_id[i].max(level_id[b >> 1] + 1);
                }
            }
        }
        let mut endpt_level_id = vec![0; aig.num_endpoint_groups()];
        for &endpt in unrealized_orig_endpoints {
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                num_fanouts[i] += 1;
                endpt_level_id[endpt] = endpt_level_id[endpt].max(level_id[i]);
            });
        }
        let mut nodes_at_split = IndexSet::new();
        for &i in &order {
            if level_id[i] > split_at_level { continue }
            nodes_at_split.insert(i);
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) {
                continue
            }
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    num_fanouts[a >> 1] -= 1;
                    if num_fanouts[a >> 1] == 0 {
                        assert!(nodes_at_split.swap_remove(&(a >> 1)));
                    }
                }
                if b >= 2 {
                    num_fanouts[b >> 1] -= 1;
                    if num_fanouts[b >> 1] == 0 {
                        assert!(nodes_at_split.swap_remove(&(b >> 1)));
                    }
                }
            }
        }
        let mut endpoints_before_split = Vec::new();
        for &endpt in unrealized_orig_endpoints {
            if endpt_level_id[endpt] > split_at_level { continue }
            endpoints_before_split.push(endpt);
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                num_fanouts[i] -= 1;
                if num_fanouts[i] == 0 {
                    assert!(nodes_at_split.swap_remove(&i));
                }
            });
        }

        StagedAIG {
            primary_inputs: primary_inputs.cloned(),
            primary_output_pins: nodes_at_split.iter().copied()
                .filter(|po| !matches!(primary_inputs, Some(pi) if pi.contains(po)))
                .collect(),
            endpoints: endpoints_before_split
        }
    }

    /// Build the major stage `stage_j` of a macro-aware split.
    ///
    /// Unlike [`StagedAIG::from_split`], which cuts at a boolean level, this
    /// cuts at combinational macro boundaries: stage `j` realises every
    /// endpoint whose inputs are all available by `j`, evaluates every macro
    /// whose operands are ready by `j`, and hands forward -- through global
    /// state -- every value a later stage needs.
    ///
    /// The forwarded set has two parts, and the second is easy to miss. A
    /// macro output with `avail == j + 1` is PRODUCED here, by this stage's
    /// macro phase, and consumed only later. If it is not forwarded it never
    /// reaches state and the next stage reads a stale bit -- which is the
    /// silent-wrong-answer bug this whole split exists to close.
    pub fn from_macro_stage(
        aig: &AIG,
        unrealized_orig_endpoints: &IndexSet<usize>,
        primary_inputs: Option<&IndexSet<usize>>,
        ms: &MacroStaging,
        stage_j: u32,
    ) -> Self {
        let (avail, eval) = (&ms.avail, &ms.eval);
        // Which endpoints can be finished here, and which slip to a later
        // stage. An endpoint needs every input READABLE, so it keys off
        // `avail`, not `eval`.
        let mut endpoints_here = Vec::new();
        let mut deferred_inputs = Vec::new();
        let mut all_endpoint_inputs = Vec::new();
        for &endpt in unrealized_orig_endpoints {
            let mut latest = 0u32;
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                latest = latest.max(avail[i]);
                all_endpoint_inputs.push(i);
            });
            if latest <= stage_j { endpoints_here.push(endpt); }
            else {
                // Anything this stage can already produce and a later stage
                // still wants. Keyed off eval, not avail, so a macro output
                // evaluated here (eval == stage_j, avail == stage_j + 1) is
                // forwarded rather than dropped.
                aig.get_endpoint_group(endpt).for_each_input(|i| {
                    if eval[i] <= stage_j {
                        deferred_inputs.push(i);
                    }
                });
            }
        }

        // Only nodes in the live cone are candidates -- walking every aigpin
        // would forward dead logic into state.
        let live = aig.topo_traverse_generic(
            Some(&all_endpoint_inputs), primary_inputs);

        // A value must cross into state when this stage can produce it but
        // something that runs later still needs it.
        let needed_later = |n: usize| -> bool {
            let st = aig.fanouts_start[n];
            let en = aig.fanouts_start[n + 1];
            aig.fanouts[st..en].iter().any(|&c| {
                // The stage a consumer runs in is the stage it is produced in:
                // an AND gate at `avail`, a macro at `avail - 1`.
                eval[c] > stage_j
            })
        };

        let mut nodes_at_split = IndexSet::new();
        for &i in &live {
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) { continue }
            if eval[i] > stage_j { continue }
            if needed_later(i) { nodes_at_split.insert(i); }
        }
        // Inputs of endpoints that slipped to a later stage also have to
        // survive, even when every ordinary fanout of theirs was satisfied
        // here.
        for i in deferred_inputs {
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) { continue }
            nodes_at_split.insert(i);
        }

        StagedAIG {
            primary_inputs: primary_inputs.cloned(),
            primary_output_pins: nodes_at_split.into_iter().collect(),
            endpoints: endpoints_here,
        }
    }
}

/// Given the level split points, return a list of split stages.
///
/// For example, given [10, 20], will return a list like this:
/// [(0, 10, stage0_10), (10, 20, stage10_20), (20, MAX, stage20_MAX)]
///
/// If the netlist ends early before all split points, the length might be
/// shorter than expected.
pub fn build_staged_aigs(
    aig: &AIG, level_split: &[usize]
) -> Vec<(usize, usize, StagedAIG)> {
    let ms = macro_avail_stages(aig);
    if ms.max_stage > 0 {
        if !level_split.is_empty() {
            clilog::warn!(
                "--level-split is ignored on a netlist with combinational \
                 macros: major stages are derived from macro depth ({} \
                 boundaries) instead.", ms.max_stage);
        }
        return build_staged_aigs_by_macro(aig, &ms)
    }
    build_staged_aigs_by_level(aig, level_split)
}

/// One major stage per combinational-macro depth.
///
/// Stage `j` evaluates every macro whose operands are ready by `j` and every
/// AND cone that needs nothing deeper; the macro results reach global state at
/// the commit, and stage `j + 1` reads them back as ordinary primary inputs.
/// That grid sync between major stages is what makes a macro result visible to
/// boolean logic at all.
fn build_staged_aigs_by_macro(
    aig: &AIG, ms: &MacroStaging
) -> Vec<(usize, usize, StagedAIG)> {
    let mut ret = Vec::new();
    let mut unrealized_orig_endpoints =
        (0..aig.num_endpoint_groups()).collect::<IndexSet<_>>();
    let mut primary_inputs: Option<IndexSet<usize>> = None;

    for j in 0..=ms.max_stage {
        let staged = StagedAIG::from_macro_stage(
            aig, &unrealized_orig_endpoints, primary_inputs.as_ref(), ms, j);
        for &endpt in &staged.endpoints {
            assert!(unrealized_orig_endpoints.swap_remove(&endpt));
        }
        let is_last = staged.primary_output_pins.is_empty();
        let pi = primary_inputs.get_or_insert_with(Default::default);
        for &inp in &staged.primary_output_pins { pi.insert(inp); }
        ret.push((j as usize,
                  if is_last { usize::MAX } else { j as usize + 1 },
                  staged));
        if is_last { break }
    }
    assert!(unrealized_orig_endpoints.is_empty(),
            "macro staging left {} endpoint group(s) unrealised after {} \
             stages -- macro_avail_stages and from_macro_stage disagree about \
             when a value becomes readable",
            unrealized_orig_endpoints.len(), ret.len());
    ret
}

fn build_staged_aigs_by_level(
    aig: &AIG, level_split: &[usize]
) -> Vec<(usize, usize, StagedAIG)> {
    let mut ret = Vec::new();
    let mut unrealized_orig_endpoints = (0..aig.num_endpoint_groups()).collect::<IndexSet<_>>();
    let mut primary_inputs: Option<IndexSet<usize>> = None;

    for i in 0..level_split.len() {
        let cur_split = level_split[i];
        let last_split = match i {
            0 => 0,
            i @ _ => level_split[i - 1]
        };
        let staged = StagedAIG::from_split(
            aig, &unrealized_orig_endpoints, primary_inputs.as_ref(),
            cur_split - last_split
        );
        for &endpt in &staged.endpoints {
            assert!(unrealized_orig_endpoints.swap_remove(&endpt));
        }
        let primary_inputs = primary_inputs.get_or_insert_with(|| Default::default());
        for &inp in &staged.primary_output_pins {
            primary_inputs.insert(inp);
        }
        if staged.primary_output_pins.is_empty() {
            ret.push((last_split, usize::MAX, staged));
            return ret
        }
        ret.push((last_split, cur_split, staged));
    }

    let last_split = match level_split.len() {
        0 => 0,
        i @ _ => level_split[i - 1]
    };
    ret.push((last_split, usize::MAX, StagedAIG::from_split(
        aig, &unrealized_orig_endpoints, primary_inputs.as_ref(),
        usize::MAX
    )));

    ret
}

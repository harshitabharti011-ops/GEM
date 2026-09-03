# The Big-GEM Theory

### Pool SHAuryas
**Takneek 2026 • PS Zenith • Electronics Club, IIT Kanpur**

> Native `DSP48E2`, `CARRY4` and `SRLC32E` execution inside GEM's Boomerang engine, through a macro-preserving heterogeneous GPU execution pipeline.

A fork of [NVlabs/GEM](https://github.com/NVlabs/GEM) in which word-level FPGA primitives are never shredded into Boolean logic. They survive Yosys as first-class cells, are scheduled by a heterogeneous DAG, laid out in 64-bit aligned VRAM, and evaluated natively on the GPU ALU as `int64_t` arithmetic — inside the existing Boomerang kernel, whose reduction tree is unmodified. 1,591 lines of new host and device code.

📄 **[Full technical report](docs/BigGEM_Project_Report.pdf)** — scheduling equations, memory architecture, verification methodology, complete numerical analysis.

---

## Why it matters

GEM re-reads its compiled instruction script from global memory **every simulated cycle**, so shredding a word-level macro costs you once *per cycle*, forever. One `DSP48E2`, four `CARRY4` and one `SRLC32E` expand from **95 AIG cells to 9,922** — a factor of 104.

## Measured results

`macro_bench` at 16 lanes · RTX 3050 Laptop (`sm_86`, 1,536 KiB L2) · identical stimulus, `num_blocks` and cycle count on both sides · baseline = the same codebase with macros shredded to AIG, asserted macro-free.

| Quantity | Baseline | Native | |
|---|---:|---:|---|
| AIG cells | 85,444 | **5,754** | 14.8× fewer |
| Per-cycle instruction script | 2,060.0 KiB | **454.5 KiB** | 4.53× smaller |
| Script vs 1,536 KiB L2 | 1.34× — spills | **0.30× — fits** | residency crossed |
| DRAM throughput (% of peak) | 46.91 | **0.04** | 1,173× less traffic |
| Warps active (% of peak) | 33.33 | 33.33 | occupancy identical |
| Local memory ld/st | 0 | 0 | nothing spills |
| Grid barriers per cycle | 1 | 2 | the one added cost |
| **Throughput (cycles/s)** | **22,426** | **13,657** | **0.61×** |

**Why the wall clock is below the baseline, and how we know.** Bandwidth, spilling and occupancy are each eliminated by a measured counter, not by argument — DRAM traffic fell 1,173×, local ld/st is 0 in both configurations, and `sm__warps_active` is bit-identical at 33.33 %. What remains is the one structural difference: a second grid-wide barrier per simulated cycle, sized consistently by two independent instruments at **37.75 µs** (Nsight, 200 cycles) and **31.1 µs** (wall clock, 400 cycles). The macro path is *synchronisation-bound*, not memory-bound.

The script reduction is a **residency transition, not a linear saving**: 4.53× smaller produced 1,173× less traffic because it carried the script across the L2 capacity. Confirmed arithmetically — script re-reads alone account for 56.2 of the baseline's 61.8 GB/s (91 % of all measured traffic), while the native run moves ≈ 0.8 MB total against a 0.465 MB script.

The fix follows from the equations rather than from guesswork: eq. (4) charges the extra major stage exactly where a combinational macro result enters the Boolean domain. Collapsing it (injecting macro results into the Boomerang level-1 input set within the same partition) is specified in the report and **not implemented** — no claim here depends on it.

## Verification

No golden reference was supplied, so correctness is layered — each rung isolates one layer, so a disagreement names the layer that caused it.

| Layer | What is checked | Evidence |
|---|---|---|
| Macro models | CUDA models vs an independent Python transcription of the PS equations | **14,552** vectors, **0 disagreements** — `CARRY4` **exhaustive** (1,024), `DSP48E2` 1,648, `SRLC32E` 11,880 |
| Synthesis | Macros **present** in the native netlist, **absent** in the shredded one | `check_macros.sh`, non-zero exit if a macro leaks into the baseline |
| Boolean GPU path | Shredded GPU vs `naive_sim` CPU ground truth | 32,800 comparisons — Pass |
| Design equivalence | Shredded vs macro-preserving netlist, **both on CPU** | 32,800 — Pass |
| Macro GPU path | Native GPU vs `naive_sim` | 32,800 — Pass |
| Host/kernel bisect | Compiled script executed on CPU (`flatten_test`) vs ground truth | 32,800 — Pass |
| Regression | Full ladder on `macro_smoke` | 49,200 — Pass |

## Reproducing

Requires Rust, Python 3, a CUDA Toolkit matching your GPU (**≥ 12.8** for Blackwell / `sm_120`), and Yosys ≥ 0.68 — the last is fetched automatically if missing.

```bash
git clone --recursive <repo-url> && cd GEM
bash judge_run.sh bench/macro_bench.sv macro_bench      # Linux
```
```bat
compile.bat bench\macro_bench.sv macro_bench            :: Windows, via WSL2
```

Either entry point builds, synthesises the design both ways, maps, runs the correctness ladder, measures throughput and profiles with Nsight Compute, writing everything to `judge_results/`. Point it at any SystemVerilog design and top module.

## PS Zenith coverage

| | Implementation | Evidence |
|---|---|---|
| **A** Host parser & Yosys pre-processor | 3 blackboxes (`synth/gem_macros.v`) + `macro_normalize_xilinx.v`; `(* blackbox *)` and `keep 1` defeat techmap/abc; `MacroLayout::build` maps nodes into 64-bit aligned `UVec<u64>` buffers, warp-padded per kind | `check_macros.sh` asserts both directions; alignment holds structurally |
| **B** CUDA engine & Boomerang modification | Two-label heterogeneous schedule (`staging.rs`, eq. 1–4); type-homogeneous `MacroBatch` (`pe.rs:31`); `CO[3]→CIN` chains ordered by one batch per link without intermediate Boolean nodes (`pe.rs:751`); macro phase at `kernel_v1_impl.cuh:305–437`; `shared_writeouts` for operands, `__shfl_down_sync` preserved at 5 sites, no atomics | 64 chained `CARRY4` resolve in **one** major stage, 5 batches; 86 regs, 0 spill |
| **C** Hardware macro implementations | Cycle-accurate `DSP48E2` (PREG-only clocked, 27-bit pre-adder truncation), `CARRY4`, `SRLC32E` (read-old/commit-new) — one `__host__ __device__` source for CPU and GPU | 14,552 vectors, cross-language, 0 disagreements |
| **D** Benchmarks & performance analysis | `throughput.sh`, `sweep.sh`, `profile.sh` with `--replay-mode application` (a cooperative launch cannot survive default kernel replay) | Tables above; `bench/sweep_results.csv` |
| **E** Documentation | Scheduling equations, block diagrams, numerical analysis | `docs/BigGEM_Project_Report.pdf` |

## Where the code lives

`synth/` Yosys interception and both flows · `src/aig.rs:393` macro recognition · `src/staging.rs` scheduling equations · `src/pe.rs` batching and chain ordering · `src/flatten.rs`, `src/macro_layout.rs` VRAM layout and script emission · `csrc/kernel_v1_impl.cuh` Boomerang + macro phase · `csrc/macros.cuh` macro models · `bench/` verification ladder, benchmarks, profiling

---

**Pool SHAuryas** — Harshita Bharti · Lavanya Prakash · Aditi · Aaahana Mehrotra · Gauri Singh · Soundarya Murugaiyan · Dharshini S · Swadha Singh · Ashritha Aryapu

Built on **GEM** — Guo, Zhang, Wang, Lin and Ren, *GEM: GPU-Accelerated Emulator-Inspired RTL Simulation*, DAC 2025, NVIDIA Research. Upstream licence retained; see `LICENSE`.

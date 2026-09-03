# The Big-GEM Theory

**Native word-level macro evaluation in a GPU-accelerated RTL simulator.**

A fork of [NVlabs/GEM](https://github.com/NVlabs/GEM) that teaches the boomerang
scheduler to evaluate Xilinx hardware macros — `DSP48E2`, `CARRY4`, `SRLC32E` —
as native 64-bit functional units on the GPU ALU, instead of letting the Yosys
frontend shred them into thousands of primitive AIG nodes.

Takneek 2026 · Zenith · Pool Shauryas
---

## The idea

Stock GEM reduces every design to 1-bit boolean logic and compiles it into an
instruction *script* that each CUDA block re-reads from global memory **every
simulated cycle**. That script is the per-cycle DRAM traffic floor of the whole
simulator, and shredding a 27×18 multiplier into an AIG is what makes it large.

So the target is not cell count — it is script bytes, and specifically whether
the script still fits in L2. We intercept the three macros before `techmap`/`abc`
can flatten them, carry them through mapping as first-class heterogeneous nodes,
pack their state into 64-bit-aligned buffers alongside the 1-bit boolean state,
and evaluate them in a dedicated macro phase inside the existing boomerang
kernel.

## Results

Measured on an RTX 3050 6 GB Laptop GPU (sm_86, 20 SMs, 131.7 GB/s peak, 1.5 MiB
L2), `macro_bench` at 16 lanes, 400 cycles, 40 blocks, median of 5 reps.

**The memory goal was met, decisively:**

| Metric (16 lanes)                | Shredded baseline | Macro-preserving | Change |
|----------------------------------|------------------:|-----------------:|-------:|
| Script size                      |    527,360 × u32  |   116,340 × u32  | **4.53× smaller** |
| Script residency vs L2           |  2060 KiB = 1.34× |  454 KiB = 0.30× | **crosses the cliff** |
| DRAM throughput (% of peak)      |           46.80 % |          0.04 %  | **~1000× less traffic** |
| Partitions                       |                 6 |            2 + 1 | 3× fewer |
| Local memory ld/st (spill check) |                 0 |                0 | clean |

**And the throughput still went down:**

| Lanes | Baseline (cyc/s) | Macro path (cyc/s) | Speedup | Script reduction |
|------:|-----------------:|-------------------:|--------:|-----------------:|
|     8 |           29,616 |             12,676 |  0.43×  |            4.57× |
|    16 |           24,206 |             14,903 |  0.62×  |            4.53× |
|    32 |           19,249 |             10,268 |  0.53×  |            2.45× |
The optimisation did exactly what it was
designed to do at the memory level — the script went from *not fitting* in L2 to
*comfortably fitting*, and DRAM traffic essentially vanished — and the simulation
still got ~1.6× slower.

A speedup number we cannot reproduce would be worth less than this. The full
attribution is in the technical report.

## Correctness

All four rungs of the verification ladder pass at 16 lanes — 82 ports, 400
cycles, 32,800 comparisons per rung:

| # | Comparison                                     | Isolates                        | Result |
|---|------------------------------------------------|---------------------------------|--------|
| 1 | shredded GPU vs CPU reference                  | the stock GPU path              | PASS   |
| 2 | shredded CPU vs macro-preserving CPU           | *synthesis* equivalence         | PASS   |
| 3 | macro GPU vs CPU reference                     | the native macro GPU path       | PASS   |
| 4 | compiled script on CPU vs CPU reference        | host compiler vs kernel         | PASS   |

Rung 2 is the one no GPU-only comparison can give you: it proves the two netlists
are the same design before any GPU is involved. Rung 4 bisects rung 3 — if 3
fails and 4 passes, the bug is in the kernel; if both fail, it is in scheduling,
placement or script encoding.

No golden reference was provided by the problem statement, so the CPU models in
`csrc/tests/` and `src/bin/naive_sim.rs` are ours and are shipped for inspection.

## Layout

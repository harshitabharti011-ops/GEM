# GEM macro-preserving synthesis flow

Deliverable A, first half: intercept `DSP48E2`, `CARRY4` and `SRLC32E` before
Yosys shreds them into the AIG, and hand GEM a gate-level netlist where they
survive as single cells.

## Files

| File | Role |
|---|---|
| `gem_macros.v` | Canonical blackbox cells `$__GEMDSP_`, `$__GEMCARRY4_`, `$__GEMSRL32_`. **Frozen** — these names and widths are the contract with `src/aigpdk.rs`. |
| `macro_normalize.v` | Swap layer. Maps the incoming primitive spelling onto the canonical cells. Currently the PS's simplified subset. |
| `macro_normalize_xilinx.v` | Alternative swap layer for real Xilinx ports (OPMODE→MODE extraction lives here). `cp` it over `macro_normalize.v` if the hidden benchmarks need it. |
| `gem_macros_behav.v` | Simulation-only bodies. Golden reference for deliverable C; `csrc/macros.cuh` must match it bit for bit. |
| `step1_memory.ys` | usage.md step 1 + macro interception. |
| `step2_logic.ys` | usage.md step 2, macros preserved through `synth`/`abc`/`techmap`/`opt_clean`. |
| `setup_yosys.sh` | Installs/verifies a Yosys new enough for PS note 4. Distro packages are too old. |
| `run_synth.sh` | Fills the `@DESIGN@`/`@TOP@`/`@OUT@` placeholders and runs both steps. |
| `check_macros.sh` | Pass/fail gate: counts macro cells in the JSON. Non-zero exit if they were shredded. |

## Run it

Needs Yosys 0.68 (PS note 4). None of this needs a GPU.

**Do not use the distro package.** Ubuntu 22.04 ships Yosys 0.13 and 24.04
ships 0.33 -- both far below the 0.68 the PS mandates, and old enough that
`read_verilog -sv` will choke on IEEE 1800-2012 constructs in the hidden
benchmarks. This applies to Kaggle too, where `apt-get install yosys` looks
like it worked and quietly leaves you on 0.33.

```sh
./synth/setup_yosys.sh          # install a new-enough build, or say why not
./synth/setup_yosys.sh --check  # verify only
```

It prefers Homebrew on macOS and falls back to the YosysHQ oss-cad-suite
nightly, the only prebuilt carrying a current Yosys on both macOS arm64 and
Kaggle's linux-x64.

Then run the flow:

```sh
./synth/run_synth.sh synth/tests/macro_smoke.sv macro_smoke
```

A pass looks like:

```
  cell type                 count
  ----------------------------------
  AND2_00_0                    ...
  $__GEMCARRY4_                  4  <-- MACRO
  $__GEMDSP_                     2  <-- MACRO
  $__GEMSRL32_                   1  <-- MACRO

PASS: 7 macro instances intact alongside NNN AIG cells.
```

If it says `FAIL: ... they were shredded`, check in this order:

1. `gem_macros.v` was read with `-lib` (that is what makes a module a blackbox);
2. it was read **before** the design;
3. the `setattr -set keep 1` line ran — without it `opt_clean -purge` sweeps any
   macro whose outputs are not yet used;
4. the design's port names match `macro_normalize.v` — a mismatch makes
   `hierarchy -check` elaborate a different module instead.

## Simulate the result

`usage.md` asks you to check each synthesis step against a CPU simulator. The
behavioural bodies make that a one-liner:

```sh
iverilog -g2012 -o sim build/gatelevel.gv synth/gem_macros_behav.v \
         aigpdk/aigpdk.v your_testbench.sv && ./sim
```

## What this does not do yet

`src/aigpdk.rs` does not know these cell names, so GEM's Rust parser will still
panic with *"Cannot recognize pin type"*. That is step 2 of the build plan.

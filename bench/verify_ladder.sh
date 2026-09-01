#!/usr/bin/env bash
# The verification ladder.
#
# Each rung isolates exactly one layer, so a disagreement names the layer that
# caused it instead of just reporting a wrong waveform:
#
#   naive_sim   CPU, evaluates the NETLIST directly -- no script, no
#               partitioning, no boomerang scheduling. Ground truth.
#   cuda_test   GPU, evaluates the compiled script. Everything else.
#
# Three comparisons, three distinct questions:
#
#   1. Is the shredded GPU path correct?        naive(shred)  vs GPU(shred)
#   2. Is macro-preserving synthesis correct?   naive(shred)  vs naive(native)
#   3. Is the macro GPU path correct?           naive(native) vs GPU(native)
#
# Question 2 is the one no GPU-only comparison can answer: it establishes that
# the two netlists are the same design before any GPU is involved.
#
#     ./bench/verify_ladder.sh
set -uo pipefail
cd "$(dirname "$0")/.."
V=build_native/in.vcd
NB=${NB:-40}          # blocks; MUST be identical for both GPU runs

# Regenerate BOTH GPU waveforms from the current binary and the current
# stimulus. Comparing a freshly-run CPU reference against a stale out.vcd is
# the one way to make this whole ladder lie, so the runs live here rather than
# being left to the caller to remember.
echo "==> GPU: shredded"
./target/release/cuda_test --top-module macro_smoke \
    build_shred/gatelevel.gv  build_shred/r.gemparts \
    "$V" build_shred/out.vcd  "$NB" 2>&1 | grep -E "Elapsed|error|panic" | tail -2
echo "==> GPU: macro-preserving"
./target/release/cuda_test --top-module macro_smoke \
    build_native/gatelevel.gv build_native/r.gemparts \
    "$V" build_native/out.vcd "$NB" 2>&1 | grep -E "Elapsed|error|panic" | tail -2

echo "==> CPU reference: shredded netlist"
./target/release/naive_sim --top-module macro_smoke \
    build_shred/gatelevel.gv  "$V" build_shred/naive.vcd  2>&1 | grep -v "simulating t=" | tail -2

echo "==> CPU reference: macro-preserving netlist"
./target/release/naive_sim --top-module macro_smoke \
    build_native/gatelevel.gv "$V" build_native/naive.vcd 2>&1 | grep -v "simulating t=" | tail -2

echo
echo "==> 1. shredded GPU vs CPU ground truth"
python3 bench/compare_vcd.py build_shred/naive.vcd  build_shred/out.vcd  --shift -1

echo
echo "==> 2. design equivalence: shredded vs macro-preserving, both on CPU"
python3 bench/compare_vcd.py build_shred/naive.vcd  build_native/naive.vcd

echo
echo "==> 3. macro GPU vs CPU ground truth"
python3 bench/compare_vcd.py build_native/naive.vcd build_native/out.vcd --shift -1

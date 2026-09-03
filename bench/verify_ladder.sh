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

# Which design to verify. Defaults to the macro_smoke micro-benchmark; set
# these to point at a scaled build produced by bench/gen_bench.sh, e.g.
#     TOP=macro_bench SHRED=build_bench16_shred NATIVE=build_bench16_native \
#         bash bench/verify_ladder.sh
TOP=${TOP:-macro_smoke}
SHRED=${SHRED:-build_shred}
NATIVE=${NATIVE:-build_native}
V=${V:-$NATIVE/in.vcd}
NB=${NB:-40}          # blocks; MUST be identical for both GPU runs
CYC=${CYC:-400}       # cycles compared; must not exceed what the VCD holds

# Build first, and STOP if the build fails.
#
# cargo leaves the previous binary in place when a build fails, so a ladder
# that just runs ./target/release/* will happily verify code that is not the
# code on disk -- and report it as a result. That is worse than no result. It
# bit us once already: a flatten.rs fix that flatten_test picked up and
# cuda_test did not, which read as "the host is fixed, the kernel is broken"
# when both were fine and one binary was simply old.
#
# cuda_test is the one that fails silently, because it is the only target that
# needs nvcc and the CUDA libraries. If it cannot link, everything else still
# builds. Set BUILD=0 only when you have just built by hand.
. bench/env.sh
if [ "${BUILD:-1}" = "1" ]; then gem_build || exit 1; fi

# Remap both flows unless REMAP=0.
#
# A .gemparts is not an input, it is a BUILD PRODUCT of the scheduler in
# src/pe.rs: it stores each partition's stages AND its macro batches. Change
# the scheduler, keep the old file, and the simulator faithfully executes the
# old schedule -- which is how a run once came back with sixteen DSP48E2s
# present as endpoints, holding state slots, committing every cycle, and never
# once scheduled into a batch. Nothing reports that; the DSPs simply never
# evaluate. Remapping here costs a minute and removes the whole failure mode.
if [ "${REMAP:-1}" = "1" ]; then
  for d in "$SHRED" "$NATIVE"; do
    echo "==> remap $d"
    ./target/release/cut_map_interactive --top-module "$TOP" \
        "$d/gatelevel.gv" "$d/r.gemparts" 2>&1 \
      | grep -E "after merging|effective partitions" | tail -2
  done
fi

# Regenerate BOTH GPU waveforms from the current binary and the current
# stimulus. Comparing a freshly-run CPU reference against a stale out.vcd is
# the one way to make this whole ladder lie, so the runs live here rather than
# being left to the caller to remember.
echo "==> GPU: shredded"
./target/release/cuda_test --top-module "$TOP" \
    "$SHRED/gatelevel.gv"  "$SHRED/r.gemparts" \
    "$V" "$SHRED/out.vcd"  "$NB" 2>&1 | grep -E "Elapsed|error|panic" | tail -2
echo "==> GPU: macro-preserving"
./target/release/cuda_test --top-module "$TOP" \
    "$NATIVE/gatelevel.gv" "$NATIVE/r.gemparts" \
    "$V" "$NATIVE/out.vcd" "$NB" 2>&1 | grep -E "Elapsed|error|panic" | tail -2

echo "==> CPU reference: shredded netlist"
./target/release/naive_sim --top-module "$TOP" \
    "$SHRED/gatelevel.gv"  "$V" "$SHRED/naive.vcd"  2>&1 | grep -v "simulating t=" | tail -2

echo "==> CPU reference: macro-preserving netlist"
./target/release/naive_sim --top-module "$TOP" \
    "$NATIVE/gatelevel.gv" "$V" "$NATIVE/naive.vcd" 2>&1 | grep -v "simulating t=" | tail -2

echo
echo "==> 1. shredded GPU vs CPU ground truth"
python3 bench/compare_vcd.py -c "$CYC" "$SHRED/naive.vcd"  "$SHRED/out.vcd"  --shift -1

echo
echo "==> 2. design equivalence: shredded vs macro-preserving, both on CPU"
python3 bench/compare_vcd.py -c "$CYC" "$SHRED/naive.vcd"  "$NATIVE/naive.vcd"

echo
echo "==> 3. macro GPU vs CPU ground truth"
python3 bench/compare_vcd.py -c "$CYC" "$NATIVE/naive.vcd" "$NATIVE/out.vcd" --shift -1

# Rung 4 bisects rung 3 when it fails. flatten_test runs the SAME compiled
# script on the CPU, so it shares the host compiler (staging, pe, flatten) with
# the GPU but none of the kernel. Rung 3 failing tells you the macro path is
# wrong; only this says which half.
#
#   3 fails, 4 fails  -> host: scheduling, placement or script encoding
#   3 fails, 4 passes -> kernel: indexing, fencing or a race
#
# Off by default because it is a slow interpreter; FLAT=1 turns it on.
if [ "${FLAT:-0}" = "1" ]; then
  echo
  echo "==> 4. compiled script on CPU vs CPU ground truth"
  ./target/release/flatten_test --top-module "$TOP" \
      "$NATIVE/gatelevel.gv" "$NATIVE/r.gemparts" \
      "$V" "$NATIVE/flat.vcd" 2>&1 | grep -E "error|panic|thread" | tail -3
  python3 bench/compare_vcd.py -c "$CYC" "$NATIVE/naive.vcd" "$NATIVE/flat.vcd" --shift -1
fi

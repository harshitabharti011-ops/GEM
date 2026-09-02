#!/usr/bin/env bash
# Synthesise bench/macro_bench.sv at a given lane count, both ways, and report
# the numbers that decide whether the workload has entered the regime the
# macro optimisation targets.
#
#     bash bench/gen_bench.sh 16
#     bash bench/gen_bench.sh 64
#
# Produces build_bench<N>_shred/ and build_bench<N>_native/, laid out exactly
# like build_shred/ and build_native/ so every existing tool works unchanged.
#
# What to watch, in order of importance:
#
#   partitions  The number that actually drives script size. macro_smoke maps
#               to ONE partition either way, which is why its script barely
#               shrank. Scale until the shredded flow needs many and the
#               macro-preserving flow needs materially fewer.
#
#   script size Read from global memory every simulated cycle. This is the
#               per-cycle bandwidth tax the whole approach attacks.
#
#   AIG cells   The headline reduction, but on its own it predicts nothing
#               about runtime -- macro_smoke reduced 104x and got slower.
set -uo pipefail
cd "$(dirname "$0")/.."

N=${1:?usage: gen_bench.sh <LANES>}
SRC=/tmp/macro_bench_${N}.sv
sed "s/parameter LANES = 16/parameter LANES = ${N}/" \
    bench/macro_bench.sv > "$SRC"

echo "=== LANES=$N : synthesising both flows ==="
./synth/run_synth.sh --shred "$SRC" macro_bench "build_bench${N}_shred"  2>&1 | tail -20
./synth/run_synth.sh         "$SRC" macro_bench "build_bench${N}_native" 2>&1 | tail -20

echo
echo "=== stimulus ==="
python3 bench/make_vcd.py "build_bench${N}_native/gatelevel.gv" macro_bench \
        -n 500 -o "build_bench${N}_native/in.vcd"

echo
echo "=== mapping (partition count is the number that matters) ==="
for f in shred native; do
    D="build_bench${N}_${f}"
    echo "--- $f ---"
    ./target/release/cut_map_interactive --top-module macro_bench \
        "$D/gatelevel.gv" "$D/r.gemparts" 2>&1 \
      | grep -E "after merging|effective partitions|netlist has" | tail -5
done

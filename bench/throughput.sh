#!/usr/bin/env bash
# Deliverable D: throughput of the shredded baseline against the
# macro-preserving path.
#
#     bash bench/throughput.sh
#     TOP=macro_bench SHRED=build_bench16_shred NATIVE=build_bench16_native \
#         bash bench/throughput.sh
#
# Both sides use the SAME stimulus, the SAME num_blocks and the SAME cycle
# count. Vary any of those and the comparison is void, so they are set once
# here rather than passed per side.
#
# The timer is GEM's own `simulation` span (cuda_test.rs), which brackets the
# kernel launches and excludes netlist parsing, mapping and VCD writing. Those
# are one-off costs; the PS asks about simulation rate.
#
# REPS runs per side, and the report keeps the MINIMUM as well as the median.
# A laptop GPU clocks down under sustained load and shares the die with the
# desktop, so the distribution has a long right tail and a hard left edge --
# the minimum is the closest thing to "what the machine can do", the median is
# what you would actually observe. Reporting only the mean hides both.
#
# Also printed: script size in bytes. That is the quantity this whole approach
# attacks -- GEM re-reads the instruction script from global memory EVERY
# simulated cycle, so script bytes x cycles is the per-run DRAM traffic floor.
# A cell-count reduction that does not shrink the script cannot make anything
# faster, and saying so plainly is worth more than a flattering ratio.
set -uo pipefail
cd "$(dirname "$0")/.."

TOP=${TOP:-macro_bench}
SHRED=${SHRED:-build_bench16_shred}
NATIVE=${NATIVE:-build_bench16_native}
V=${V:-$NATIVE/in.vcd}
NB=${NB:-40}            # identical on both sides
CYC=${CYC:-400}
REPS=${REPS:-5}

if [ ! -x ./target/release/cuda_test ]; then
    echo "cuda_test is missing. Build with:"
    echo "    cargo build --release --features cuda"
    exit 1
fi
STALE=$(find src csrc build.rs Cargo.toml -type f \
          -newer target/release/cuda_test 2>/dev/null | head -3)
if [ -n "$STALE" ]; then
    echo "STALE cuda_test -- newer sources exist:"; echo "$STALE" | sed 's/^/    /'
    echo "Rebuild with: cargo build --release --features cuda"
    exit 1
fi

echo "GPU:    $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"
echo "Config: num_blocks=$NB  cycles=$CYC  reps=$REPS  stimulus=$V"
echo

measure() {
    local label=$1 dir=$2 tag=$3
    local raw=/tmp/thr_${tag}.raw
    : > "$raw"
    local size=""
    for _ in $(seq 1 "$REPS"); do
        ./target/release/cuda_test --top-module "$TOP" --max-cycles "$CYC" \
            "$dir/gatelevel.gv" "$dir/r.gemparts" "$V" "/tmp/thr_${tag}.vcd" \
            "$NB" > "/tmp/thr_${tag}.log" 2>&1
        grep -oP 'simulation, Elapsed=\K[0-9.]+(ms|s|µs|us)' \
            "/tmp/thr_${tag}.log" >> "$raw"
        [ -z "$size" ] && size=$(grep -oP 'script size \K[0-9]+' \
                                 "/tmp/thr_${tag}.log" | tail -1)
    done
    python3 - "$label" "$raw" "$CYC" "${size:-0}" <<'PY'
import sys, statistics as st
label, raw, cyc, words = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mult = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}
secs = []
for line in open(raw):
    line = line.strip()
    if not line: continue
    for suf in ("ms", "µs", "us", "s"):
        if line.endswith(suf):
            secs.append(float(line[:-len(suf)]) * mult[suf]); break
if not secs:
    print(f"{label:22s}  NO TIMING -- check /tmp for the log"); sys.exit()
best, med = min(secs), st.median(secs)
print(f"{label:22s}  min {best*1e3:8.2f} ms  median {med*1e3:8.2f} ms   "
      f"cycles/s  min-based {cyc/best:10,.0f}  median-based {cyc/med:10,.0f}")
print(f"{'':22s}  script {words*4/1024:8.1f} KiB "
      f"({words:,} u32)   per-cycle DRAM floor {words*4/1024:.1f} KiB")
PY
}

measure "SHREDDED BASELINE" "$SHRED"  shred
measure "MACRO-PRESERVING"  "$NATIVE" native

echo
cat <<'EOF'
Reading these numbers
---------------------
  speedup       median-based native / median-based shredded. Below 1.0 means
                the macro path is SLOWER, which is a real and reportable
                result: it says the workload was not bandwidth-bound, so the
                script it shrank was not what the run was waiting on.

  script KiB    Re-read from global memory every simulated cycle. If this
                barely moved, no throughput change should be expected and the
                cell-count reduction is not the relevant metric.

  min vs median A wide gap means thermal or contention noise, not a property
                of either implementation. Close the laptop's other work and
                re-run before quoting the number.
EOF

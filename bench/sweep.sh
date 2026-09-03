#!/usr/bin/env bash
# Deliverable D: throughput across scales.
#
#     bash bench/sweep.sh 8 16 32 64
#     REPS=3 bash bench/sweep.sh 16 32
#
# A single design gives a single ratio, and a single ratio cannot distinguish
# "this optimisation does not work" from "this design is too small for it to
# work yet". A sweep can, because the two predict different SHAPES:
#
#   not bandwidth-bound at any scale  ->  the ratio stays flat below 1
#   too small, crossover further out  ->  the ratio climbs with lane count
#
# So the deliverable is the curve. Every row is measured; nothing is
# extrapolated, and a lane count that fails to synthesise, map or run is
# recorded as a failure rather than dropped -- a sweep that silently omits its
# hard cases reports the easy ones as if they were the whole story.
#
# Emitted to bench/sweep_results.csv for the report, and printed as a table.
#
# Correctness is NOT checked here: naive_sim on a 340k-cell shredded netlist is
# minutes per run. Verify separately at the largest lane count that matters:
#
#   FLAT=1 TOP=macro_bench SHRED=build_bench64_shred NATIVE=build_bench64_native \
#       CYC=400 bash bench/verify_ladder.sh
#
# A throughput number from an unverified configuration is not a result.
set -uo pipefail
cd "$(dirname "$0")/.."

LANES=("$@")
[ ${#LANES[@]} -eq 0 ] && LANES=(8 16 32 64)
NB=${NB:-40}
CYC=${CYC:-400}
REPS=${REPS:-5}
CSV=${CSV:-bench/sweep_results.csv}

# Sets CUDA_LIBRARY_PATH / UCC_CUDA_GENCODE if unset, builds with
# --features cuda, and fails loudly on a stale or missing cuda_test. A fresh
# terminal without those exports is how the last sweep died.
. bench/env.sh
if [ "${BUILD:-1}" = "1" ]; then gem_build || exit 1; fi
if [ ! -x ./target/release/cuda_test ]; then
    echo "cuda_test missing."; exit 1
fi

mkdir -p bench/logs
echo "lanes,flow,cells,dffs,macros,partitions,script_u32,median_ms,cycles_per_s" > "$CSV"

echo "GPU:    $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"
echo "Config: num_blocks=$NB cycles=$CYC reps=$REPS lanes=${LANES[*]}"

for N in "${LANES[@]}"; do
    echo
    echo "################ LANES=$N ################"
    GEN="bench/logs/gen_${N}.log"
    if ! bash bench/gen_bench.sh "$N" > "$GEN" 2>&1; then
        echo "  SYNTHESIS/MAPPING FAILED -- last 15 lines of $GEN:"
        tail -15 "$GEN" | sed 's/^/    /'
        echo "$N,shred,FAIL,,,,,," >> "$CSV"
        echo "$N,native,FAIL,,,,,," >> "$CSV"
        continue
    fi

    for FLOW in shred native; do
        D="build_bench${N}_${FLOW}"
        V="build_bench${N}_native/in.vcd"
        LOG="bench/logs/run_${N}_${FLOW}.log"
        RAW="bench/logs/raw_${N}_${FLOW}.txt"
        : > "$RAW"
        ok=1
        for _ in $(seq 1 "$REPS"); do
            if ! ./target/release/cuda_test --top-module macro_bench \
                    --max-cycles "$CYC" \
                    "$D/gatelevel.gv" "$D/r.gemparts" \
                    "$V" "/tmp/sweep_${N}_${FLOW}.vcd" "$NB" > "$LOG" 2>&1; then
                ok=0; break
            fi
            grep -oP 'simulation, Elapsed=\K[0-9.]+(ms|s|µs|us)' "$LOG" >> "$RAW"
        done
        if [ "$ok" = 0 ]; then
            echo "  $FLOW: RUN FAILED -- last 12 lines of $LOG:"
            tail -12 "$LOG" | sed 's/^/    /'
            echo "$N,$FLOW,,,,,,FAIL," >> "$CSV"
            continue
        fi

        # Design facts come from the logs the tools already emit, so the CSV
        # cannot drift from what was actually built and run.
        SCRIPT=$(grep -oP 'script size \K[0-9]+' "$LOG" | tail -1)
        # Partitions and cells come from THIS flow's own section of the gen
        # log. gen_bench.sh runs shred then native into one file, so grepping
        # the whole file concatenated both flows' numbers into every row --
        # a column that looked populated and meant nothing.
        SEC=$(awk -v f="--- $FLOW ---" 'BEGIN{p=0} $0 ~ f {p=1;next}
                                        /^--- /{if(p)exit} p' "$GEN")
        PARTS=$(printf '%s\n' "$SEC" | grep -oP 'after merging: \K[0-9]+' \
                | tr '\n' '+' | sed 's/+$//')
        CELLS=$(awk -v f="=== \\[$FLOW\\]" 'BEGIN{p=0} $0 ~ f {p=1} p' "$GEN" \
                | grep -oP 'Number of cells:\s+\K[0-9]+' | head -1)

        python3 - "$N" "$FLOW" "$RAW" "$CYC" "${SCRIPT:-0}" \
                   "${CELLS:-}" "${PARTS:-}" "$CSV" <<'PY'
import sys, statistics as st
N, flow, raw, cyc, script, cells, parts, csv = sys.argv[1:9]
cyc, script = int(cyc), int(script)
mult = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}
secs = []
for line in open(raw):
    line = line.strip()
    if not line: continue
    for suf in ("ms", "µs", "us", "s"):
        if line.endswith(suf):
            secs.append(float(line[:-len(suf)]) * mult[suf]); break
if not secs:
    print(f"  {flow:7s} NO TIMING"); sys.exit()
med = st.median(secs)
cps = cyc / med
print(f"  {flow:7s} median {med*1e3:8.2f} ms   {cps:10,.0f} cycles/s   "
      f"script {script*4/1024:9.1f} KiB   cells {cells or '?':>8}   "
      f"parts {parts or '?'}")
with open(csv, "a") as f:
    f.write(f"{N},{flow},{cells},,,{parts},{script},{med*1e3:.3f},{cps:.0f}\n")
PY
    done
done

echo
echo "================ SUMMARY ================"
python3 - "$CSV" <<'PY'
import sys, csv as C
rows = list(C.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    if not r.get("cycles_per_s"): continue
    by.setdefault(r["lanes"], {})[r["flow"]] = r
print(f"{'lanes':>6} {'shred c/s':>12} {'native c/s':>12} {'speedup':>9} "
      f"{'script ratio':>13} {'shred parts':>12} {'native parts':>13}")
for n in sorted(by, key=lambda v: int(v)):
    s, m = by[n].get("shred"), by[n].get("native")
    if not (s and m): continue
    sp = float(m["cycles_per_s"]) / float(s["cycles_per_s"])
    sr = int(s["script_u32"]) / max(int(m["script_u32"]), 1)
    flag = "  <-- native faster" if sp > 1.0 else ""
    print(f"{n:>6} {float(s['cycles_per_s']):>12,.0f} "
          f"{float(m['cycles_per_s']):>12,.0f} {sp:>8.2f}x "
          f"{sr:>12.2f}x {s['partitions']:>12} {m['partitions']:>13}{flag}")
print()
print("A speedup column that CLIMBS with lane count locates a crossover and")
print("makes the small-scale loss a scale effect. A FLAT column below 1 says")
print("the script was never the bottleneck on this device, which is a")
print("different and equally reportable finding. Do not average the two.")
PY
echo
echo "CSV: $CSV"

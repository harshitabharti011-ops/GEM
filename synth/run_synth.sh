#!/usr/bin/env bash
# usage: synth/run_synth.sh <design.sv> <TOP_MODULE> [outdir]
set -euo pipefail
MODE=native
if [ "${1:-}" = "--shred" ]; then MODE=shred; shift; fi
DESIGN="${1:?need a design file}"
TOP="${2:?need a top module name}"
OUT="${3:-build}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p "$OUT"
SUF=""; [ "$MODE" = shred ] && SUF="_shred"

fill() { sed -e "s|@DESIGN@|$DESIGN|g" -e "s|@TOP@|$TOP|g" -e "s|@OUT@|$OUT|g" "$1" > "$OUT/$(basename "$1")"; }
fill "synth/step1_memory${SUF}.ys"
fill "synth/step2_logic${SUF}.ys"

echo "=== [$MODE] step 1: memory mapping ==="
yosys -q -s "$OUT/step1_memory${SUF}.ys"
echo "=== [$MODE] step 2: logic synthesis to aigpdk ==="
yosys -q -s "$OUT/step2_logic${SUF}.ys"

if [ "$MODE" = shred ]; then
  echo "=== baseline: macros should be GONE ==="
  synth/check_macros.sh "$OUT/gatelevel.json" && {
    echo "FAIL: macros survived the baseline path -- the comparison would be void"
    exit 1
  } || echo "OK: macros shredded into AIG, as the baseline requires"
else
  synth/check_macros.sh "$OUT/gatelevel.json"
fi

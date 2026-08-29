#!/usr/bin/env bash
# usage: synth/run_synth.sh <design.sv> <TOP_MODULE> [outdir]
set -euo pipefail
DESIGN="${1:?need a design file}"
TOP="${2:?need a top module name}"
OUT="${3:-build}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p "$OUT"

fill() { sed -e "s|@DESIGN@|$DESIGN|g" -e "s|@TOP@|$TOP|g" -e "s|@OUT@|$OUT|g" "$1" > "$OUT/$(basename "$1")"; }
fill synth/step1_memory.ys
fill synth/step2_logic.ys

echo "=== step 1: memory mapping + macro interception ==="
yosys -q -s "$OUT/step1_memory.ys"
echo "=== step 2: logic synthesis to aigpdk ==="
yosys -q -s "$OUT/step2_logic.ys"

synth/check_macros.sh "$OUT/gatelevel.json"

#!/usr/bin/env bash
# Deliverable C: prove the macro models are structurally accurate.
#
# No golden reference is provided by the PS, so the oracle is two independent
# transcriptions of its equations -- csrc/macros.cuh in C++ and
# reference_model.py in Python -- checked against each other bit for bit over
# an identical vector set. CARRY4 coverage is exhaustive (all 1024 cases).
#
# Needs no GPU and no CUDA toolkit: macros.cuh compiles as ordinary C++.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "==> compiling the C++ model as host code"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -o "$TMP/dump" "$HERE/dump_vectors.cpp"

echo "==> generating vectors"
"$TMP/dump"                        > "$TMP/cpp.txt"
python3 "$HERE/reference_model.py" > "$TMP/py.txt"

c4=$(grep -c '^C4 '   "$TMP/cpp.txt")
ds=$(grep -c '^DSP '  "$TMP/cpp.txt")
sr=$(grep -c '^SRL '  "$TMP/cpp.txt")
se=$(grep -c '^SRLE ' "$TMP/cpp.txt")
echo "    CARRY4  $c4 cases (exhaustive: 16x16x2x2 = 1024)"
echo "    DSP48E2 $ds cases (8 directed + 400 random, x4 accumulate cycles)"
echo "    SRLC32E $sr reads + $se edges"

if diff -q "$TMP/cpp.txt" "$TMP/py.txt" >/dev/null; then
    echo "PASS: C++ and Python models agree on all $(wc -l < "$TMP/cpp.txt") vectors."
else
    echo "FAIL: models disagree. First 20 differences:"
    diff "$TMP/cpp.txt" "$TMP/py.txt" | head -20
    exit 1
fi
[ "$c4" -eq 1024 ] || { echo "FAIL: CARRY4 coverage not exhaustive"; exit 1; }

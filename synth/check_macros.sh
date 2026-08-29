#!/usr/bin/env bash
# Pass/fail gate for deliverable A: did the macros survive synthesis intact?
set -euo pipefail
JSON="${1:?usage: check_macros.sh build/gatelevel.json}"
python3 - "$JSON" <<'PY'
import json, sys, collections

# Yosys spells a PUBLIC identifier with a leading backslash in RTLIL. It is
# normally stripped on the way into JSON, but our macro names themselves start
# with '$', and stripping there would make them indistinguishable from Yosys's
# own internal cell types ($_AND_, $mux, ...). So the JSON carries the
# backslash and we normalise it away here. sverilogparse does the same thing
# on the Verilog side (sverilognom.rs:67), which is why src/aigpdk.rs matches
# on the unescaped spelling.
def norm(t): return t[1:] if t.startswith("\\") else t

d = json.load(open(sys.argv[1]))
counts = collections.Counter()
for mod in d["modules"].values():
    for c in mod.get("cells", {}).values():
        counts[norm(c["type"])] += 1

MACROS = ("$__GEMDSP_", "$__GEMCARRY4_", "$__GEMSRL32_")
print("  cell type                 count")
print("  " + "-" * 34)
for t, n in counts.most_common():
    print(f"  {t:<24} {n:>6}" + ("  <-- MACRO" if t in MACROS else ""))

total = sum(counts[m] for m in MACROS)
aig   = sum(n for t, n in counts.items()
            if t.startswith("AND2") or t in ("INV", "BUF"))
ffs   = sum(n for t, n in counts.items() if t in ("DFF", "DFFSR"))
print()

if total == 0:
    print(f"FAIL: no macros survived. {aig} AIG cells present -- they were shredded.")
    sys.exit(1)

for m in MACROS:
    if counts[m] == 0:
        print(f"WARN: no {m} instances. Fine if the design has none; a bug if it does.")

# Secondary evidence: a shredded macro dumps its internal state into DFFs.
# Each SRLC32E would add 32 and each DSP48E2 48, so a low FF count next to a
# healthy macro count is independent confirmation nothing was expanded.
print(f"PASS: {total} macro instances intact "
      f"({', '.join(f'{counts[m]}x {m}' for m in MACROS if counts[m])}) "
      f"alongside {aig} AIG cells and {ffs} flip-flops.")
PY

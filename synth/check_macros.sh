#!/usr/bin/env bash
# Pass/fail gate for deliverable A: did the macros survive synthesis intact?
set -euo pipefail
JSON="${1:?usage: check_macros.sh build/gatelevel.json}"
python3 - "$JSON" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1]))
counts = collections.Counter()
for mod in d["modules"].values():
    for c in mod.get("cells", {}).values():
        counts[c["type"]] += 1
macros = ["$__GEMDSP_", "$__GEMCARRY4_", "$__GEMSRL32_"]
print("  cell type                 count")
print("  " + "-"*34)
for t, n in counts.most_common():
    mark = " <-- MACRO" if t in macros else ""
    print(f"  {t:<24} {n:>6}{mark}")
total = sum(counts[m] for m in macros)
ands  = sum(n for t, n in counts.items() if t.startswith("AND2") or t in ("INV","BUF"))
print()
if total == 0:
    print(f"FAIL: no macros survived. {ands} AIG cells present -- they were shredded.")
    sys.exit(1)
print(f"PASS: {total} macro instances intact alongside {ands} AIG cells.")
PY

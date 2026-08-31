#!/usr/bin/env bash
# Type-check, borrow-check and semantically test the macro path with plain
# rustc -- no cargo registry, no network, no GPU.
#
#     ./devtools/stubcheck/run.sh
#
# Covers src/aigpdk.rs, src/macros.rs and src/aig.rs against the API-shaped
# stubs in stubs.rs. `cargo test` stays authoritative; this exists so the path
# can be checked on a machine that cannot fetch crates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/devtools/stubcheck"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
mkdir -p "$OUT/src"
cp "$HERE/stubs.rs" "$HERE/semantics.rs" "$HERE/staging_tests.rs" "$OUT/src/"

python3 - "$ROOT" "$OUT" <<'PY'
import pathlib, re, sys
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for name in ("aigpdk", "macros", "macro_layout", "aig", "staging"):
    t = (root / "src" / f"{name}.rs").read_text()
    # repoint external-crate imports at the stubs; swap the logger for println!
    t = t.replace("use netlistdb::", "use crate::stubs::netlistdb::")
    t = t.replace("use indexmap::", "use crate::stubs::indexmap::")
    t = t.replace("use compact_str::CompactString;", "use crate::stubs::CompactString;")
    t = t.replace("use sverilogparse::", "use crate::stubs::sverilogparse::")
    # serde is a proc-macro crate; it cannot be stubbed by a plain rustc lib.
    # Drop the import and the two derives -- serialization is not what this
    # harness checks, and leaving them in makes the whole run fail to compile.
    t = t.replace("use serde::{Serialize, Deserialize};", "")
    t = re.sub(r",\s*(?:Serialize|Deserialize)\b", "", t)
    t = re.sub(r"#\[serde\([^\)]*\)\]", "", t)
    t = re.sub(r"clilog::(info|error|warn)!", "println!", t)
    t = t.replace("//! ", "// ")     # inner doc comments cannot follow items
    (out / "src" / f"{name}.rs").write_text(t)
(out / "src/lib.rs").write_text(
    "pub mod stubs;\npub mod aigpdk;\npub mod macros;\npub mod macro_layout;\npub mod aig;\npub mod staging;\npub mod semantics;\npub mod staging_tests;\n")
PY

echo "==> compile"
rustc --edition 2021 --crate-type=lib "$OUT/src/lib.rs" -o "$OUT/h.rlib"
echo "==> tests"
rustc --edition 2021 --test "$OUT/src/lib.rs" -o "$OUT/t"
"$OUT/t"

#!/usr/bin/env bash
# Get a Yosys new enough for the PS (note 4 requires 0.68).
#
#   ./synth/setup_yosys.sh          # check, install if needed
#   ./synth/setup_yosys.sh --check  # check only
#
# Distro packages are the trap here: Ubuntu 22.04 ships Yosys 0.13 and even
# 24.04 ships 0.33, both far below 0.68. Homebrew tracks upstream closely; if
# it is still short, we fall back to the YosysHQ oss-cad-suite nightly, which
# is the only prebuilt that reliably carries a current Yosys on both macOS and
# Kaggle.
set -euo pipefail
NEED=0.68

ver()  { yosys -V 2>/dev/null | sed -n 's/^Yosys \([0-9][0-9.]*\).*/\1/p'; }
# 0.9 must compare below 0.68, so compare the minor component numerically
# rather than lexically -- sort -V gets this wrong for Yosys's scheme.
ge()   { python3 -c "
import sys
def k(v): 
    p=v.split('.')
    return (int(p[0]), int(p[1]) if len(p)>1 else 0)
sys.exit(0 if k(sys.argv[1]) >= k(sys.argv[2]) else 1)" "$1" "$2"; }

have="$(ver || true)"
if [ -n "$have" ] && ge "$have" "$NEED"; then
  echo "OK: yosys $have (>= $NEED) at $(command -v yosys)"; exit 0
fi
[ -n "$have" ] && echo "found yosys $have -- too old, need >= $NEED" \
                || echo "yosys not found"
[ "${1:-}" = "--check" ] && exit 1

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)  PLAT=darwin-arm64 ;;
  Darwin/x86_64) PLAT=darwin-x64   ;;
  Linux/x86_64)  PLAT=linux-x64    ;;
  Linux/aarch64) PLAT=linux-arm64  ;;
  *) echo "unsupported platform: $(uname -s)/$(uname -m)"; exit 1 ;;
esac

if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null; then
  echo "==> trying Homebrew first"
  brew install yosys || true
  have="$(ver || true)"
  if [ -n "$have" ] && ge "$have" "$NEED"; then
    echo "OK: yosys $have via Homebrew"; exit 0
  fi
  echo "Homebrew gave ${have:-nothing}; falling back to oss-cad-suite"
fi

echo "==> fetching latest oss-cad-suite ($PLAT)"
URL=$(curl -fsSL https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest \
      | python3 -c "
import json,sys
rel=json.load(sys.stdin)
for a in rel['assets']:
    if '$PLAT' in a['name'] and a['name'].endswith(('.tgz','.tar.gz')):
        print(a['browser_download_url']); break
else:
    sys.exit('no $PLAT asset in ' + rel['tag_name'])")
echo "    $URL"
mkdir -p "$HOME/.local/opt"
curl -fL "$URL" | tar xz -C "$HOME/.local/opt"

BIN="$HOME/.local/opt/oss-cad-suite/bin"
echo
echo "Installed. Add to your shell profile:"
echo "    export PATH=\"$BIN:\$PATH\""
echo
echo "Then re-run:  ./synth/setup_yosys.sh --check"

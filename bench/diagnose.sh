#!/usr/bin/env bash
# One-shot diagnostic for the DSP macro-phase fault.
# Collects, in a single file, everything needed to localise it:
#   1. whether a sanitizer exists on this machine
#   2. the underlying CUresult, with launches serialised
#   3. the same run at 2 blocks instead of 40 (isolates block-count scaling)
#   4. sanitizer output if available
#
#     ./bench/diagnose.sh   > .out.txt 2>&1
set +e
cd "$(dirname "$0")/.."

echo "===== 1. TOOLING ====="
for t in compute-sanitizer cuda-memcheck ncu nsys; do
  printf '%-20s %s\n' "$t" "$(command -v $t || echo MISSING)"
done
ls /usr/lib/nvidia-cuda-toolkit/bin/ 2>/dev/null | head -20
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv

echo; echo "===== 2. BLOCKING RUN AT 40 BLOCKS ====="
CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=1 \
  ./target/release/cuda_test --top-module macro_smoke \
  build_native/gatelevel.gv build_native/r.gemparts \
  build_native/in.vcd build_native/out.vcd 40 2>&1 | tail -40

echo; echo "===== 3. SAME RUN AT 2 BLOCKS ====="
CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=1 \
  ./target/release/cuda_test --top-module macro_smoke \
  build_native/gatelevel.gv build_native/r.gemparts \
  build_native/in.vcd /tmp/out2.vcd 2 2>&1 | tail -25

echo; echo "===== 4. SANITIZER (if present) ====="
SAN="$(command -v compute-sanitizer || command -v cuda-memcheck)"
if [ -n "$SAN" ]; then
  "$SAN" --tool memcheck ./target/release/cuda_test --top-module macro_smoke \
    build_native/gatelevel.gv build_native/r.gemparts \
    build_native/in.vcd /tmp/out3.vcd 2 2>&1 | grep -vE "^========= Program hit" | head -60
else
  echo "no sanitizer installed; sections 2 and 3 are the evidence"
fi
echo; echo "===== DONE ====="

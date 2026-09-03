# Sourced by the bench scripts. Sets up the CUDA build environment, and builds.
#
#     source bench/env.sh
#
# This exists because the same two environment mistakes have derailed three
# separate measurement runs on this project, each time silently.
#
# 1. find_cuda_helper (pulled in via cust_raw) does NOT read CUDA_PATH on
#    Linux -- that code path is Windows-only. On Linux it consults
#    CUDA_LIBRARY_PATH plus a short list of hardcoded roots, and every
#    candidate must contain a lib64/ directory. Get it wrong and the build
#    panics with "Could not find a cuda installation", which names neither the
#    variable it wanted nor the directory it looked in.
#
# 2. Cargo.toml declares required-features = ["cuda"] on cuda_test. Cargo
#    SKIPS a binary whose required features are absent -- silently, exit 0. So
#    `cargo build --release` relinks everything except the one target that
#    needs CUDA and reports success, and the next benchmark run measures a
#    stale kernel while looking perfectly healthy.
#
# Neither is exotic. Both are invisible. Encoding them here means they cannot
# recur through someone opening a fresh terminal.

if [ -z "${CUDA_LIBRARY_PATH:-}" ]; then
    for _c in /usr/lib/cuda /usr/local/cuda /opt/cuda; do
        if [ -d "$_c/lib64" ]; then export CUDA_LIBRARY_PATH="$_c"; break; fi
    done
    unset _c
fi

if [ -z "${UCC_CUDA_GENCODE:-}" ]; then
    # Must match this GPU's compute capability. An sm_89 cubin will not load on
    # an sm_86 device, and the failure appears at launch, not at build.
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d ' .')
    [ -n "$_cc" ] && export UCC_CUDA_GENCODE="$_cc"
    unset _cc
fi

# Build everything including cuda_test, and stop on failure. Callers should use
# this rather than calling cargo directly.
gem_build() {
    if [ -z "${CUDA_LIBRARY_PATH:-}" ]; then
        echo "CUDA_LIBRARY_PATH is unset and no CUDA root was found under"
        echo "  /usr/lib/cuda  /usr/local/cuda  /opt/cuda"
        echo "Set it to a directory containing lib64/ and re-run."
        return 1
    fi
    echo "==> build  (CUDA_LIBRARY_PATH=$CUDA_LIBRARY_PATH" \
         "UCC_CUDA_GENCODE=${UCC_CUDA_GENCODE:-unset})"
    if ! cargo build --release --features cuda > /tmp/gem_build.log 2>&1; then
        echo "BUILD FAILED -- refusing to run against stale binaries."
        tail -30 /tmp/gem_build.log
        return 1
    fi
    # A successful build is not proof the binary is current: see note 2 above.
    local stale
    stale=$(find src csrc build.rs Cargo.toml -type f \
              -newer target/release/cuda_test 2>/dev/null | head -5)
    if [ -n "$stale" ]; then
        echo "STALE cuda_test -- these sources are newer than the binary:"
        echo "$stale" | sed 's/^/    /'
        return 1
    fi
    ls -l --time-style=+%H:%M:%S target/release/cuda_test \
          target/release/flatten_test target/release/naive_sim \
          target/release/cut_map_interactive | sed 's/^/    /'
}

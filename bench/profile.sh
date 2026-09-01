#!/usr/bin/env bash
# Deliverable D: Nsight Compute profiling of the shredded baseline against the
# macro-preserving path.
#
#     bash bench/profile.sh > profile.txt 2>&1
#
# Both runs use the SAME stimulus, the SAME num_blocks and the SAME cycle
# count, because the comparison is void otherwise.
#
# The metrics answer three questions the problem statement asks directly:
#
#   dram__throughput...pct_of_peak    Memory bandwidth utilisation. GEM re-reads
#                                     its instruction script from global memory
#                                     EVERY simulated cycle, so this is the
#                                     number that says whether the workload is
#                                     bandwidth-bound at all -- and therefore
#                                     whether shrinking the script can help.
#
#   smsp__thread_inst_executed_       Warp divergence: average active threads
#     per_inst_executed.ratio         per issued instruction, out of 32.
#                                     The boomerang tree is perfectly uniform;
#                                     a macro phase activates one lane per
#                                     macro, so this is expected to FALL.
#
#   l1tex__t_sectors_pipe_lsu_        Local-memory traffic. Must be zero.
#     mem_local_op_ld.sum             -maxrregcount CAPS registers rather than
#                                     letting them grow, so excess pressure
#                                     spills silently instead of failing to
#                                     launch. Launch success proves nothing;
#                                     this does.
#
# NOTE ON PERMISSIONS: profiling counters usually need elevated access. If you
# see ERR_NVGPUCTRPERM, re-run with sudo -E (keeping the environment), or
# enable counters for all users per NVIDIA's documented driver option.
set -uo pipefail
cd "$(dirname "$0")/.."

NB=${NB:-40}            # identical on both sides
CYC=${CYC:-200}         # metrics are rates; a short run keeps ncu replay cheap
V=build_native/in.vcd

METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
gpu__time_duration.sum"

run() {
    local label=$1 dir=$2
    echo
    echo "================================================================"
    echo "  $label   (num_blocks=$NB, cycles=$CYC)"
    echo "================================================================"
    ncu --metrics "$METRICS" \
        --kernel-name simulate_v1_noninteractive_simple_scan \
        --launch-count 1 --target-processes all \
        ./target/release/cuda_test --top-module macro_smoke \
        --max-cycles "$CYC" \
        "$dir/gatelevel.gv" "$dir/r.gemparts" \
        "$V" "/tmp/prof_$(basename "$dir").vcd" "$NB" 2>&1 \
      | grep -E "dram__|sm__warps|smsp__thread|l1tex__|gpu__time|ERR_|error" \
      | sed 's/^ *//'
}

echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "Registers:"
cuobjdump -res-usage "$(find target/release -name libgemcu.a | head -1)" \
  | grep -A1 simulate_v1 | sed 's/^ */  /'

run "SHREDDED BASELINE" build_shred
run "MACRO-PRESERVING"  build_native

echo
echo "================================================================"
echo "  Reading these numbers"
echo "================================================================"
cat <<'EOF'
  dram throughput %   If the BASELINE is low (single digits), the workload is
                      not bandwidth-bound and shrinking the script cannot make
                      it faster. That is a property of the benchmark, not of
                      the implementation, and it should be reported as such.

  thread_inst ratio   Out of 32. Lower in the macro path is expected and is a
                      deliberate trade: idle arithmetic lanes bought in
                      exchange for less memory traffic. Only a good trade when
                      the baseline was actually bandwidth-bound.

  local op ld/st      Must be 0. Anything else means register spilling, and
                      every conclusion above becomes unsafe.
EOF

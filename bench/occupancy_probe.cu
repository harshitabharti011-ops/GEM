// Cooperative-launch headroom on this GPU.
//
// GEM launches simulate_v1_noninteractive_simple_scan with
// cudaLaunchCooperativeKernel, which requires the ENTIRE grid to be resident
// simultaneously. So the usable num_blocks is capped by
//   SM count x blocks-per-SM,
// and blocks-per-SM falls as register pressure rises.
//
// This matters for the macro work specifically: a macro phase that holds many
// operands live inflates register count, occupancy drops, the cooperative grid
// shrinks, and throughput falls in proportion to the parallelism lost -- which
// can quietly cancel out the gain from evaluating macros natively.
//
// Run this on the BASELINE build to record the headroom, then again after the
// macro phase lands. A drop here is the first thing to check if native mode is
// not faster than shredded.
//
//   nvcc -O3 -arch=native -o occupancy_probe occupancy_probe.cu && ./occupancy_probe
#include <cstdio>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    return 1; } } while (0)

// Stand-in for the real kernel at a few register footprints. __launch_bounds__
// caps registers per thread, which is how you trade register pressure for
// occupancy.
template <int MINBLOCKS>
__global__ void __launch_bounds__(256, MINBLOCKS) probe(float *out, int n) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = threadIdx.x * (i + 1);
    for (int i = 0; i < n; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(acc[j], 1.000001f, 1.0f);
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j];
    if (threadIdx.x == 1024) out[0] = s;   // never taken; keeps it live
}

template <int MB>
static void report(const char *label, int sms) {
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, probe<MB>, 256, 0);
    cudaFuncAttributes at;
    cudaFuncGetAttributes(&at, probe<MB>);
    printf("  %-22s regs/thread %3d   blocks/SM %2d   max coop grid %4d\n",
           label, at.numRegs, blocks_per_sm, blocks_per_sm * sms);
}

int main() {
    int dev = 0; CHK(cudaGetDevice(&dev));
    cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p, dev));
    printf("device            : %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("SMs               : %d\n", p.multiProcessorCount);
    printf("regs per SM       : %d\n", p.regsPerMultiprocessor);
    printf("shared per SM     : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("shared per block  : %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("cooperative launch: %s\n",
           p.cooperativeLaunch ? "supported" : "NOT SUPPORTED");
    if (!p.cooperativeLaunch) {
        printf("\nFATAL: GEM cannot run here -- it launches cooperatively.\n");
        return 2;
    }
    printf("\ncooperative grid limit at 256 threads/block:\n");
    report<1>("low occupancy",  p.multiProcessorCount);
    report<4>("medium",         p.multiProcessorCount);
    report<8>("high occupancy", p.multiProcessorCount);
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, probe<1>, 256, 0);
    const int advised = 2 * p.multiProcessorCount;
    printf("\nGEM's usage.md advises num_blocks = 2 x SMs = %d.\n", advised);
    printf("Headroom: %d / %d = %.1fx before the cooperative grid is the limit.\n",
           bps * p.multiProcessorCount, advised,
           (double)(bps * p.multiProcessorCount) / advised);
    // The build pins -maxrregcount=128 (see build.rs), which is what actually
    // determines GEM's occupancy -- not this toy kernel's 15 registers.
    printf("\nGEM builds with -maxrregcount=128. At 256 threads that is\n");
    for (int r = 32; r <= 128; r += 32) {
        int b = p.regsPerMultiprocessor / (256 * r);
        if (b > 8) b = 8;
        printf("    %3d regs/thread -> %d blocks/SM -> max coop grid %4d%s\n",
               r, b, b * p.multiProcessorCount,
               (r == 128) ? "   <- the actual cap" : "");
    }
    printf("  so the real ceiling is %d, and usage.md advises %d. There is no\n"
           "  headroom: they are the same number. And because maxrregcount\n"
           "  CAPS registers rather than letting them grow, a heavy macro phase\n"
           "  does not fail to launch -- it SPILLS to local memory and quietly\n"
           "  gets slower. Watch l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\n"
           "  not the launch status.\n",
           (p.regsPerMultiprocessor / (256 * 128)) * p.multiProcessorCount,
           2 * p.multiProcessorCount);

    printf("\nCAVEAT: the three rows above are likely identical. This probe\n"
           "kernel is too light to be register-limited, so blocks/SM is pinned\n"
           "at the architectural ceiling (max threads per SM / 256) and\n"
           "__launch_bounds__ changes nothing. The headroom figure is real; the\n"
           "register sensitivity is NOT demonstrated here. Re-run this against\n"
           "the real kernel once the macro phase exists -- cudaFuncGetAttributes\n"
           "on simulate_v1_noninteractive_simple_scan is the number that matters.\n");
    return 0;
}

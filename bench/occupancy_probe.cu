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
    printf("\nGEM's usage.md advises num_blocks = 2 x SMs = %d.\n",
           2 * p.multiProcessorCount);
    printf("That is only reachable while the kernel stays at or below\n"
           "%d blocks/SM. If the macro phase pushes registers past that,\n"
           "num_blocks must come down and the grid loses parallelism.\n", 2);
    return 0;
}

// Memory-hierarchy facts for this GPU, measured rather than looked up.
//
//   nvcc -O3 -o memory_probe bench/memory_probe.cu && ./memory_probe
//
// Why this exists. Nsight reports the shredded baseline at 46.91 % of peak
// DRAM throughput and the macro-preserving path at 0.04 % -- a factor of over
// a thousand, for a script that shrank only 4.53x. A linear bandwidth argument
// cannot produce that gap, so something discontinuous happened, and the
// obvious candidate is L2 residency: GEM re-reads its whole instruction script
// every simulated cycle, so a script that FITS in L2 is read from DRAM once
// and served from cache forever after, while one that does not is re-fetched
// from DRAM every cycle.
//
// That hypothesis is only worth stating if the L2 capacity is a measurement.
// Quoting a figure from a spec sheet for "the RTX 3050" is exactly the kind of
// step that turned a reasonable inference into a wrong claim earlier in this
// project: the laptop parts differ from the desktop parts, and the 6 GB
// variant differs from the 4 GB one. cudaGetDeviceProperties knows.
//
// Peak bandwidth is derived as busWidth x memoryClock x 2 (DDR). Treat it as
// indicative: it is the theoretical pin rate, and Nsight's "% of peak" uses
// its own sustained-rate model, so the two need not agree exactly.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        printf("cannot query device 0\n");
        return 1;
    }

    const double l2_kib = p.l2CacheSize / 1024.0;
    // memoryClockRate is in kHz; busWidth in bits; DDR transfers twice a cycle.
    const double bw_gbs =
        (double)p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) * 2.0 / 1e9;

    printf("Device                    : %s (sm_%d%d)\n",
           p.name, p.major, p.minor);
    printf("SMs                       : %d\n", p.multiProcessorCount);
    printf("Registers per SM          : %d\n", p.regsPerMultiprocessor);
    printf("Max threads per SM        : %d  (%d warps)\n",
           p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / 32);
    printf("Shared memory per SM      : %zu bytes\n",
           (size_t)p.sharedMemPerMultiprocessor);
    printf("L2 cache                  : %d bytes  (%.0f KiB, %.2f MiB)\n",
           p.l2CacheSize, l2_kib, l2_kib / 1024.0);
    printf("Memory bus width          : %d bit\n", p.memoryBusWidth);
    printf("Memory clock              : %.0f MHz\n", p.memoryClockRate / 1e3);
    printf("Theoretical peak bandwidth: %.1f GB/s\n", bw_gbs);

    // The two measured script sizes from bench/throughput.sh at LANES=16.
    // Hard-coded deliberately: this probe answers one question -- does either
    // script fit? -- and a number typed here is visible and checkable, whereas
    // a number parsed from a log is neither.
    struct { const char *name; double kib; } script[] = {
        { "shredded baseline (527,360 u32)", 527360.0 * 4 / 1024.0 },
        { "macro-preserving (116,340 u32)",  116340.0 * 4 / 1024.0 },
    };

    printf("\nScript residency at LANES=16 (script is re-read every cycle):\n");
    for (auto &s : script) {
        const double frac = s.kib / l2_kib;
        printf("  %-34s %8.1f KiB  = %5.2f x L2  -> %s\n",
               s.name, s.kib, frac,
               frac <= 1.0 ? "FITS: DRAM once, then cache-resident"
                           : "DOES NOT FIT: re-fetched from DRAM every cycle");
    }

    printf("\nIf exactly one of the two fits, the DRAM-throughput gap Nsight\n"
           "reports is a residency transition, not a linear bandwidth saving --\n"
           "which means the benefit has a CLIFF at the L2 capacity rather than\n"
           "scaling smoothly with script size, and the design target is to get\n"
           "the script under that capacity rather than merely smaller.\n");
    return 0;
}

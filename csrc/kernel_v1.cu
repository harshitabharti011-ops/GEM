// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "kernel_v1_impl.cuh"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

extern "C"
void simulate_v1_noninteractive_simple_scan_cuda(
  usize num_blocks,
  usize num_major_stages,
  const usize *blocks_start,
  const u32 *blocks_data,
  u32 *sram_data,
  usize num_cycles,
  usize state_size,
  u32 *states_noninteractive,
  const u32 *macro_desc,
  const usize *macro_desc_start,
  u64 *macro_word_state
  )
{
  // cudaLaunchCooperativeKernel takes a void** -- the compiler cannot check
  // this against the kernel signature. A mismatch is silent garbage, not an
  // error, so this array and the __global__ declaration must be edited
  // together, always.
  void *arg_ptrs[11] = {
    (void *)&num_blocks, (void *)&num_major_stages,
    (void *)&blocks_start, (void *)&blocks_data,
    (void *)&sram_data, (void *)&num_cycles, (void *)&state_size,
    (void *)&states_noninteractive,
    (void *)&macro_desc, (void *)&macro_desc_start, (void *)&macro_word_state
  };

  // GEM launches cooperatively, so the WHOLE grid must be co-resident or the
  // launch fails outright. Check it here and say so precisely -- the raw
  // failure is cudaErrorCooperativeLaunchTooLarge, which does not hint that
  // register pressure in the macro phase is the likely cause.
  {
    int dev = 0, sms = 0, bps = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bps, (void *)simulate_v1_noninteractive_simple_scan, 256, 0);
    if ((int)num_blocks > bps * sms) {
      printf("FATAL: %d blocks requested but only %d can be co-resident "
             "(%d SMs x %d blocks/SM). Lower num_blocks, or reduce register "
             "pressure in the macro phase.\n",
             (int)num_blocks, bps * sms, sms, bps);
      exit(EXIT_FAILURE);
    }
  }
  checkCudaErrors(cudaLaunchCooperativeKernel(
    (void *)simulate_v1_noninteractive_simple_scan, num_blocks, 256,
    arg_ptrs, 0, (cudaStream_t)0
    ));
}

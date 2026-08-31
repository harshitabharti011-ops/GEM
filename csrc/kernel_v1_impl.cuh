// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <crates/ulib/includes.hpp>
#include <cstdio>
#include <cooperative_groups.h>
#include "macros.cuh"

struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) {
    *this = *t;
  }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) {
    *this = *t;
  }
};

__device__ void simulate_block_v1(
  const u32 *__restrict__ script,
  usize script_size,
  const u32 *__restrict__ input_state,
  u32 *__restrict__ output_state,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ shared_metadata,
  u32 *__restrict__ shared_writeouts,
  u32 *__restrict__ shared_state,
  const u32 *__restrict__ macro_desc,
  const usize *__restrict__ macro_desc_start,
  u64 *__restrict__ macro_word_state
  )
{
  int script_pi = 0;
  while(true) {
    VectorRead2 t2_1, t2_2;
    VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;
    const int part_start = script_pi;
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    script_pi += 256;
    t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
    __syncthreads();
    int num_stages = shared_metadata[0];
    if(!num_stages) {
      break;
    }
    int is_last_part = shared_metadata[1];
    int num_ios = shared_metadata[2];
    int io_offset = shared_metadata[3];
    int num_srams = shared_metadata[4];
    int sram_offset = shared_metadata[5];
    int num_global_read_rounds = shared_metadata[6];
    int num_output_duplicates = shared_metadata[7];
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if(threadIdx.x % 2 == 0) {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    }
    else {
      writeout_hook_i = writeout_hook_i >> 16;
    }

    t4_1.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds) + threadIdx.x);
    t4_2.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4) + threadIdx.x);
    t4_3.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 2) + threadIdx.x);
    t4_4.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 3) + threadIdx.x);
    t4_5.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 4) + threadIdx.x);
    u32 t_global_rd_state = 0;
    for(int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
      u32 idx = t2_1.c1;
      u32 mask = t2_1.c2;
      script_pi += 256 * 2;
      t2_2.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }

      if(gr_i + 1 >= num_global_read_rounds) break;
      idx = t2_2.c1;
      mask = t2_2.c2;
      script_pi += 256 * 2;
      t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    __syncthreads();

    for(int bs_i = 0; bs_i < num_stages; ++bs_i) {
      u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;
#define GEMV1_SHUF_INPUT_K(k_outer, k_inner, t_shuffle) {           \
        u32 k = k_outer * 4 + k_inner;                              \
        u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);          \
        u32 t_shuffle_2_idx = t_shuffle >> 16;                      \
                                                                    \
        hier_input |= (shared_state[t_shuffle_1_idx >> 5] >>        \
                       (t_shuffle_1_idx & 31) & 1) << (k * 2);      \
        hier_input |= (shared_state[t_shuffle_2_idx >> 5] >>        \
                       (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);  \
      }
#define GEMV1_SHUF_INPUT_K_4(k_outer, t_shuffle) {    \
        GEMV1_SHUF_INPUT_K(k_outer, 0, t_shuffle.c1); \
        GEMV1_SHUF_INPUT_K(k_outer, 1, t_shuffle.c2); \
        GEMV1_SHUF_INPUT_K(k_outer, 2, t_shuffle.c3); \
        GEMV1_SHUF_INPUT_K(k_outer, 3, t_shuffle.c4); \
      }
      script_pi += 256 * 4 * 5;
      GEMV1_SHUF_INPUT_K_4(0, t4_1);
      t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(1, t4_2);
      t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(2, t4_3);
      t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(3, t4_4);
      t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_INPUT_K
#undef GEMV1_SHUF_INPUT_K_4
      hier_flag_xora = t4_5.c1;
      hier_flag_xorb = t4_5.c2;
      hier_flag_orb = t4_5.c3;
      t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      __syncthreads();

      // hier[0]
      if(threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
        shared_state[threadIdx.x] = ret;
      }
      __syncthreads();
      // hier[1..3]
      u32 tmp_cur_hi;
      for(int hi = 1; hi <= 3; ++hi) {
        int hier_width = 1 << (7 - hi);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          tmp_cur_hi = ret;
          shared_state[threadIdx.x] = ret;
        }
        __syncthreads();
      }
      // hier[4..7], within the first warp.
      if(threadIdx.x < 32) {
        for(int hi = 4; hi <= 7; ++hi) {
          int hier_width = 1 << (7 - hi);
          u32 hier_input_a = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
          u32 hier_input_b = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
          if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            tmp_cur_hi = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          }
        }
        u32 v1 = __shfl_down_sync(0xffffffff, tmp_cur_hi, 1);
        // hier[8..12]
        if(threadIdx.x == 0) {
          u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
          u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
          u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
          u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12 /* 0b1100 */;
          u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2 /* 0b10 */;
          tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
        }
        shared_state[threadIdx.x] = tmp_cur_hi;
      }
      __syncthreads();

      // write out
      if((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
      }
    }
    __syncthreads();

    // sram & duplicate permutation
    u32 sram_duplicate_t = 0;
#define GEMV1_SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                          \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);      \
      u32 t_shuffle_2_idx = t_shuffle >> 16;                  \
                                                              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>            \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>            \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);          \
    }
#define GEMV1_SHUF_SRAM_DUPL_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_SRAM_DUPL_K_4(0, t4_1);
    t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(1, t4_2);
    t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(2, t4_3);
    t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(3, t4_4);
    t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_SRAM_DUPL_K_4
#undef GEMV1_SHUF_SRAM_DUPL_K
    sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
    t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

    // sram read fires here.
    u32 *ram = nullptr;
    u32 r, w0;
    u32 port_w_addr_iv, port_w_wr_en, port_w_wr_data_iv;
    if(threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t;
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
        ? 0xffffffff : (0xffffffff >> (last_tid - num_srams * 4));
      port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);

      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13);
        // u32 sram_ed = sram_st + (1 << 13);
        u32 port_r_addr_iv = addrs & 0xffff;
        port_w_addr_iv = addrs >> 16;

        ram = sram_data + sram_st;
        r = ram[port_r_addr_iv];
        w0 = ram[port_w_addr_iv];
      }
    }
    // __syncthreads();

    // clock enable permutation
    u32 clken_perm = 0;
#define GEMV1_SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                      \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);  \
      u32 t_shuffle_2_idx = t_shuffle >> 16;              \
                                                          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>        \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>        \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);      \
    }
#define GEMV1_SHUF_CLKEN_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_CLKEN_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_CLKEN_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_CLKEN_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_CLKEN_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_CLKEN_K_4(0, t4_1);
    GEMV1_SHUF_CLKEN_K_4(1, t4_2);
    GEMV1_SHUF_CLKEN_K_4(2, t4_3);
    GEMV1_SHUF_CLKEN_K_4(3, t4_4);
#undef GEMV1_SHUF_CLKEN_K
#undef GEMV1_SHUF_CLKEN_K_4

    // sram commit
    if(threadIdx.x < num_srams * 4) {
      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_output_duplicates) {
      shared_writeouts[num_ios - num_srams - num_output_duplicates + (threadIdx.x - num_srams * 4)] = sram_duplicate_t;
    }

    __syncthreads();
    u32 writeout_inv = shared_writeouts[threadIdx.x];

    clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
    writeout_inv ^= t4_5.c3;

    // ---------------------------------------------------------------------
    // macro phase
    //
    // Runs after the boolean write-outs have settled in shared_writeouts and
    // before they are committed to output_state, so a macro reads its operands
    // from what this partition just produced and drops its results into the
    // same write-out block. The normal commit path then carries them out: no
    // separate store, and the clock-enable machinery applies unchanged.
    //
    // No atomics are needed for the result scatter, which is worth stating
    // because it is not obvious. flatten.rs gives every macro its OWN whole u32
    // write-out slots -- ceil(outputs/32) of them -- so two macros can never
    // contend for the same word. That falls out of the slot allocation rather
    // than being lucky.
    //
    // One thread per macro; batches are type-homogeneous by construction
    // (pe.rs groups by kind), so every active lane in a batch takes the same
    // switch arm and the warp does not diverge across kinds.
    // ---------------------------------------------------------------------
    __syncthreads();
    {
      const int num_macro_batches = shared_metadata[8];
      const u32 *msec = script + part_start + shared_metadata[9];
      for(int b = 0; b < num_macro_batches; ++b) {
        const u32 kind  = msec[1];
        const u32 count = msec[2];
        const u32 *slots = msec + 3;
        if(threadIdx.x < count) {
          const u32 *d = macro_desc + macro_desc_start[slots[threadIdx.x]];
          const u32 word_idx = d[1];
          const u32 clk_pos  = d[2];
          const u32 n_in     = d[3];
          const u32 n_out    = d[4];
          const u32 *in_pos  = d + 5;
          const u32 *out_pos = in_pos + n_in;

          // a state bit position maps to (write-out slot, bit) as
          //   slot = (pos >> 5) - io_offset,   bit = pos & 31
#define GEM_RD_BIT(pos)                                                 \
          (((pos) == GEM_NO_BIT) ? 0u :                                 \
           ((shared_writeouts[((pos) >> 5) - io_offset] >> ((pos) & 31)) & 1u))

          const bool clk_en = (clk_pos == GEM_NO_BIT)
                              ? true : (GEM_RD_BIT(clk_pos) != 0u);
          u64 o_word = 0;

          if(kind == GEM_KIND_DSP48E2) {
            u64 a = 0, bb = 0, c = 0, dd = 0;
            u32 k = 0;
            for(u32 i = 0; i < GEM_DSP_A_BITS; ++i, ++k)
              a  |= (u64)GEM_RD_BIT(in_pos[k]) << i;
            for(u32 i = 0; i < GEM_DSP_B_BITS; ++i, ++k)
              bb |= (u64)GEM_RD_BIT(in_pos[k]) << i;
            for(u32 i = 0; i < GEM_DSP_C_BITS; ++i, ++k)
              c  |= (u64)GEM_RD_BIT(in_pos[k]) << i;
            for(u32 i = 0; i < GEM_DSP_D_BITS; ++i, ++k)
              dd |= (u64)GEM_RD_BIT(in_pos[k]) << i;
            const u32 pre  = GEM_RD_BIT(in_pos[k]); ++k;
            const u32 mode = GEM_RD_BIT(in_pos[k])
                           | (GEM_RD_BIT(in_pos[k + 1]) << 1);

            const long long p_cur = (long long)macro_word_state[word_idx];
            const long long p_nxt =
              gem_dsp48e2(a, bb, c, dd, pre != 0, mode, p_cur);
            // PREG is clocked: hold the old value when the enable is low.
            const long long p_com = clk_en ? p_nxt : p_cur;
            macro_word_state[word_idx] = (u64)p_com;
            o_word = (u64)p_com & ((1ull << GEM_DSP_P_BITS) - 1);
          }
          else if(kind == GEM_KIND_CARRY4) {
            u32 S = 0, DI = 0;
            for(u32 i = 0; i < GEM_CARRY4_BITS; ++i)
              S  |= GEM_RD_BIT(in_pos[i]) << i;
            for(u32 i = 0; i < GEM_CARRY4_BITS; ++i)
              DI |= GEM_RD_BIT(in_pos[GEM_CARRY4_BITS + i]) << i;
            const bool cin = GEM_RD_BIT(in_pos[2 * GEM_CARRY4_BITS]) != 0;
            const bool cyi = GEM_RD_BIT(in_pos[2 * GEM_CARRY4_BITS + 1]) != 0;
            u32 CO, O;
            gem_carry4(S, DI, cin, cyi, &CO, &O);
            // output slots are CO[3:0] then O[3:0] -- MacroKind::output_slot
            o_word = (u64)CO | ((u64)O << GEM_CARRY4_BITS);
          }
          else {  // GEM_KIND_SRLC32E
            const u32 d_in = GEM_RD_BIT(in_pos[0]);
            const u32 ce   = GEM_RD_BIT(in_pos[1]);
            u32 addr = 0;
            for(u32 i = 0; i < 5; ++i)
              addr |= GEM_RD_BIT(in_pos[2 + i]) << i;
            const u32 sr = (u32)macro_word_state[word_idx];
            bool q, q31;
            // read the CURRENT state, then shift: GEM's read-old / commit-new
            // discipline, exactly as the DFF path does it
            gem_srlc32e_read(sr, addr, &q, &q31);
            o_word = (u64)(q ? 1u : 0u) | ((u64)(q31 ? 1u : 0u) << 1);
            if(clk_en)
              macro_word_state[word_idx] =
                (u64)gem_srlc32e_edge(sr, d_in != 0, ce != 0);
          }

          // Scatter. This macro owns these words outright, so a plain
          // read-modify-write is safe -- see the note above.
          for(u32 i = 0; i < n_out; ++i) {
            const u32 pos = out_pos[i];
            if(pos == GEM_NO_BIT) continue;
            const u32 slot = (pos >> 5) - io_offset;
            const u32 bit  = pos & 31;
            const u32 v    = (u32)((o_word >> i) & 1ull);
            shared_writeouts[slot] =
              (shared_writeouts[slot] & ~(1u << bit)) | (v << bit);
          }
#undef GEM_RD_BIT
        }
        __syncthreads();
        msec += 3 + count;
      }
    }
    __syncthreads();
    writeout_inv = shared_writeouts[threadIdx.x] ^ t4_5.c3;

    if(threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;
    }

    // Jump past the macro section instead of walking off the end of it: the
    // assert below requires script_pi to land exactly on script_size.
    script_pi = part_start + (int)shared_metadata[10];

    if(is_last_part) break;
  }
  assert(script_size == script_pi);
}

__global__ void simulate_v1_noninteractive_simple_scan(
  usize num_blocks,
  usize num_major_stages,
  const usize *__restrict__ blocks_start,
  const u32 *__restrict__ blocks_data,
  u32 *__restrict__ sram_data,
  usize num_cycles,
  usize state_size,
  u32 *__restrict__ states_noninteractive,
  const u32 *__restrict__ macro_desc,
  const usize *__restrict__ macro_desc_start,
  u64 *__restrict__ macro_word_state
  )
{
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 script_starts[32], script_sizes[32];
  assert(num_major_stages <= 32);
  if(threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] - script_starts[threadIdx.x];
  }
  __syncthreads();
  for(usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    for(usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(
        blocks_data + script_starts[stage_i],
        script_sizes[stage_i],
        states_noninteractive + cycle_i * state_size,
        states_noninteractive + (cycle_i + 1) * state_size,
        sram_data,
        shared_metadata, shared_writeouts, shared_state,
        macro_desc, macro_desc_start, macro_word_state
        );
      cooperative_groups::this_grid().sync();
    }
  }
}

// SPDX-License-Identifier: Apache-2.0
//
// Cycle-accurate models of the three GEM word-level macros.
//
// ONE source, compiled twice: as ordinary C++ for the CPU golden model and
// testbenches, and as __device__ code inside the CUDA kernel. That is
// deliberate -- a separate CPU reference would drift from the GPU path, and
// the whole verification argument rests on them being the same code.
//
// Every equation below is transcribed from the problem statement, and matches
// synth/gem_macros_behav.v line for line. If one changes, change all three.
//
// Conventions:
//   * Operands arrive as raw bit patterns from the PACK step -- NOT
//     pre-sign-extended. Each model sign-extends what it needs, so callers
//     never have to know a port's width.
//   * All arithmetic is signed two's complement in int64_t. The widest
//     intermediate is the 46-bit product, so nothing can overflow.
//   * No branching on data in the hot paths: CARRY4 is a fully unrolled
//     4-iteration chain and the DSP's only switch is on MODE, which is
//     warp-uniform within a batch.

#ifndef GEM_MACROS_CUH
#define GEM_MACROS_CUH

#include <stdint.h>

#if defined(__CUDACC__)
  #define GEM_HD __host__ __device__ __forceinline__
#else
  #define GEM_HD inline
#endif

// ---------------------------------------------------------------------------
// widths -- must match src/aigpdk.rs and synth/gem_macros.v
// ---------------------------------------------------------------------------
#define GEM_DSP_A_BITS     27
#define GEM_DSP_B_BITS     18
#define GEM_DSP_C_BITS     48
#define GEM_DSP_D_BITS     27
#define GEM_DSP_AD_BITS    27   /* pre-adder output wraps here */
#define GEM_DSP_M_BITS     45   /* 27 x 18 product             */
#define GEM_DSP_P_BITS     48
#define GEM_CARRY4_BITS     4
#define GEM_SRL_DEPTH      32

// ---------------------------------------------------------------------------
// script/descriptor contract -- MUST match src/macro_layout.rs
//   kind_code():  Dsp48e2 = 0, Srlc32e = 1, Carry4 = 2
//   NO_BIT:       u32::MAX, an input tied low or an output nothing reads
// ---------------------------------------------------------------------------
#define GEM_KIND_DSP48E2  0u
#define GEM_KIND_SRLC32E  1u
#define GEM_KIND_CARRY4   2u
#define GEM_NO_BIT        0xFFFFFFFFu
/* An input tied to constant ONE. Distinct from GEM_NO_BIT because the AIG
   keeps both constants on aigpin 0 -- iv 0 is false, iv 1 is true -- so one
   sentinel for both makes a hard-wired 1 read as 0, silently disabling
   USE_PREADD and collapsing MODE 1 to MODE 0. Must match
   macro_layout::CONST_ONE. */
#define GEM_CONST_ONE     0xFFFFFFFEu

// ---------------------------------------------------------------------------
// sign extension
// ---------------------------------------------------------------------------

/// Sign-extend the low `bits` of `v` to a full int64_t.
///
/// Shift-left-then-arithmetic-shift-right. C++20 mandates the right shift on a
/// signed type be arithmetic; every compiler that matters did so long before.
GEM_HD int64_t gem_sext(uint64_t v, int bits) {
    const int sh = 64 - bits;
    return (int64_t)(v << sh) >> sh;
}

/// Wrap a value back into the DSP's 48-bit accumulator.
GEM_HD int64_t gem_sext48(int64_t v) {
    return (v << (64 - GEM_DSP_P_BITS)) >> (64 - GEM_DSP_P_BITS);
}

// ---------------------------------------------------------------------------
// A. DSP48E2 (simplified subset)
//
//   AD = use_preadd ? (A + D) : A          A, D are 27-bit signed
//   M  = AD * B                            B is 18-bit signed; M fits in 46
//   MODE 0: P_next = C                     bypass
//   MODE 1: P_next = M                     multiply only
//   MODE 2: P_next = P_current + M         multiply-accumulate
//
// All input/internal registers are combinational (PS: set to 0); only PREG is
// clocked, so P_next is committed on the global rising edge by the caller.
// OVERFLOW / UNDERFLOW pins are ignored, as the PS permits.
// ---------------------------------------------------------------------------
GEM_HD int64_t gem_dsp48e2(uint64_t a_raw, uint64_t b_raw,
                           uint64_t c_raw, uint64_t d_raw,
                           bool use_preadd, uint32_t mode,
                           int64_t p_cur) {
    const int64_t a = gem_sext(a_raw, GEM_DSP_A_BITS);
    const int64_t b = gem_sext(b_raw, GEM_DSP_B_BITS);
    const int64_t c = gem_sext(c_raw, GEM_DSP_C_BITS);
    const int64_t d = gem_sext(d_raw, GEM_DSP_D_BITS);

    // The pre-adder output AD is 27 bits and WRAPS -- it does not widen to 28.
    // That is what real DSP48E2 silicon does (the pre-adder is a 27-bit
    // adder), and it is exactly why the PS specifies a 45-bit product rather
    // than 46. Letting A+D widen changes the result whenever the pre-adder
    // overflows: A = D = 2^26-1 gives AD = -2 in hardware, not +134217726.
    const int64_t ad = gem_sext(
        (uint64_t)(use_preadd ? (a + d) : a), GEM_DSP_AD_BITS);
    const int64_t m  = gem_sext((uint64_t)(ad * b), GEM_DSP_M_BITS);

    int64_t p;
    switch (mode) {
        case 0:  p = c;          break;
        case 1:  p = m;          break;
        default: p = p_cur + m;  break;
    }
    return gem_sext48(p);
}

// ---------------------------------------------------------------------------
// B. CARRY4
//
//   C[0]   = CYINIT | CIN                  (only one is active in valid RTL)
//   C[i+1] = (S[i] & C[i]) | (~S[i] & DI[i])
//   O[i]   = S[i] ^ C[i]
//   CO[i]  = C[i+1]
//
// Purely combinational -- no state, no clock. Four unrolled iterations, no
// data-dependent branch, so a warp of CARRY4s never diverges.
// ---------------------------------------------------------------------------
GEM_HD void gem_carry4(uint32_t s, uint32_t di, bool cin, bool cyinit,
                       uint32_t *co_out, uint32_t *o_out) {
    const uint32_t mask = (1u << GEM_CARRY4_BITS) - 1u;
    s  &= mask;
    di &= mask;

    const uint32_t c0 = (uint32_t)(cyinit | cin) & 1u;
    uint32_t c  = c0;
    uint32_t co = 0;
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int i = 0; i < GEM_CARRY4_BITS; ++i) {
        const uint32_t cn = ((~s >> i) & di >> i & 1u) | (((s >> i) & 1u) & c);
        co |= cn << i;
        c = cn;
    }
    // C = { co[2:0], c0 } -- each bit's incoming carry
    *co_out = co;
    *o_out  = (s ^ (c0 | (co << 1))) & mask;
}

// ---------------------------------------------------------------------------
// C. SRLC32E
//
//   On the global rising edge, if CE: state <<= 1, D into index 0 (LSB->MSB).
//   Q   = state[A[4:0]]   combinational read of the CURRENT state
//   Q31 = state[31]       combinational cascade output
//
// Split into read and edge halves so the caller can honour GEM's existing
// read-old / commit-new discipline, exactly as the DFF path does.
// ---------------------------------------------------------------------------
GEM_HD void gem_srlc32e_read(uint32_t state, uint32_t a,
                             bool *q_out, bool *q31_out) {
    *q_out   = (state >> (a & (GEM_SRL_DEPTH - 1))) & 1u;
    *q31_out = (state >> (GEM_SRL_DEPTH - 1)) & 1u;
}

GEM_HD uint32_t gem_srlc32e_edge(uint32_t state, bool d, bool ce) {
    // Branchless: a warp of SRLs with mixed CE values must not diverge.
    const uint32_t shifted = (state << 1) | (uint32_t)d;
    const uint32_t m = (uint32_t)0 - (uint32_t)ce;   // all ones iff ce
    return (shifted & m) | (state & ~m);
}

#endif // GEM_MACROS_CUH

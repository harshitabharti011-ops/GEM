// Emits macro results for a deterministic vector set, one per line.
// csrc/tests/reference_model.py computes the same set independently from the
// problem-statement equations; verify_macros.sh diffs the two.
//
//   g++ -std=c++17 -O2 -I.. dump_vectors.cpp -o dump && ./dump
#include "../macros.cuh"
#include <cstdio>

static uint64_t st = 0x243F6A8885A308D3ULL;
static uint64_t rnd() {                       // must match the Python LCG
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st;
}
static uint64_t rnd_bits(int n) { return rnd() >> (64 - n); }

int main() {
    // --- CARRY4: exhaustive. 4-bit S x 4-bit DI x CIN x CYINIT = 1024 cases.
    for (uint32_t s = 0; s < 16; ++s)
      for (uint32_t di = 0; di < 16; ++di)
        for (int cin = 0; cin < 2; ++cin)
          for (int cyi = 0; cyi < 2; ++cyi) {
            uint32_t co, o;
            gem_carry4(s, di, cin, cyi, &co, &o);
            printf("C4 %u %u %d %d %u %u\n", s, di, cin, cyi, co, o);
          }

    // --- DSP: directed signed edges first, then random, each accumulated
    //     over 4 cycles so PREG feedback and 48-bit wrap are exercised.
    const int64_t A_MIN = -((int64_t)1 << (GEM_DSP_A_BITS - 1));
    const int64_t B_MIN = -((int64_t)1 << (GEM_DSP_B_BITS - 1));
    const int64_t A_MAX = ((int64_t)1 << (GEM_DSP_A_BITS - 1)) - 1;
    const int64_t B_MAX = ((int64_t)1 << (GEM_DSP_B_BITS - 1)) - 1;
    struct D { uint64_t a, b, c, d; int pre; uint32_t mode; };
    D directed[] = {
        {(uint64_t)A_MIN, (uint64_t)B_MIN, 0, 0, 0, 2},
        {(uint64_t)A_MIN, (uint64_t)B_MAX, 0, 0, 0, 2},
        {(uint64_t)A_MAX, (uint64_t)B_MIN, 0, 0, 1, 2},
        {(uint64_t)A_MAX, (uint64_t)B_MAX, 0, 0, 1, 2},
        {(uint64_t)A_MAX, (uint64_t)B_MAX, (uint64_t)-1, (uint64_t)A_MAX, 1, 0},
        {(uint64_t)A_MIN, (uint64_t)B_MIN, 0, (uint64_t)A_MIN, 1, 1},
        {1, 1, 0, 0, 0, 2},
        {0, 0, 0, 0, 0, 2},
    };
    int idx = 0;
    for (const D &v : directed) {
        int64_t p = 0;
        for (int cyc = 0; cyc < 4; ++cyc) {
            p = gem_dsp48e2(v.a, v.b, v.c, v.d, v.pre, v.mode, p);
            printf("DSP %d %llu %llu %llu %llu %d %u %d %lld\n", idx,
                   (unsigned long long)(v.a & ((1ULL<<GEM_DSP_A_BITS)-1)),
                   (unsigned long long)(v.b & ((1ULL<<GEM_DSP_B_BITS)-1)),
                   (unsigned long long)(v.c & ((1ULL<<GEM_DSP_C_BITS)-1)),
                   (unsigned long long)(v.d & ((1ULL<<GEM_DSP_D_BITS)-1)),
                   v.pre, v.mode, cyc, (long long)p);
        }
        ++idx;
    }
    for (int i = 0; i < 400; ++i, ++idx) {
        uint64_t a = rnd_bits(GEM_DSP_A_BITS), b = rnd_bits(GEM_DSP_B_BITS);
        uint64_t c = rnd_bits(GEM_DSP_C_BITS), d = rnd_bits(GEM_DSP_D_BITS);
        int pre = (int)(rnd() & 1); uint32_t mode = (uint32_t)(rnd() % 3);
        int64_t p = 0;
        for (int cyc = 0; cyc < 4; ++cyc) {
            p = gem_dsp48e2(a, b, c, d, pre, mode, p);
            printf("DSP %d %llu %llu %llu %llu %d %u %d %lld\n", idx,
                   (unsigned long long)a, (unsigned long long)b,
                   (unsigned long long)c, (unsigned long long)d,
                   pre, mode, cyc, (long long)p);
        }
    }

    // --- SRL: random start state, a shift sequence, and every read address.
    for (int i = 0; i < 60; ++i) {
        uint32_t state = (uint32_t)rnd_bits(32);
        for (int step = 0; step < 6; ++step) {
            int d = (int)(rnd() & 1), ce = (int)(rnd() & 1);
            // read BEFORE the edge -- GEM's read-old / commit-new discipline
            for (uint32_t a = 0; a < GEM_SRL_DEPTH; ++a) {
                bool q, q31;
                gem_srlc32e_read(state, a, &q, &q31);
                printf("SRL %d %d %u %u %d %d\n", i, step, state, a, (int)q, (int)q31);
            }
            state = gem_srlc32e_edge(state, d, ce);
            printf("SRLE %d %d %d %d %u\n", i, step, d, ce, state);
        }
    }
    return 0;
}

#!/usr/bin/env python3
"""Independent reference model for the three GEM macros.

Written straight from the problem-statement equations, in a different language
and a different style from csrc/macros.cuh, so that a shared misreading is
unlikely to survive the comparison. verify_macros.sh diffs this against the
C++ model over an identical vector set.

No golden reference code is provided by the PS, so this pair IS the oracle:
two independent transcriptions that must agree bit for bit.
"""
M64 = (1 << 64) - 1

class LCG:
    """Must match the generator in dump_vectors.cpp exactly."""
    def __init__(self, seed): self.s = seed
    def __call__(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & M64
        return self.s
    def bits(self, n): return self() >> (64 - n)

def sext(v, bits):
    """Interpret the low `bits` of v as two's complement."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v

def wrap(v, bits):
    """Wrap into a `bits`-wide two's complement value."""
    return sext(v & ((1 << bits) - 1), bits)

# --- A. DSP48E2 -------------------------------------------------------------
def dsp48e2(a, b, c, d, use_preadd, mode, p_cur):
    A, B, C, D = sext(a, 27), sext(b, 18), sext(c, 48), sext(d, 27)
    # AD is a 27-bit pre-adder output and wraps; that is why the PS says the
    # product is 45 bits and not 46.
    AD = wrap(A + D, 27) if use_preadd else A
    M = wrap(AD * B, 45)
    if   mode == 0: P = C
    elif mode == 1: P = M
    else:           P = p_cur + M
    return wrap(P, 48)

# --- B. CARRY4 --------------------------------------------------------------
def carry4(S, DI, CIN, CYINIT):
    # C[0] = CYINIT | CIN ; C[i+1] = (S[i]&C[i]) | (~S[i]&DI[i])
    C = [(CYINIT | CIN) & 1]
    for i in range(4):
        s_i, d_i = (S >> i) & 1, (DI >> i) & 1
        C.append((s_i & C[i]) | ((s_i ^ 1) & d_i))
    O  = sum((((S >> i) & 1) ^ C[i]) << i for i in range(4))
    CO = sum(C[i + 1] << i for i in range(4))
    return CO, O

# --- C. SRLC32E -------------------------------------------------------------
def srl_read(state, a):
    return (state >> (a & 31)) & 1, (state >> 31) & 1

def srl_edge(state, d, ce):
    return (((state << 1) | d) & 0xFFFFFFFF) if ce else state

# --- driver: mirrors dump_vectors.cpp line for line -------------------------
def main():
    out = []
    for S in range(16):
        for DI in range(16):
            for CIN in range(2):
                for CYI in range(2):
                    CO, O = carry4(S, DI, CIN, CYI)
                    out.append(f"C4 {S} {DI} {CIN} {CYI} {CO} {O}")

    A_MIN, A_MAX = -(1 << 26), (1 << 26) - 1
    B_MIN, B_MAX = -(1 << 17), (1 << 17) - 1
    MA, MB, MC, MD = (1 << 27) - 1, (1 << 18) - 1, (1 << 48) - 1, (1 << 27) - 1
    directed = [
        (A_MIN & MA, B_MIN & MB, 0, 0, 0, 2),
        (A_MIN & MA, B_MAX & MB, 0, 0, 0, 2),
        (A_MAX & MA, B_MIN & MB, 0, 0, 1, 2),
        (A_MAX & MA, B_MAX & MB, 0, 0, 1, 2),
        (A_MAX & MA, B_MAX & MB, (-1) & MC, A_MAX & MD, 1, 0),
        (A_MIN & MA, B_MIN & MB, 0, A_MIN & MD, 1, 1),
        (1, 1, 0, 0, 0, 2),
        (0, 0, 0, 0, 0, 2),
        # pre-adder wrap: A = D = 2^26-1 truncates to -2
        (A_MAX & MA, B_MAX & MB, 0, A_MAX & MD, 1, 1),
        (A_MAX & MA, B_MIN & MB, 0, A_MAX & MD, 1, 2),
        (A_MIN & MA, B_MAX & MB, 0, A_MIN & MD, 1, 1),
        (A_MIN & MA, B_MIN & MB, 0, A_MIN & MD, 1, 2),
    ]
    rng = LCG(0x243F6A8885A308D3)
    idx = 0
    for (a, b, c, d, pre, mode) in directed:
        p = 0
        for cyc in range(4):
            p = dsp48e2(a, b, c, d, pre, mode, p)
            out.append(f"DSP {idx} {a} {b} {c} {d} {pre} {mode} {cyc} {p}")
        idx += 1
    for _ in range(400):
        a, b = rng.bits(27), rng.bits(18)
        c, d = rng.bits(48), rng.bits(27)
        pre = rng() & 1
        mode = rng() % 3
        p = 0
        for cyc in range(4):
            p = dsp48e2(a, b, c, d, pre, mode, p)
            out.append(f"DSP {idx} {a} {b} {c} {d} {pre} {mode} {cyc} {p}")
        idx += 1

    for i in range(60):
        state = rng.bits(32)
        for step in range(6):
            d, ce = rng() & 1, rng() & 1
            for a in range(32):
                q, q31 = srl_read(state, a)
                out.append(f"SRL {i} {step} {state} {a} {q} {q31}")
            state = srl_edge(state, d, ce)
            out.append(f"SRLE {i} {step} {d} {ce} {state}")

    print("\n".join(out))

if __name__ == "__main__":
    main()

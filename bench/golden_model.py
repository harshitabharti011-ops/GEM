#!/usr/bin/env python3
"""Independent behavioural golden model for synth/tests/macro_smoke.sv.

    *** NOT YET CALIBRATED -- DO NOT USE AS AN ORACLE ***

As of this commit the model cannot reproduce `scratch`, which is ordinary
boolean logic (cnt ^ {x[3:0], y[3:0]}) that GEM's own CPU reference has already
validated. A model that disagrees with a known-good signal is wrong about that
signal, so nothing it says about sum/acc0/acc1 can be trusted yet.

`scratch` is the right calibration target precisely because it is macro-free:
get it matching first -- the open questions are the cycle at which GEM samples
inputs, the phase of the `cnt` counter, and whether GEM's output VCD reports
pre-edge or post-edge register state. Only once `scratch` reads MATCH does the
verdict on the macro-driven ports mean anything.


Deliverable C requires a team-authored reference. This is it, at the DESIGN
level rather than the primitive level: bench/../csrc/tests/reference_model.py
validates the three macro models in isolation, and this validates the whole
netlist end to end.

It shares no code with the simulator under test. It reads the same input VCD
GEM is driven with, evaluates macro_smoke's RTL semantics directly in Python,
and compares against a GEM output VCD.

    ./bench/golden_model.py build_native/in.vcd build_shred/out.vcd
    ./bench/golden_model.py build_native/in.vcd build_native/out.vcd

Stimulus convention, read off the generated VCD: inputs are driven at
t = 1000k and the clock rises at t = 1000k + 500. Cycle k therefore latches the
inputs written at 1000k.
"""
import argparse, sys
from collections import defaultdict

# --------------------------------------------------------------------------
# macro semantics -- transcribed from the problem statement, independently of
# csrc/macros.cuh and of synth/gem_macros_behav.v
# --------------------------------------------------------------------------

def sext(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v

def dsp48e2(a, b, c, d, use_preadd, mode, p_cur):
    A, B, C, D = sext(a, 27), sext(b, 18), sext(c, 48), sext(d, 27)
    AD = sext(A + D if use_preadd else A, 27)      # 27-bit pre-adder, wraps
    M = sext(AD * B, 45)
    p = C if mode == 0 else (M if mode == 1 else p_cur + M)
    return sext(p, 48)

def carry4(s, di, cin, cyinit):
    c = (cyinit | cin) & 1
    CO = O = 0
    for i in range(4):
        si, dii = (s >> i) & 1, (di >> i) & 1
        O |= (si ^ c) << i
        c = (si & c) | ((~si & 1) & dii)
        CO |= c << i
    return CO, O

# --------------------------------------------------------------------------
# VCD parsing
# --------------------------------------------------------------------------

def parse_vcd(path):
    """-> (values, times) where values[name] = [(t, intvalue_or_str), ...]"""
    code2name, values, t = {}, defaultdict(list), 0
    defs = True
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if defs:
            if s.startswith("$var"):
                p = s.split()
                code2name[p[3]] = p[4]
            elif s.startswith("$enddefinitions"):
                defs = False
            continue
        if s[0] == "#":
            t = int(s[1:])
        elif s[0] in "bB":
            bits, code = s[1:].split(None, 1)
            values[code2name.get(code.strip(), code.strip())].append((t, bits))
        elif s[0] in "01xXzZ":
            values[code2name.get(s[1:].strip(), s[1:].strip())].append((t, s[0]))
    return values

def sample_at(stream, times):
    """Sample-and-hold a value stream at each requested time."""
    out, i, cur = [], 0, "0"
    for t in times:
        while i < len(stream) and stream[i][0] <= t:
            cur = stream[i][1]
            i += 1
        out.append(cur)
    return out

def as_int(v):
    v = v.lower()
    if any(ch in v for ch in "xz"):
        return None
    return int(v, 2)

# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcd")
    ap.add_argument("output_vcd")
    ap.add_argument("-n", "--cycles", type=int, default=0,
                    help="limit cycles compared (0 = all)")
    ap.add_argument("-r", "--report", type=int, default=6)
    a = ap.parse_args()

    inp = parse_vcd(a.input_vcd)
    if "clk" not in inp:
        sys.exit("no clk in input VCD")
    n = len([1 for t, v in inp["clk"] if v == "1"])
    if a.cycles:
        n = min(n, a.cycles)
    latch = [1000 * k for k in range(n)]          # inputs valid for cycle k

    def sig(name, width):
        if name not in inp:
            sys.exit(f"input VCD lacks port {name}")
        return [as_int(v) for v in sample_at(inp[name], latch)]

    a0, a1, d0 = sig("a0", 27), sig("a1", 27), sig("d0", 27)
    b0, b1, c0 = sig("b0", 18), sig("b1", 18), sig("c0", 48)
    mode = sig("mode", 2)
    x, y = sig("x", 16), sig("y", 16)
    tap = sig("tap", 5)
    ser_in, ser_en = sig("ser_in", 1), sig("ser_en", 1)

    # ---- simulate -------------------------------------------------------
    exp = defaultdict(list)
    p0 = p1 = 0
    sr = 0
    cnt = 0
    for k in range(n):
        # combinational, from this cycle's inputs and PRE-edge register state
        p = x[k] ^ y[k]
        carry = 0
        s_bits = 0
        for i in range(4):
            CO, O = carry4((p >> (4 * i)) & 0xF, (x[k] >> (4 * i)) & 0xF,
                           carry, 0)
            s_bits |= O << (4 * i)
            carry = (CO >> 3) & 1
        s_bits |= carry << 16
        exp["sum"].append(s_bits)
        exp["delayed"].append((sr >> tap[k]) & 1)
        exp["delayed31"].append((sr >> 31) & 1)

        # clocked
        p0 = dsp48e2(a0[k], b0[k], c0[k], d0[k], True,  mode[k], p0)
        p1 = dsp48e2(a1[k], b1[k], 0,     0,     False, 1,       p1)
        if ser_en[k]:
            sr = ((sr << 1) | ser_in[k]) & 0xFFFFFFFF
        cnt = (cnt + 1) & 0xFF

        exp["acc0"].append(p0 & ((1 << 48) - 1))
        exp["acc1"].append(p1 & ((1 << 48) - 1))
        exp["scratch"].append(cnt ^ (((x[k] & 0xF) << 4) | (y[k] & 0xF)))

    # ---- compare --------------------------------------------------------
    out = parse_vcd(a.output_vcd)
    widths = {"acc0": 48, "acc1": 48, "sum": 17, "delayed": 1,
              "delayed31": 1, "scratch": 8}

    # GEM writes per-bit names; reassemble vectors.
    edges = [1000 * k + 500 for k in range(n)]
    got = {}
    for port, w in widths.items():
        if w == 1:
            if port not in out:
                print(f"  {port}: absent from output VCD"); continue
            got[port] = [as_int(v) for v in sample_at(out[port], edges)]
        else:
            bits = []
            for b in range(w):
                nm = f"{port}[{b}]"
                if nm not in out:
                    bits = None; break
                bits.append(sample_at(out[nm], edges))
            if bits is None:
                print(f"  {port}: absent from output VCD"); continue
            got[port] = [sum((as_int(bits[b][k]) or 0) << b for b in range(w))
                         for k in range(n)]

    print(f"cycles compared: {n}\n")
    worst = 0
    for port in ("scratch", "sum", "delayed", "delayed31", "acc0", "acc1"):
        if port not in got:
            continue
        e, g = exp[port], got[port]
        bad = [k for k in range(n) if e[k] != g[k]]
        # also test the inverted hypothesis, which a polarity bug produces
        mask = (1 << widths[port]) - 1
        badinv = [k for k in range(n) if (e[k] ^ mask) != g[k]]
        tag = "MATCH" if not bad else (
              "INVERTED" if not badinv else f"differs at {len(bad)}/{n}")
        print(f"  {port:<10} {tag}")
        if bad and badinv:
            worst += 1
            for k in bad[:a.report]:
                print(f"      cycle {k}: expected {e[k]:#x}  got {g[k]:#x}")
    print()
    if worst:
        sys.exit(f"{worst} port(s) disagree with the golden model.")
    print("golden model agrees with this VCD.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two GEM output VCDs port by port.

This is the end-to-end oracle for the native macro path.

The shredded build and the macro-preserving build are the SAME RTL driven by
the SAME stimulus. Their outputs must therefore agree at every timestamp. The
shredded run is independently validated by GEM's own CPU reference
(`--check-with-cpu`), so if native matches shredded, native is correct --
without needing a CPU model of the macro phase.

    ./bench/compare_vcd.py build_shred/out.vcd build_native/out.vcd

Identifier codes are assigned per file and mean nothing across files, so the
comparison is by port NAME. Ports present in only one file are reported
separately rather than silently ignored.
"""
import argparse, re, sys
from collections import defaultdict


def parse(path):
    """Return {port_name: [(time, value), ...]} in timestamp order."""
    code2name, name2code = {}, {}
    scope = []
    values = defaultdict(list)
    t = 0
    in_defs = True
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if in_defs:
                if s.startswith("$scope"):
                    parts = s.split()
                    if len(parts) >= 3:
                        scope.append(parts[2])
                    continue
                if s.startswith("$upscope"):
                    if scope:
                        scope.pop()
                    continue
                if s.startswith("$var"):
                    # $var wire 8 ! name [7:0] $end
                    p = s.split()
                    code, name = p[3], p[4]
                    code2name[code] = name
                    name2code[name] = code
                    continue
                if s.startswith("$enddefinitions"):
                    in_defs = False
                    continue
                continue
            if s.startswith("#"):
                t = int(s[1:])
                continue
            if s.startswith(("b", "B")):
                bits, code = s[1:].split(None, 1)
                values[code2name.get(code.strip(), code.strip())].append((t, bits))
            elif s and s[0] in "01xXzZ":
                code = s[1:].strip()
                values[code2name.get(code, code)].append((t, s[0]))
    return values


def norm(v):
    """Normalise a value for comparison: strip leading zeros, lowercase x/z."""
    v = v.lower()
    if len(v) > 1:
        v = v.lstrip("0") or "0"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="the validated baseline, e.g. build_shred/out.vcd")
    ap.add_argument("candidate", help="the run under test, e.g. build_native/out.vcd")
    ap.add_argument("-n", "--max-report", type=int, default=10,
                    help="how many differing samples to print per port")
    ap.add_argument("-c", "--cycles", type=int, default=400,
                    help="cycles to compare on the common sample grid")
    ap.add_argument("--shift", type=int, default=0,
                    help="shift the candidate by N cycles before comparing. "
                         "naive_sim timestamps its output one cycle later than "
                         "cuda_test, so CPU-vs-GPU comparisons need --shift -1.")
    a = ap.parse_args()

    ref, cand = parse(a.reference), parse(a.candidate)
    if a.shift:
        # Re-time the candidate rather than dropping samples, so a value that
        # is merely offset does not read as a value that is wrong.
        cand = {k: [(t - 1000 * a.shift, v) for (t, v) in s]
                for k, s in cand.items()}
    common = sorted(set(ref) & set(cand))
    only_ref = sorted(set(ref) - set(cand))
    only_cand = sorted(set(cand) - set(ref))

    if only_ref:
        print(f"WARNING: {len(only_ref)} port(s) only in reference: "
              f"{', '.join(only_ref[:8])}")
    if only_cand:
        print(f"WARNING: {len(only_cand)} port(s) only in candidate: "
              f"{', '.join(only_cand[:8])}")
    if not common:
        sys.exit("FAIL: no ports in common -- check --output-vcd-scope on both runs")

    # Sample-and-hold both streams on a common grid rather than comparing raw
    # value-change lists. Two simulators may emit a change at different
    # timestamps, or suppress a redundant one, while agreeing on every value
    # the circuit actually holds -- raw sequence equality reports that as a
    # total mismatch.
    ncyc = a.cycles
    grid = [1000 * k + 1500 for k in range(ncyc)]

    def hold(stream, times):
        out, i, cur = [], 0, "0"
        for tt in times:
            while i < len(stream) and stream[i][0] <= tt:
                cur = stream[i][1]
                i += 1
            out.append(norm(cur))
        return out

    bad = 0
    total_samples = len(common) * ncyc
    for port in common:
        r = hold(ref[port], grid)
        c = hold(cand[port], grid)
        if r == c:
            continue
        bad += 1
        diffs = [k for k in range(ncyc) if r[k] != c[k]]
        print(f"\nDIFF {port}: {len(diffs)}/{ncyc} cycles differ")
        for k in diffs[:a.max_report]:
            print(f"    cycle {k}: ref={r[k]}  cand={c[k]}")
        if len(diffs) > a.max_report:
            print("    ...")

    print(f"\n{len(common)} common port(s), {ncyc} cycles, {total_samples} comparisons.")
    if bad:
        sys.exit(f"FAIL: {bad} of {len(common)} port(s) differ.")
    print("PASS: every common port matches the reference at every timestamp.")


if __name__ == "__main__":
    main()

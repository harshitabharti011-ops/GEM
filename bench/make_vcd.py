#!/usr/bin/env python3
"""Generate an input VCD for a gate-level netlist.

GEM is driven by a static waveform, so any benchmark needs stimulus. This
reads the top module's input ports straight out of the synthesised .gv --
rather than re-deriving them from the RTL -- so the stimulus can never
disagree with what the simulator is actually going to see.

    ./bench/make_vcd.py build/gatelevel.gv macro_smoke -n 2000 -o build/in.vcd

The clock port is detected by name and toggles once per cycle; every other
input is re-randomised on the rising edge, which is the pattern GEM's
non-interactive path expects (PS note 2: one global clock domain).
"""
import argparse, random, re, sys

def parse_ports(path, top):
    """Return [(name, width)] for the top module's inputs, in declared order."""
    src = open(path).read()
    # Yosys writes: module top(a, b, c); input x; input [7:0] y; ...
    m = re.search(r'\bmodule\s+\\?' + re.escape(top) + r'\b(.*?)\bendmodule',
                  src, re.S)
    if not m:
        sys.exit(f"top module '{top}' not found in {path}")
    body = m.group(1)
    ports = []
    for decl in re.finditer(r'^\s*input\s+(?:\[(\d+):(\d+)\]\s*)?\\?([\w$.\[\]]+)\s*;',
                            body, re.M):
        hi, lo, name = decl.groups()
        width = 1 if hi is None else abs(int(hi) - int(lo)) + 1
        ports.append((name, width))
    if not ports:
        sys.exit(f"no input ports found in module '{top}'")
    return ports

def ident(i):
    """Printable VCD identifier codes, starting at '!'."""
    out = ""
    i += 1
    while i:
        i -= 1
        out = chr(33 + i % 94) + out
        i //= 94
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("netlist"); ap.add_argument("top")
    ap.add_argument("-n", "--cycles", type=int, default=1000)
    ap.add_argument("-o", "--out", default="input.vcd")
    ap.add_argument("--clock", default=None,
                    help="clock port name (default: auto-detect clk/clock/ck)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--period", type=int, default=1000, help="clock period, ps")
    a = ap.parse_args()
    random.seed(a.seed)

    ports = parse_ports(a.netlist, a.top)
    clk = a.clock
    if clk is None:
        for n, w in ports:
            if w == 1 and n.lower() in ("clk", "clock", "ck", "clk_i"):
                clk = n; break
    if clk is None:
        sys.exit("could not auto-detect a clock port; pass --clock")

    ids = {n: ident(i) for i, (n, _) in enumerate(ports)}
    half = a.period // 2

    with open(a.out, "w") as f:
        f.write("$timescale 1ps $end\n")
        f.write(f"$scope module {a.top} $end\n")
        for n, w in ports:
            f.write(f"$var wire {w} {ids[n]} {n} $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n")

        def put(n, w, v):
            if w == 1: f.write(f"{v & 1}{ids[n]}\n")
            else:      f.write(f"b{v & ((1 << w) - 1):b} {ids[n]}\n")

        f.write("#0\n")
        for n, w in ports:
            put(n, w, 0 if n == clk else random.getrandbits(w))

        t = 0
        for _ in range(a.cycles):
            t += half
            f.write(f"#{t}\n"); put(clk, 1, 1)          # rising edge
            t += half
            f.write(f"#{t}\n"); put(clk, 1, 0)
            # new stimulus settles while the clock is low, so it is stable
            # before the next rising edge
            for n, w in ports:
                if n != clk:
                    put(n, w, random.getrandbits(w))
        f.write(f"#{t + half}\n")

    tot = sum(w for _, w in ports)
    print(f"{a.out}: {len(ports)} ports ({tot} bits), clock '{clk}', "
          f"{a.cycles} cycles, {t + half} ps")

if __name__ == "__main__":
    main()

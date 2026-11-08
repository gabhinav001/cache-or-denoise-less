#!/usr/bin/env python3
"""How much of every published speedup was instrumentation?

usage: report_stocktime.py <dir>

`full` records a residual for every block on every cycle. `stock` is the same
forward pass with recording off. Every speedup in this repository was quoted
against `full`, so this measures the denominator's own overhead and recomputes
the ratios against the honest one.

Recording stores a detached clone of a block's output minus its input, so it
cannot change an action. That is checked here, not assumed: if `stock` and
`full` disagree on any episode, the arm is wrong and the timing means nothing.
"""
import glob, json, math, os, statistics as st, sys


def load(p):
    d = json.load(open(p))
    out, tim, per = {}, {}, {}
    for r in d["rows"]:
        k = (r["task"], r["episode"])
        out.setdefault(r["config"], {})[k] = bool(r["success"])
        if "ms_per_cycle" in r and not r.get("warmup"):
            tim.setdefault(r["config"], []).append(r["ms_per_cycle"])
            per.setdefault(r["config"], {})[k] = r["ms_per_cycle"]
    return d, out, tim, per


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no stocktime artefacts in {d} -- experiment not yet collected.")
        return 0

    rc, ORDER = 0, {}
    for p in files:
        nm = os.path.basename(p)[:-5]
        j, out, tim, per = load(p)
        steps = (j.get("meta") or {}).get("num_inference_steps")
        print(f"\n  {nm}  ({steps} denoising steps)\n")

        # 1. correctness gate -- stock must be outcome-identical to full
        if "stock" in out and "full" in out:
            ks = sorted(set(out["stock"]) & set(out["full"]))
            dis = [k for k in ks if out["stock"][k] != out["full"][k]]
            if dis:
                print(f"    FAIL: stock and full disagree on {len(dis)} of {len(ks)} episodes.")
                print("    Recording is supposed to be a detached read that cannot change")
                print("    an action. It changed one. The timing below is not usable.")
                rc = 1
            else:
                print(f"    stock == full on all {len(ks)} episodes, as recording being a")
                print("    detached read requires. The arms differ only in instrumentation.")

        # 2. the denominators
        print(f"\n    {'arm':28} {'ms/cycle':>9} {'sd':>7} {'n':>4}")
        for c in sorted(tim, key=lambda c: st.mean(tim[c])):
            v = tim[c]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"    {c[:28]:28} {st.mean(v):9.2f} {sd:7.2f} {len(v):4}")

        if "stock" in tim and "full" in tim:
            ms, mf = st.mean(tim["stock"]), st.mean(tim["full"])
            over = 100.0 * (mf - ms) / ms
            _ks = sorted(set(per["stock"]) & set(per["full"]))
            ORDER[nm] = (mf - ms, over, (j.get("meta") or {}).get("configs"), steps,
                         [per["full"][k] - per["stock"][k] for k in _ks])
            print(f"\n    full minus stock: {mf-ms:+.2f} ms/cycle, {over:+.1f}% of stock.")
            if steps:
                print(f"    That is {steps} x 30 = {int(steps)*30} subtract-and-clone per cycle.")

            # 3. every ratio, both ways
            cache = [c for c in tim if "c3ache" in c]
            if cache:
                print(f"\n    {'cache arm':28} {'vs full':>9} {'vs stock':>9}  {'overstated by':>14}")
                for c in sorted(cache, key=lambda c: st.mean(tim[c])):
                    m = st.mean(tim[c])
                    print(f"    {c[:28]:28} {mf/m:8.3f}x {ms/m:8.3f}x  {100*(mf/m)/(ms/m)-100:13.1f}%")
            print("\n    The right-hand column is what a deployment would see. The")
            print("    left-hand one is what this repository has been reporting.")

    # THE ORDER CONTROL. The first arm in a run absorbs process warmup, so
    # whichever of stock and full runs first is handicapped -- and that is the
    # comparison. Both orders are collected; only agreement establishes anything.
    if len(ORDER) > 1:
        print("\n\n  ORDER CONTROL\n")
        print(f"    {'run':12} {'steps':>5}  {'first arm':10} {'full-stock':>11} {'%':>8}")
        for nm in sorted(ORDER):
            dms, pct, cfgs, steps, _ = ORDER[nm]
            first = (cfgs or ["?"])[0]
            print(f"    {nm:12} {str(steps):>5}  {first:10} {dms:+10.2f} {pct:+8.1f}")
        for k in ("10", "4"):
            pair = {n: v for n, v in ORDER.items() if str(v[3]) == k}
            if len(pair) != 2:
                continue
            print(f"\n    AT {k} STEPS")
            signs = {(v[0] > 0) for v in pair.values()}
            # Averaging the two orders cancels the position term: whichever arm
            # runs first carries the same handicap in both, with opposite sign.
            avg = sum(v[0] for v in pair.values()) / 2.0
            allp = [x for v in pair.values() for x in v[4]]
            sd = st.stdev(allp) if len(allp) > 1 else 0.0
            se = sd / math.sqrt(len(allp)) if allp else 0.0
            t = avg / se if se else 0.0
            print(f"      order-averaged  {avg:+.2f} ms/cycle   se {se:.2f}   t {t:+.2f}   n {len(allp)}")
            if len(signs) > 1:
                print("      THE SIGN FOLLOWS THE ORDER, so arm position -- not recording --")
                print("      dominates this comparison. A single-order run would have")
                print("      reported whichever answer its order produced.")
            if abs(t) < 2.0:
                print("      NOT DISTINGUISHABLE FROM ZERO. The recording overhead is below")
                print("      this measurement's noise floor. Report it as unmeasured, never")
                print("      as a number, and do not quote the ratios above as corrections.")
            elif avg < 0:
                print("      The estimate is NEGATIVE: removing work appears to make it")
                print("      slower. At source level that is surprising, but it is NOT")
                print("      impossible under torch.compile -- removing a consumer changes")
                print("      what fuses, what stays resident and how kernels schedule, and")
                print("      any of those can cost more than the clone saved. Read it as")
                print("      unresolved, not as proof of noise.")
            else:
                print(f"      Recording costs {avg:.1f} ms/cycle and survives the order control.")

    print("\n    ON THE HOST. This measurement is dominated by the machine, not by")
    print("    the design. On a 36-core box at load 63 the stock arm's sd was 217 ms")
    print("    and nothing was resolvable; on a 48-core box at load 4 it is 1.2 ms")
    print("    and the overhead comes out at t = 18. Check loadavg in provenance")
    print("    before trusting any row above.")
    print("\n    ON NEGATIVE ESTIMATES, if one appears. A negative overhead is not")
    print("    impossible, though an earlier version of this file said it was on the")
    print("    grounds that deleting a detached clone cannot slow a forward pass.")
    print("    That is source-level reasoning and does not hold under a compiler:")
    print("    removing a tensor's only consumer changes fusion boundaries, liveness")
    print("    and scheduling. A negative estimate is either host noise or a real")
    print("    compilation effect, and this measurement cannot separate them.")
    return rc



if __name__ == "__main__":
    sys.exit(main())

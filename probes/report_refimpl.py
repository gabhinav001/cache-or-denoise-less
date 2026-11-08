#!/usr/bin/env python3
"""Does the executed horizon explain our disagreement with the published numbers?

usage: report_refimpl.py <dir>

C3ache reports 96.4% on LIBERO Goal for full compute and 96.4% again for steps
[0,7] at tau=8. We measure ~95% full compute -- agreeing -- and ~68% cached. The
reference specifies a predicted horizon of 32 but not how many actions are
executed before replanning. These local runs compare executing 10 versus 32;
neither setting is established as the published protocol.
"""
import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PUB_FULL, PUB_CACHED = 96.4, 96.4          # C3ache Table 1, LIBERO Goal, [0,7] tau=8


def load(p):
    d = json.load(open(p))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], {})[(r["task"], r["episode"])] = bool(r["success"])
    return d, by


def rate(by, c):
    v = by.get(c)
    return 100.0 * sum(v.values()) / len(v) if v else None


def mcnemar(b, w):
    n = b + w
    if n == 0: return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(p)
        if b.startswith(("gate", "_")): continue
        j, by = load(p)
        if "full" not in by: continue
        runs[b[:-5]] = (j, by)
    if not runs:
        print(f"  no refimpl artefacts in {d} -- experiment not yet collected.")
        return 0

    print("  THE EXECUTED HORIZON\n")
    print("    Same suite, seed, checkpoint, denoising steps, arms and episodes.")
    print("    The only difference is how many actions of each predicted chunk")
    print("    are executed before the policy re-plans.\n")
    print(f"    {'run':10} {'replan':>7} {'full':>8} {'plain':>8} {'atten.':>8} "
          f"{'plain-full':>11}   n")
    got = {}
    for nm in sorted(runs):
        j, by = runs[nm]
        rp = (j.get("meta") or {}).get("replan")
        f = rate(by, "full")
        pl = next((c for c in by if c.endswith("tau8")), None)
        sh = next((c for c in by if c.endswith("tau8_shrink")), None)
        rp_, rs = rate(by, pl) if pl else None, rate(by, sh) if sh else None
        n = len(by["full"])
        got[nm] = (rp, f, rp_, rs)
        print(f"    {nm:10} {str(rp):>7} {f:7.1f}% {rp_:7.1f}% {rs:7.1f}% "
              f"{rp_-f:+10.1f}   {n}")

    print(f"\n    published (C3ache, LIBERO Goal)      {PUB_FULL:5.1f}% {PUB_CACHED:7.1f}%")

    a = got.get("replan10"); b = got.get("replan32")
    if not (a and b):
        print("\n    need both replan10 and replan32 to conclude.")
        return 0

    print("\n  WHAT THIS SETTLES\n")
    moved = b[2] - a[2]
    gap10 = PUB_CACHED - a[2]
    gap32 = PUB_CACHED - b[2]
    print(f"    plain tau=8 at replan=10   {a[2]:5.1f}%   gap to published {gap10:+6.1f}")
    print(f"    plain tau=8 at replan=32   {b[2]:5.1f}%   gap to published {gap32:+6.1f}")
    print(f"    the horizon moves it       {moved:+5.1f} points")

    closed = abs(gap32) < 0.25 * abs(gap10) if gap10 else False
    if closed:
        print("\n    Executing 32 actions closes most of the numerical gap in this")
        print("    local control. This does not establish the reference's protocol")
        print("    or exclude other implementation differences.")
    elif moved > 5:
        print("\n    Executing 32 actions improves local cache success but does not")
        print("    close the gap. The reference's executed length remains unspecified.")
    else:
        print("\n    Executing 32 actions does not close the local cache-success gap.")
        print("    This does not exclude other horizon settings or interactions with")
        print("    checkpoint, precision, runtime, or whole-stack cache semantics.")

    print("\n  ATTENUATION AT EACH TESTED EXECUTED HORIZON\n")
    for nm in ("replan10", "replan32"):
        j, by = runs[nm]
        pl = next((c for c in by if c.endswith("tau8")), None)
        sh = next((c for c in by if c.endswith("tau8_shrink")), None)
        ks = sorted(set(by[pl]) & set(by[sh]))
        bb = sum(1 for k in ks if by[sh][k] and not by[pl][k])
        ww = sum(1 for k in ks if by[pl][k] and not by[sh][k])
        print(f"    {nm:10} delta {rate(by,sh)-rate(by,pl):+6.1f}  "
              f"{bb} better / {ww} worse  McNemar p={mcnemar(bb,ww):.3g}")
    print("\n    These contrasts concern the local implementation at two executed")
    print("    lengths. They do not demonstrate attenuation on the reference system.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The full-compute arm across two collections -- and why it does NOT isolate arm order.

usage: report_armshift.py <dirA> <dirB>
       (default: results/frontier results/tau8)

The full arm caches nothing, so it is the same computation in both collections
and ought to give the same number. It does not. An earlier version of this probe
attributed that to the arm list, on the grounds that a different arm list is a
different set of compiled graphs (A8). That attribution was wrong, for two
reasons found on re-examination:

  1. THE HARDWARE CHANGED TOO. The frontier runs are an RTX A6000; the tau=8
     re-collection is an A100-PCIE-40GB. Arm list and GPU moved together, so
     this pair isolates neither.

  2. THE CLEAN CONTROL SAYS ARM ORDER IS NOT IT. results/rngdrift/ holds a
     three-arm and a five-arm run at one seed on ONE machine. There, `full`
     runs first in both and is bit-identical -- 20/20 outcomes and 20/20 step
     counts. Arms that MOVE position diverge (18/20 and 19/20); an arm that
     keeps position does not. `full` is listed first in both collections
     compared here, so arm order predicts no change in it at all.

What is left is the hardware, which the repository already reports as a hazard
worth up to 14.6% cross-machine. This probe therefore reports a COLLECTION
difference and names the confound instead of assigning a cause.
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tstat                                       # noqa: E402


def full_by_seed(d):
    """seed -> (full %, n, arms, how the seed was established, gpu).

    A seed read from meta is a record. A seed read from a filename is a guess
    about what the launcher passed, and the four frontier runs that carry no
    recorded seed are guesses of exactly the kind that made the original nine
    tau=8 runs unusable. The two are kept apart and never mixed into one
    interval.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p).startswith(("gate", "_")):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if not isinstance(j, dict) or "rows" not in j:
            continue
        seed, how = (j.get("meta") or {}).get("seed"), "recorded"
        if seed is None:
            m = re.search(r"seed(\d+)", os.path.basename(p))
            if not m:
                continue
            seed, how = m.group(1), "from filename"
        rows = [r for r in j["rows"] if r["config"] == "full"]
        if not rows:
            continue
        arms = sorted({r["config"] for r in j["rows"]})
        prov = (j.get("meta") or {}).get("provenance") or {}
        gpu = prov.get("gpu") or (j.get("meta") or {}).get("gpu") or "unrecorded"
        out[int(seed)] = (100.0 * sum(bool(r["success"]) for r in rows) / len(rows),
                          len(rows), arms, how, gpu)
    return out


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "results/frontier"
    b = sys.argv[2] if len(sys.argv) > 2 else "results/tau8"
    A, B = full_by_seed(a), full_by_seed(b)
    shared = sorted(set(A) & set(B))
    if not shared:
        print(f"  no seed is present in both {a} and {b}.")
        return 0

    print(f"  THE FULL-COMPUTE ARM, {os.path.basename(a)} vs {os.path.basename(b)}")
    print("\n    Same suite, same episode count, same checkpoint. The full arm caches")
    print("    nothing, so it is the same computation in both. What differs is the")
    print("    arm list AND the GPU -- which is why this pair attributes nothing.\n")
    print(f"    {'seed':>6}  {'A':>7}  {'B':>7}  {'B-A':>6}    n  seed known by")
    firm, loose = [], []
    gpus = set()
    for s_ in shared:
        fa, na, _, ha, ga = A[s_]
        fb, nb, _, hb, gb = B[s_]
        gpus.add((ga, gb))
        how = "both recorded" if ha == hb == "recorded" else f"A {ha}, B {hb}"
        if na != nb:
            print(f"    {s_:>6}  {fa:6.1f}%  {fb:6.1f}%   (skipped: {na} vs {nb} episodes)")
            continue
        (firm if ha == hb == "recorded" else loose).append(fb - fa)
        print(f"    {s_:>6}  {fa:6.1f}%  {fb:6.1f}%  {fb-fa:+6.1f}  {na:>4}  {how}")

    def block(ds, label):
        if not ds:
            return
        if len(ds) < 2:
            print(f"\n    {label}: {ds[0]:+.1f} points on one seed -- no interval from one pair.")
            return
        m, lo, hi, sd, t = tstat.mean_ci(ds)
        print(f"\n    {label}:")
        print(f"      mean {m:+.2f}  sd {sd:.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (t={t:.3f})")

    block(firm, f"seeds recorded on both sides (n={len(firm)})")
    block(firm + loose, f"including seeds taken from filenames (n={len(firm) + len(loose)})")

    print("\n    HARDWARE")
    for ga, gb in sorted(gpus):
        print(f"      A: {ga:28} B: {gb}")
    if any(ga != gb for ga, gb in gpus):
        print("\n    The GPU is not the same. Arm list and hardware moved together, so")
        print("    this difference cannot be assigned to either one.")

    if loose:
        print(f"\n    {len(loose)} of these pairings also rest on a filename rather than a")
        print( "    recorded seed, so they may differ by seed as well.")

    print("\n    WHY ARM ORDER IS NOT THE EXPLANATION")
    print( "      full is listed FIRST in both collections. The same-machine control")
    print( "      in results/rngdrift/ (a 3-arm and a 5-arm run at one seed on one")
    print( "      A6000) shows an arm that keeps its position is bit-identical --")
    print( "      full agrees 20/20 on outcomes AND 20/20 on step counts -- while")
    print( "      arms that move position diverge. Arm order predicts NO change in")
    print( "      full. Run probes/check_arm_position.py on that pair to see it.")
    print("\n      What remains is the hardware, which this repository already reports")
    print( "      as worth up to 14.6% cross-machine. That is a collection-level")
    print( "      caveat, not a measurement of arm order.")
    print("\n    Either way: absolute rates compare only WITHIN a collection. Each")
    print( "    collection's paired delta is unaffected -- both arms sit in one run.")

    print("\n    arm lists")
    for nm, D in ((os.path.basename(a), A), (os.path.basename(b), B)):
        print(f"      {nm:<12} {', '.join(D[shared[0]][2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

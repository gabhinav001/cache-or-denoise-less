"""Report substitute/skip agreement and latency from existing rollout artifacts.

The CPU stub check proves only the tested arithmetic. Recorded GPU rollouts
can diverge; these sequential-arm records do not isolate why. Failure to detect
a success difference is not evidence of equivalence between inference paths.
Latencies cover synchronized chunk prediction, with warmup episodes excluded.

usage: report_latency.py <dir>
"""
from __future__ import annotations

import json, math, os, sys


def mcnemar_p(n01, n10):
    """Exact two-sided McNemar from discordant counts."""
    n = n01 + n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d, name):
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def rows_by(d):
    out = {}
    for r in d["rows"]:
        out.setdefault(r["config"], {})[(r["task"], r["episode"])] = r
    return out


def rate(rows):
    v = [r["success"] for r in rows.values()]
    return 100.0 * sum(v) / len(v) if v else float("nan")


def timing(rows, warmup_only=False):
    """Mean ms/cycle over non-warmup episodes."""
    v = [r["ms_per_cycle"] for r in rows.values()
         if "ms_per_cycle" in r and bool(r.get("warmup")) == warmup_only]
    if not v: return None, None, 0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd, len(v)


def equivalence(d):
    """Compare recorded substitute/skip rollouts.

    CPU stub equality does not establish GPU compiled-path equivalence. These
    sequential-arm records do not isolate compiler, numerical, or arm-order
    effects. A nonsignificant McNemar test is not an equivalence test.
    The return value indicates only whether any pointwise test rejects equal
    success probabilities; True must not be read as validated equivalence.
    """
    print("\n  EQUIVALENCE -- skipping the block against substituting for it\n")
    by = rows_by(d)
    pairs = [(c, c + "_skip") for c in by if c + "_skip" in by]
    if not pairs:
        print("    no substitute/skip pair present"); return None
    verdict = True
    print("    %-24s %-24s %-11s %-11s %s"
          % ("substituted", "skipped", "outcomes", "steps", "McNemar"))
    for a, b in sorted(pairs):
        ks = sorted(set(by[a]) & set(by[b]))
        so = sum(1 for k in ks if by[a][k]["success"] == by[b][k]["success"])
        ss = sum(1 for k in ks if by[a][k]["steps"] == by[b][k]["steps"])
        n01 = sum(1 for k in ks if not by[a][k]["success"] and by[b][k]["success"])
        n10 = sum(1 for k in ks if by[a][k]["success"] and not by[b][k]["success"])
        p = mcnemar_p(n01, n10)
        if p < 0.05: verdict = False
        print("    %-24s %-24s %3d/%-7d %3d/%-7d %d/%d  p=%.3g%s"
              % (a[:24], b[:24], so, len(ks), ss, len(ks), n01, n10, p,
                 "   <-- SYSTEMATIC" if p < 0.05 else ""))
    print()
    print("    Step-count differences show trajectory divergence. These records")
    print("    do not isolate its cause: compiled arithmetic and arm order both")
    print("    remain candidates. A paired test concerns success probabilities,")
    print("    not the numerical equivalence of the two inference paths.")
    nd = sum(1 for a2, b2 in sorted(pairs)
             for k in (set(by[a2]) & set(by[b2]))
             if by[a2][k]["success"] != by[b2][k]["success"])
    if nd < 10:
        print(f"\n    Only {nd} discordant episodes across all pairs. A test on that many")
        print("    has almost no power, so this is an absence of evidence for a")
        print("    systematic difference and NOT evidence that none exists -- a failure")
        print("    to reject, the same way the oracle ceiling is treated.")
    if verdict:
        print("\n    No difference detected by the pointwise success tests.")
        print("    Equivalence remains UNRESOLVED: no equivalence margin was tested,")
        print("    and these small samples cannot validate transferring quality")
        print("    results from substitution to an explicitly skipping deployment.")
    else:
        print("\n    A pointwise success test detects a difference in this collection.")
        print("    Its cause remains unresolved; compare outputs on identical inputs")
        print("    and counterbalance arm order before attributing it to cache semantics.")
    return verdict


def dce_check(d):
    """Does the substituting arm actually compute every block?

    Every quality result in this repository was collected with a harness that
    calls the block and then substitutes the cached residual for its output. The
    stated premise was that this keeps the arms at identical compute, differing
    in one scalar multiply. Compilation could delete the unused computation,
    which would make that premise false.

    It is testable directly: if every block really ran, the substituting arm
    would cost the same as full compute. Anything faster is work the compiler
    removed.
    """
    by = rows_by(d)
    if "full" not in by: return
    f, _sd, _n = timing(by["full"])
    subs = [c for c in by if c.startswith("c3ache") and not c.endswith("_skip")]
    if f is None or not subs: return
    print("\n\n  DOES THE SUBSTITUTING ARM REALLY COMPUTE EVERY BLOCK?\n")
    print("    %-30s %10s %10s %s" % ("arm", "ms/cycle", "vs full", "implied reuse cost"))
    print("    %-30s %10.1f %9.3fx %s" % ("full", f, 1.0, "-"))
    flagged = False
    for c in sorted(subs):
        m, sd, n = timing(by[c])
        if m is None: continue
        tau = None
        for part in c.split("_"):
            if part.startswith("tau"):
                try: tau = int(part[3:])
                except ValueError: pass
        # at interval tau, one cycle in tau is a full refresh, so
        #   mean = (1/tau)*full + (1 - 1/tau)*reuse
        reuse = ""
        if tau and tau > 1:
            fr = 1.0 / tau
            reuse = f"{((m / f) - fr) / (1 - fr):.3f} x a full cycle"
        if m < f * 0.97: flagged = True
        print("    %-30s %10.1f %9.3fx %s" % (c, m, f / m, reuse))
    print()
    if flagged:
        print("    Substitution is faster in these records. Removing unused operations")
        print("    during compilation is a candidate explanation, but timings alone do")
        print("    not identify which operations were removed. The source-level fn()")
        print("    call does not guarantee equal executed work across arms; inspect")
        print("    compiled graphs or profiler traces to establish that.")
    else:
        print("    These timings do not show a material reduction. Similar elapsed")
        print("    times alone do not establish identical executed computation.")


def curve(d):
    print("\n\n  LATENCY AND SUCCESS, FROM THE SAME EPISODES\n")
    by = rows_by(d)
    if "full" not in by:
        print("    no full-compute arm to normalise against"); return
    base, bsd, bn = timing(by["full"])
    if base is None:
        print("    no timing recorded -- was +PROBE.timing=1 set?"); return
    wm, _, wn = timing(by["full"], warmup_only=True)
    # Fast-WAM's schedule: 240 of 300 cells are cacheable, one cycle in tau is a
    # full refresh, so removed block evaluations = (1 - 1/tau) * 240/300.
    BLOCKS = {2: 40.0, 4: 60.0, 8: 70.0, 0: 80.0}
    print("    %-26s %9s %8s %9s %9s %8s"
          % ("arm", "ms/cycle", "speedup", "blocks-", "time-", "success"))
    rows = []
    for c in sorted(by, key=lambda c: (c != "full", c)):
        m, sd, n = timing(by[c])
        if m is None: continue
        sp = base / m if m else float("nan")
        tau = None
        for part in c.split("_"):
            if part.startswith("tau"):
                try: tau = int(part[3:])
                except ValueError: pass
        blk = BLOCKS.get(tau)
        tsave = 100.0 * (1 - m / base)
        rows.append((c, blk, tsave))
        print("    %-26s %9.1f %8.3fx %8s %8.1f%% %7.1f%%"
              % (c[:26], m, sp,
                 f"{blk:.0f}%" if blk is not None else "-", tsave, rate(by[c])))

    real = [(c, b, t) for c, b, t in rows if b]
    if real:
        print("\n    BLOCK EVALUATIONS ARE NOT WALL CLOCK. The saving this study has")
        print("    reported throughout is the first column: how many block evaluations")
        print("    the schedule removes. What a deployment gets is the second.")
        print()
        print("      %-10s %-14s %-14s %s" % ("arm", "blocks removed", "time saved", "ratio"))
        for c, b, t in real:
            print("      %-10s %13.0f%% %13.1f%% %6.2f"
                  % (c.replace("c3ache_", "").replace("_shrink_skip", ""), b, t, t / b))
        mr = sum(t / b for _, b, t in real) / len(real)
        print(f"\n    About {mr:.0%} of each removed block evaluation shows up as saved time.")
        print("    The rest of a cycle is work no cache touches: the uncached cells, the")
        print("    encode, and whatever is not a cached block. Quoting the block count as")
        print("    though it were a speedup overstates the deployable gain by roughly 2x,")
        print("    and this repository has been quoting the block count.")
    print(f"\n    Baseline is the compiled full-compute arm at {base:.2f} ms/cycle "
          f"(n={bn} episodes).")
    if wm:
        print(f"    First episode of each arm averaged {wm:.2f} ms/cycle and is excluded:"
              f"\n    that is torch.compile, a one-off per arm, not what a deployment pays.")
    print("\n    Read the speedup column against the modelled ceiling from step_sweep.py")
    print("    (1.50-1.58x for any block cache on a compiled baseline). A measured")
    print("    figure above that ceiling means the timing is wrong, not that the cache")
    print("    beat its own bound.")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    e, c = load(d, "equiv"), load(d, "curve")
    if e is None and c is None:
        print(f"  no equiv.json or curve.json in {d}"); return 1
    ok = equivalence(e) if e else None
    if e: dce_check(e)
    if c: curve(c)
    if ok is False:
        print("\n  At least one pointwise substitute/skip success comparison rejects")
        print("  equal success probabilities. Investigate the difference before")
        print("  transferring quality results between these paths.\n")
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Is attenuated caching on the latency-success frontier, or behind it?

Every other comparison in this repository holds compute fixed and asks whether
attenuation helps. That is the right question for auditing the substitution rule
and the wrong one for deciding whether to deploy it. This puts the cache arms and
the obvious alternatives on one plane -- measured milliseconds against measured
success -- and reports which arms are dominated.

An arm is DOMINATED if some other arm is both faster and at least as successful.
A dominated arm has no operating point at which a practitioner should choose it,
whatever its within-family improvement looks like.

TWO COLUMNS, COLLECTED APART, FOR A REASON. Success is seed-dependent and
load-independent; latency is the exact opposite. During the seed-7 collection
this study's host ran at load 63 on 36 CPUs and later at 1.8, and one arm read
5722 ms/cycle on one task against 360 on another -- while its success rate was
unaffected, as it must be, outcomes being deterministic given seed, GPU and arm
list. So:

  success   from each baselines run, per seed, never pooled across seeds
  latency   from results/frontier_timing/, ONE quiet collection, shared by every
            seed as a standardised benchmark. Only WEAKLY seed-dependent: a cache
            arm mixes refresh and reuse cycles in a proportion that moves with
            episode length, so this is an approximation, not an identity

SEEDS ARE NEVER MIXED. An earlier version globbed the whole directory and, once
a second seed landed beside the first, ranked seed-7 arms against seed-42 arms
and printed "dominated by" verdicts across them. The frontier is computed within
one seed or not at all.

usage: report_baselines.py <baselines_dir> [more_dirs ...] [--timing <dir>]
"""
from __future__ import annotations

import glob, json, math, os, re, sys


def _nper(p1, p2, za=1.959963985, zb=0.841621234):
    """Episodes per arm for 80% power on a two-proportion test."""
    pb, d = (p1 + p2) / 2.0, abs(p1 - p2)
    if d == 0: return float("inf")
    return ((za * math.sqrt(2 * pb * (1 - pb))
             + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / d ** 2


def wilson(k, n, z=1.96):
    """Wilson interval for a success rate, in percent.

    Thirty episodes puts the standard error on a rate near 90% at about five
    points, so a dominance verdict read off point estimates alone would be
    reporting sampling noise as a deployment recommendation. The normal
    approximation is poor at rates this close to 1; Wilson is not.
    """
    if n == 0: return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def two_prop_p(k1, n1, k2, n2):
    """Two-sided two-proportion test. Normal approximation, adequate at n>=100."""
    if n1 == 0 or n2 == 0: return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0: return 1.0
    z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2))


def arms(path):
    """(label, ms/cycle, success%, n, lo, hi, successes) for each arm."""
    d = json.load(open(path))
    steps = (d.get("meta") or {}).get("num_inference_steps")
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], []).append(r)
    out = []
    for c, rs in by.items():
        timed = [r["ms_per_cycle"] for r in rs
                 if "ms_per_cycle" in r and not r.get("warmup")]
        if not timed: continue
        ms = sum(timed) / len(timed)
        k = sum(bool(r["success"]) for r in rs)
        sr = 100.0 * k / len(rs)
        lo, hi = wilson(k, len(rs))
        tag = os.path.basename(path)[:-5]
        label = c if c != "full" else (f"full ({steps} steps)" if steps else "full")
        out.append((f"{tag}:{label}", ms, sr, len(rs), lo, hi, k))
    return out


def seed_of(path, meta):
    """Seed for a run: recorded if present, else read off the filename."""
    s = (meta or {}).get("seed")
    if s is not None: return int(s)
    m = re.search(r"_seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 42          # the harness default


def collect(dirs):
    got = []
    for d in dirs:
        if not d or not os.path.isdir(d): continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(p).startswith(("_", "gate")): continue
            try:
                meta = (json.load(open(p)).get("meta") or {})
                for r in arms(p):
                    got.append(tuple(r) + (seed_of(p, meta),))
            except Exception:
                continue
    return got


def timing_table(d):
    """arm-key -> ms/cycle from the quiet timing collection."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p).startswith(("_", "gate")): continue
        try: j = json.load(open(p))
        except Exception: continue
        steps = (j.get("meta") or {}).get("num_inference_steps")
        by = {}
        for r in j["rows"]:
            if "ms_per_cycle" in r and not r.get("warmup"):
                by.setdefault(r["config"], []).append(r["ms_per_cycle"])
        for c, v in by.items():
            out[(c, steps)] = sum(v) / len(v)
    return out


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__); return 2
    tdir = None
    if "--timing" in argv:
        i = argv.index("--timing"); tdir = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    rows = collect(argv)
    if not rows:
        print(f"  no timed artefacts found in {', '.join(argv)}")
        print("  (did the runs set +PROBE.timing=1?)")
        return 1

    T = timing_table(tdir) if tdir and os.path.isdir(tdir) else {}
    if T:
        print(f"\n  LATENCY from {tdir} -- one quiet collection, shared across seeds.")
        print("  SUCCESS from each run, per seed. The two are not collected together")
        print("  because they fail differently: success is seed-dependent and load-")
        print("  independent, latency the exact opposite.")

    seeds = sorted({r[-1] for r in rows})
    rc = 0
    for sd in seeds:
        print(f"\n\n{'='*68}\n  SEED {sd}\n{'='*68}")
        rc |= one_seed([r for r in rows if r[-1] == sd], T)
    if len(seeds) > 1:
        print(f"\n\n  SEEDS ARE REPORTED SEPARATELY and never pooled: arms here move up to")
        print( "  5 points between seeds, which is the size of the gaps this table ranks.")
    return rc


def one_seed(rows, T):
    # dedupe: the same arm can appear in two directories
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r[1]):
        if r[0] in seen: continue
        seen.add(r[0])
        # Prefer the quiet timing collection where it covers this arm. The label
        # is "<run>:<config>" and carries the step count for full-compute arms as
        # "full (N steps)", which is how the timing table is keyed.
        arm = r[0].split(":", 1)[-1]
        m = re.search(r"\((\d+) steps\)", arm)
        stp = int(m.group(1)) if m else 10
        base = re.sub(r"\s*\(\d+ steps\)", "", arm)
        ms = T.get((base, stp), r[1])
        uniq.append((r[0], ms) + tuple(r[2:-1]))       # drop the trailing seed
    uniq.sort(key=lambda r: r[1])

    print("\n  LATENCY AGAINST SUCCESS, MEASURED ON THE SAME EPISODES\n")
    print("    %-34s %9s %8s %-16s %s"
          % ("arm", "ms/cycle", "success", "95% CI", "status"))
    # DOMINANCE IS A POINT-ESTIMATE RELATION. An earlier version of this
    # counted "strictly faster AND not significantly worse" as dominance, which
    # turns a failure to detect a difference into a claim of equal quality. It
    # made this table report that exactly one arm survived, when two sit on the
    # ordinary frontier: four steps is the fastest, six steps has the highest
    # observed success, and neither dominates the other.
    dominated = []
    for lab, ms, sr, n, lo, hi, k in uniq:
        by = [o for o in uniq if o[0] != lab and o[1] < ms and o[2] >= sr]
        st = ""
        if by:
            b = min(by, key=lambda o: o[1])
            p = two_prop_p(b[6], b[3], k, n)
            st = ("dominated by %s (faster and %s; %s)"
                  % (b[0].split(":")[-1][:22],
                     "higher" if b[2] > sr else "equal",
                     f"significantly better, p={p:.3g}" if (b[2] > sr and p < 0.05)
                     else f"difference not significant, p={p:.2g}"))
            dominated.append(lab)
        print("    %-34s %9.2f %7.1f%% [%5.1f,%5.1f]  %s"
              % (lab[:34], ms, sr, lo, hi, st))

    # The frontier is computed WITHIN one episode count. Ranking a 30-episode
    # arm against a 200-episode one puts an arm that happened to score 100/100
    # on 30 episodes (CI [88.6,100.0]) on the frontier, which says nothing about
    # the arm and everything about the sample. Arms at other counts are listed
    # under it, unranked.
    nmax = max(r[3] for r in uniq)
    main_ = [r for r in uniq if r[3] == nmax]
    dom_m = set()
    for r in main_:
        if any(o[0] != r[0] and o[1] < r[1] and o[2] >= r[2] for o in main_):
            dom_m.add(r[0])
    front = [r for r in main_ if r[0] not in dom_m]
    print(f"\n  ON THE FRONTIER, among the {len(main_)} arms collected at {nmax} episodes")
    print( "  (no arm is both faster and at least as successful)\n")
    for lab, ms, sr, n, lo, hi, k in sorted(front, key=lambda r: r[1]):
        print("    %-34s %9.2f %7.1f%%  [%5.1f,%5.1f]" % (lab[:34], ms, sr, lo, hi))
    if len(front) > 1:
        print("\n    More than one arm is on it, and that is the honest reading: the")
        print("    fastest arm and the most successful one are different arms, and")
        print("    being unable to show a difference is not showing equality.")
    ncache_m = sum(1 for r in main_ if "c3ache" in r[0])
    con = [r for r in front if "c3ache" in r[0]]
    print(f"\n    Cache arms on it: {len(con)} of {ncache_m}.")
    small = sorted((r for r in uniq if r[3] != nmax), key=lambda r: r[1])
    if small:
        print(f"\n    NOT ranked -- {len(small)} arms at fewer episodes, shown for reference"
              f" only:\n")
        for lab, ms, sr, n, lo, hi, k in small:
            print("    %-34s %9.2f %7.1f%%  [%5.1f,%5.1f]  n=%d"
                  % (lab[:34], ms, sr, lo, hi, n))

    # The claim that does survive, stated as a paired test rather than as
    # dominance: the fastest arm against the best cache arm, same episodes.
    fastest = min(main_, key=lambda r: r[1])
    cands = [r for r in main_ if "c3ache" in r[0]]
    if cands:
        best = max(cands, key=lambda r: r[2])
        quick = min(cands, key=lambda r: r[1])
        print(f"\n    WHAT SURVIVES. {fastest[0].split(':')[-1][:22]} at {fastest[1]:.1f} ms is faster than")
        print( "    every cache arm here. Against the cache arm that comes closest on")
        print(f"    latency ({quick[0].split(':')[-1][:24]}, {quick[1]:.1f} ms, {quick[2]:.1f}%) it is")
        print(f"    {fastest[2]-quick[2]:+.1f} points, p = {two_prop_p(fastest[6], fastest[3], quick[6], quick[3]):.3g}.")
        print(f"    Against the BEST cache arm ({best[0].split(':')[-1][:24]}, {best[1]:.1f} ms, {best[2]:.1f}%)")
        print(f"    it is {fastest[2]-best[2]:+.1f} points, p = {two_prop_p(fastest[6], fastest[3], best[6], best[3]):.3g}"
              f" -- faster, not better.")
        print( "\n    So the defensible claim is about LATENCY, not quality: no cache arm")
        print( "    buys speed this policy cannot get more cheaply by denoising less.")
        print( "    'No quality loss' is a separate claim needing a pre-specified")
        print( "    margin and a noninferiority test, and is NOT made here.")

    ns = sorted({r[3] for r in uniq})
    span = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    print(f"\n    Dominance above is a POINT-ESTIMATE relation: strictly faster AND at")
    print( "    least as successful on the observed rates. It is deliberately not")
    print( "    'faster and not significantly worse', which would let a failure to")
    print(f"    detect a difference stand in for equal quality. Arms span {span}")
    print( "    episodes, so a wide interval may mean a small arm, not a noisy one.")
    # What it would take to resolve a gap, computed rather than asserted: an
    # earlier version of this line quoted one sample size for every gap size,
    # and the results page then reused it for a gap three times smaller.
    print("\n    To resolve a gap at 80% power, two-sided 0.05, you would need")
    print("    roughly this many episodes PER ARM:\n")
    print(f"      {'gap':>6}  {'from':>14}  {'n per arm':>10}")
    for hi_, lo_ in ((0.965, 0.935), (0.965, 0.915), (0.965, 0.865), (0.965, 0.830)):
        print(f"      {100*(hi_-lo_):5.1f}  {100*hi_:6.1f} vs {100*lo_:5.1f}  {_nper(hi_, lo_):10.0f}")
    print("\n    Nothing finer than the table's own n should be read off it.")

    cache_front = [r for r in front if "c3ache" in r[0]]
    cache_all = [r for r in uniq if "c3ache" in r[0]]
    print("\n  READING\n")
    if not cache_all:
        print("    No cache arms present -- pass the latency experiment's output")
        print("    directory as a second argument to compare against them.")
    elif cache_front:
        print(f"    {len(cache_front)} of {len(cache_all)} cache arms survive on the frontier.")
        print("    At those operating points nothing simpler is both faster and at least")
        print("    as successful, which is the claim a deployment recommendation needs.")
    else:
        print(f"    NONE of the {len(cache_all)} cache arms is on the frontier: every one is")
        print("    both slower and no more successful than simply running fewer denoising")
        print("    steps. The within-tau improvement this repository measures is real, and")
        print("    at the same time there is no latency budget at which a practitioner")
        print("    should choose this cache. Both statements are results; the second is")
        print("    the one that decides deployment.")
    # Match on tau. Comparing a global weight at tau=4 against the best per-cell
    # arm at ANY tau is not a comparison of weighting schemes, it is a comparison
    # of refresh intervals wearing one.
    def _tau(lab):
        for part in lab.split("_"):
            if part.startswith("tau"):
                try: return int(part[3:])
                except ValueError: pass
        return None
    gb = [r for r in uniq if "globalbeta" in r[0] and "shrink" in r[0]]
    pc = [r for r in uniq if "c3ache" in r[0] and "shrink" in r[0]
          and "globalbeta" not in r[0]]
    pairs = [(g, p) for g in gb for p in pc if _tau(g[0]) == _tau(p[0]) and _tau(g[0])]
    if pairs:
        print("\n    ONE GLOBAL WEIGHT AGAINST THE PER-CELL FIT, at matched tau\n")
        for g, p in pairs:
            pv = two_prop_p(g[6], g[3], p[6], p[3])
            print("      tau=%-3s global %5.1f%% [%4.1f,%4.1f]   per-cell %5.1f%% [%4.1f,%4.1f]"
                  "   diff %+.1f, p=%.3g"
                  % (_tau(g[0]), g[2], g[4], g[5], p[2], p[4], p[5], p[2] - g[2], pv))
        g, p = pairs[0]
        pv = two_prop_p(g[6], g[3], p[6], p[3])
        print()
        if pv < 0.05:
            print("    The per-cell fit is worth a measurable amount over one global weight,")
            print("    so the calibration pass is doing real work rather than being ceremony.")
        else:
            print("    The two are not separable at this episode count. That does not show")
            print("    they are equal -- it shows this experiment cannot tell them apart,")
            print("    and the per-cell fit's extra machinery is unjustified by it.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

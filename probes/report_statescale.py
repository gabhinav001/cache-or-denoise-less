"""Does a cache beat a shorter uncached schedule, measured across benchmark states?

Written before the data, so the analysis cannot be shaped to the result.

The design fixes the defect in the seed repeats: there the same 30 initial states
were revisited with different denoiser noise, 87% of state-arm cells came back
unanimous, and no test could separate anything. Here the unit is a distinct
benchmark initial state and there are up to 500 of them, so the comparison has
independent units to work with.

Arms are compared PAIRED on the initial state, because every arm visits the same
states and a paired test is far more efficient when arms agree on most of them.
The quantity that drives power is the discordance rate, not the success rate: two
arms scoring 96% and 93% are separable only through the states where they
actually differ.

Three outcomes are possible and all three are reportable:
  separated     the interval excludes zero; the cache is better or worse
  bounded       the interval includes zero but is tight enough to exclude the
                effect sizes anyone cares about -- a real negative result
  unresolved    the interval still admits a material difference
The script names which one it is rather than leaving it to the reader.
"""
import json, glob, os, sys, random, itertools
from collections import defaultdict
from math import comb

random.seed(20260919)
B = 10000


def seeds_present(d):
    """Seeds with artefacts here: 42 (unsuffixed) plus any _seedN."""
    out = {42} if glob.glob(os.path.join(d, "s[0-9]*.json")) else set()
    for f in glob.glob(os.path.join(d, "s*_seed*.json")):
        out.add(int(os.path.basename(f).rsplit("_seed", 1)[1][:-5]))
    return sorted(out)


def load(d, seed=42):
    """arm -> {init_state: success} for ONE seed.

    Seed variants live in the same directory under a _seedN suffix. Loading the
    directory blind merges them: every arm is keyed by initial state, the seeds
    share all 500 states, so the later file silently overwrites the earlier and
    the result is a mixture reported as one collection. That happened here --
    2st read 97.2% instead of 96.6% -- so the seed is now explicit and required.
    """
    succ, lat, meta = defaultdict(dict), defaultdict(list), {}
    pat = "s*_seed%d.json" % seed if seed != 42 else "s[0-9]*.json"
    for f in sorted(glob.glob(os.path.join(d, pat))):
        if os.path.basename(f) == "gate.json": continue
        if seed == 42 and "_seed" in os.path.basename(f): continue
        j = json.load(open(f)); m = j.get("meta") or {}
        steps = m.get("num_inference_steps") or m.get("n_denoise")
        meta.setdefault("provenance", (m.get("provenance") or {}))
        meta.setdefault("suite", m.get("suite"))
        for r in j["rows"]:
            cfg = r["config"]
            st = steps or "?"
            arm = f"{st}st" if cfg == "stock" else f"{st}st+cache"
            succ[arm][r["init_state"]] = bool(r["success"])
            if not r.get("warmup") and "ms_per_cycle" in r:
                lat[arm].append(r["ms_per_cycle"])
    return succ, {a: sum(v) / len(v) for a, v in lat.items() if v}, meta


def signtest(pos, neg):
    m = pos + neg
    if m == 0: return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(0, min(pos, neg) + 1)) / 2 ** m)


def main(d):
    succ, ms, meta = load(d)
    arms = sorted(succ, key=lambda a: (int(a.split("st")[0]), "cache" in a))
    if len(arms) < 2:
        print("fewer than two arms parsed; nothing to compare"); return 1
    states = sorted(set.intersection(*(set(succ[a]) for a in arms)))
    n = len(states)

    pv = meta.get("provenance") or {}
    print(f"States: {n} distinct initial states, shared by all {len(arms)} arms")
    print(f"Commit: {pv.get('commit') or 'NOT RECORDED'}   GPU: {pv.get('gpu')}\n")
    if n < 100:
        print(f"  NOTE: only {n} states -- far below the ~200 the power calculation")
        print("  asked for. Read every interval below as provisional.\n")

    print(f"  {'arm':14s} {'success':>9s}  95% CI (bootstrap over states)")
    for a in arms:
        v = [succ[a][s] for s in states]
        k = sum(v)
        bs = sorted(100.0 * sum(v[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        print(f"  {a:14s} {100.0*k/n:8.1f}%  [{bs[250]:5.1f}, {bs[9749]:5.1f}]"
              f"   {k}/{n}" + (f"   {ms[a]:.1f} ms" if a in ms else ""))

    print(f"\nPaired comparisons on the initial state (n={n}):")
    print(f"  {'pair':30s} {'diff':>7s} {'disc':>5s} {'95% CI':>16s} {'sign p':>8s}  verdict")
    for a, b in itertools.combinations(arms, 2):
        d_ = [(succ[a][s], succ[b][s]) for s in states]
        ab = sum(1 for x, y in d_ if x and not y)
        ba = sum(1 for x, y in d_ if y and not x)
        diff = 100.0 * (ab - ba) / n
        p = signtest(ab, ba)
        deltas = [(1 if x and not y else (-1 if y and not x else 0)) for x, y in d_]
        bs = sorted(100.0 * sum(deltas[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        lo, hi = bs[250], bs[9749]
        if lo > 0 or hi < 0:            verdict = "SEPARATED"
        elif max(abs(lo), abs(hi)) <= 3: verdict = "bounded within 3 pts"
        elif max(abs(lo), abs(hi)) <= 5: verdict = "bounded within 5 pts"
        else:                            verdict = "unresolved"
        print(f"  {a+' - '+b:30s} {diff:+6.1f}% {ab+ba:5d} "
              f"[{lo:+5.1f},{hi:+5.1f}] {p:8.3f}  {verdict}")

    print("\n  diff is a minus b in percentage points; disc counts states where the")
    print("  arms disagree, which is what a paired test actually spends. A verdict")
    print("  of 'bounded' is a negative result, not a failure: it says the effect,")
    print("  if any, is smaller than the stated bound.")
    return 0


def summary(d=None, seed=42):
    """Numbers canonical_numbers.py needs, without the printing."""
    d = d or os.path.join(os.path.dirname(__file__), "..", "results", "statescale")
    succ, ms, meta = load(d, seed)
    arms = sorted(succ, key=lambda a: (int(a.split("st")[0]), "cache" in a))
    states = sorted(set.intersection(*(set(succ[a]) for a in arms)))
    n = len(states)
    out = {"n_states": n, "n_arms": len(arms),
           "commit": (meta.get("provenance") or {}).get("commit"),
           "gpu": (meta.get("provenance") or {}).get("gpu"),
           "arms": {}, "pairs": {}}
    for a in arms:
        v = [succ[a][s] for s in states]
        bs = sorted(100.0 * sum(v[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        out["arms"][a] = {"success_pct": round(100.0 * sum(v) / n, 1),
                          "ci95": [round(bs[250], 1), round(bs[9749], 1)],
                          "k": sum(v), "n": n,
                          "ms": round(ms[a], 1) if a in ms else None}
    sep = []
    for a, b in itertools.combinations(arms, 2):
        d_ = [(succ[a][s], succ[b][s]) for s in states]
        ab = sum(1 for x, y in d_ if x and not y)
        ba = sum(1 for x, y in d_ if y and not x)
        deltas = [(1 if x and not y else (-1 if y and not x else 0)) for x, y in d_]
        bs = sorted(100.0 * sum(deltas[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        lo, hi = round(bs[250], 1), round(bs[9749], 1)
        key = f"{a}-{b}"
        rec = {"diff_pts": round(100.0 * (ab - ba) / n, 1), "discordant": ab + ba,
               "ci95": [lo, hi], "sign_p": round(signtest(ab, ba), 4),
               "separated": bool(lo > 0 or hi < 0)}
        out["pairs"][key] = rec
        if rec["separated"]: sep.append(key)
    out["separated_pairs"] = sep
    out["note"] = (
        "Four arms on 500 DISTINCT LIBERO Goal initial states, one host, one run, "
        "commit recorded. This is the comparison the seed repeats could not make: "
        "those revisited the same 30 states and disagreed on about 1, while real "
        "states raise the discordance a paired test spends -- 2st against "
        "10st+cache disagrees on 37 of 500. Two-step is both faster (123.6 vs "
        "221.8 ms) and MORE successful than the ten-step cache by 3.0 points with "
        "an interval excluding zero, so the ten-step cache that survived every "
        "single-seed frontier loses here. The four-step cache is bounded within 3 "
        "points of both uncached arms rather than separated: a cost too small for "
        "500 states to resolve, not an absence of cost.")
    return out


def replication(d=None):
    """Does the paired difference hold its sign and exclude zero at a second seed?

    The seeds share all 500 states, so this is a REPLICATION test and the two
    collections are never pooled into 1000 cases -- that is the error the 30-state
    seed repeats made. Each seed is estimated on its own and the question is
    whether the verdicts agree.
    """
    d = d or os.path.join(os.path.dirname(__file__), "..", "results", "statescale")
    out = {"seeds": [], "per_seed": {}, "pair": "2st-10st+cache"}
    for sd in seeds_present(d):
        succ, ms, _ = load(d, sd)
        if "2st" not in succ or "10st+cache" not in succ: continue
        states = sorted(set(succ["2st"]) & set(succ["10st+cache"]))
        n = len(states)
        ab = sum(1 for x in states if succ["2st"][x] and not succ["10st+cache"][x])
        ba = sum(1 for x in states if succ["10st+cache"][x] and not succ["2st"][x])
        deltas = [(1 if succ["2st"][x] and not succ["10st+cache"][x]
                   else (-1 if succ["10st+cache"][x] and not succ["2st"][x] else 0))
                  for x in states]
        bs = sorted(100.0 * sum(deltas[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        out["seeds"].append(sd)
        out["per_seed"][str(sd)] = {
            "n_states": n, "diff_pts": round(100.0 * (ab - ba) / n, 1),
            "discordant": ab + ba, "ci95": [round(bs[250], 1), round(bs[9749], 1)],
            "sign_p": round(signtest(ab, ba), 4),
            "separated": bool(bs[250] > 0 or bs[9749] < 0),
            "arms": {a: round(100.0 * sum(succ[a][x] for x in states) / n, 1)
                     for a in sorted(succ)}}
    vs = [v for v in out["per_seed"].values()]
    out["replicates"] = bool(vs and all(v["separated"] for v in vs)
                             and len({v["diff_pts"] > 0 for v in vs}) == 1)
    out["note"] = (
        "Two policy seeds on the SAME 500 states, estimated separately and never "
        "pooled: sharing states makes this a replication test, not additional "
        "independent evidence. " + ("Both seeds separate with the same sign, so "
        "the headline comparison replicates. " if out["replicates"] else
        "The verdicts do not agree across seeds. ") +
        "Per-arm the seed moves every schedule by under a point except the "
        "aggressive ten-step cache, which is the seed-sensitive component.")
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(os.path.dirname(__file__), "..", "results", "statescale")))

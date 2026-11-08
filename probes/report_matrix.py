"""The completed 500-state matrix: reference, plain cache, fitted global, per-cell.

The state-scale collection compared
schedules but carried no full-compute reference and no plain beta=1 cache, so it
could not show what attenuation CONTRIBUTES at that scale. And the global-weight
ablation used the median of the fitted per-cell weights where the objective
defines a pooled scalar -- the median being the weaker of the two, that
comparison could not establish per-cell superiority over a correctly fitted
single scalar.

This uses the matrix's own four arms, including a separately rerun per-cell arm,
on the same 500 initial states, and reports:

  attenuation's contribution   per-cell (or global) against plain beta=1 reuse
  the reference                how far each cached arm sits below full compute
  per-cell against global      the ablation, now against the POOLED fit

The additional short-schedule comparisons use the state-scale collection, with
a different host and code revision. Matching initial states does not remove
those confounds from success differences. Timings are printed per collection
and never compared across them. Within-matrix comparisons use its own per-cell
rerun, but global and per-cell fits still use unequal calibration splits.
"""
import glob, itertools, json, os, random, sys
from collections import defaultdict
from math import comb

random.seed(20260919)
B = 10000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(d, label_map, seed=42):
    """Arms from ONE seed. Seed variants live beside the seed-42 artefacts under
    a _seedN suffix and visit the same initial states, so taking the whole
    directory merges them: the later file overwrites the earlier and the mixture
    reports as one collection. Stripping the suffix to find the stem is exactly
    how that happens, and it did -- 2st read 97.2% against the seed-42 96.6%."""
    succ, lat, meta = defaultdict(dict), defaultdict(list), {}
    pat = "*.json" if seed == 42 else "*_seed%d.json" % seed
    for f in sorted(glob.glob(os.path.join(d, pat))):
        b = os.path.basename(f)[:-5]
        if b == "gate.json" or b.startswith("_"): continue
        if seed == 42 and "_seed" in b: continue
        stem = b.rsplit("_seed", 1)[0]
        if stem not in label_map: continue
        j = json.load(open(f)); m = j.get("meta") or {}
        meta.setdefault("provenance", m.get("provenance") or {})
        for r in j["rows"]:
            arm = label_map[stem]
            succ[arm][r["init_state"]] = bool(r["success"])
            if not r.get("warmup") and "ms_per_cycle" in r:
                lat[arm].append(r["ms_per_cycle"])
    return succ, {a: sum(v) / len(v) for a, v in lat.items() if v}, meta


def signtest(pos, neg):
    m = pos + neg
    if m == 0: return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(0, min(pos, neg) + 1)) / 2 ** m)


def paired(succ, states, a, b):
    d = [(succ[a][s], succ[b][s]) for s in states]
    ab = sum(1 for x, y in d if x and not y)
    ba = sum(1 for x, y in d if y and not x)
    n = len(states)
    deltas = [(1 if x and not y else (-1 if y and not x else 0)) for x, y in d]
    bs = sorted(100.0 * sum(deltas[random.randrange(n)] for _ in range(n)) / n
                for _ in range(B))
    return {"diff": round(100.0 * (ab - ba) / n, 1), "disc": ab + ba,
            "ci95": [round(bs[250], 1), round(bs[9749], 1)],
            "p": round(signtest(ab, ba), 4),
            "sep": bool(bs[250] > 0 or bs[9749] < 0)}


def main(d=None):
    d = d or os.path.join(ROOT, "results", "matrix")
    new, ms_new, meta = load(d, {"m10full": "10st full", "m10plain": "10st plain",
                                 "m10global": "10st global",
                                 "m10percell": "10st per-cell"})
    old, ms_old, _ = load(os.path.join(ROOT, "results", "statescale"),
                          {"s2": "2st", "s4": "4st"})
    if not new:
        print(f"  no matrix arms found in {d} -- skipping"); return 0
    # matrix arms win on name collision: every within-matrix comparison must come
    # from the matrix's own host, not be silently mixed with the state-scale run
    succ = {**old, **new}
    arms = [a for a in ("10st full", "10st plain", "10st global", "10st per-cell", "2st")
            if a in succ]
    states = sorted(set.intersection(*(set(succ[a]) for a in arms)))
    pv = meta.get("provenance") or {}
    print(f"Completed matrix: {len(arms)} arms on {len(states)} shared initial states")
    print(f"New arms: commit {(pv.get('commit') or 'NOT RECORDED')[:12]}  GPU {pv.get('gpu')}\n")
    print(f"  {'arm':16s} {'success':>9s}  {'95% CI':>16s}   {'ms':>8s}  collection")
    for a in arms:
        v = [succ[a][s] for s in states]; n = len(v)
        bs = sorted(100.0 * sum(v[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        ms = ms_new.get(a) or ms_old.get(a)
        coll = "matrix" if a in new else "state-scale"
        print(f"  {a:16s} {100.0*sum(v)/n:8.1f}%  [{bs[250]:5.1f},{bs[9749]:5.1f}]   "
              f"{ms:8.1f}  {coll}" if ms else
              f"  {a:16s} {100.0*sum(v)/n:8.1f}%  [{bs[250]:5.1f},{bs[9749]:5.1f}]   {'-':>8s}  {coll}")

    print(f"\n  {'comparison':34s} {'diff':>6s} {'disc':>5s} {'95% CI':>15s} {'p':>7s}  verdict")
    QS = [("10st per-cell", "10st plain", "attenuation's contribution (per-cell)"),
          ("10st global", "10st plain", "attenuation's contribution (global)"),
          ("10st per-cell", "10st global", "per-cell vs POOLED-fit global"),
          ("10st full", "10st per-cell", "how far the cache sits below full"),
          ("10st full", "2st", "full compute vs two-step"),
          ("2st", "10st per-cell", "two-step vs the cache")]
    for a, b, lab in QS:
        if a not in succ or b not in succ: continue
        r = paired(succ, states, a, b)
        v = ("SEPARATED" if r["sep"] else
             ("bounded <3pts" if max(abs(r["ci95"][0]), abs(r["ci95"][1])) <= 3 else "unresolved"))
        print(f"  {lab:34s} {r['diff']:+5.1f}% {r['disc']:5d} "
              f"[{r['ci95'][0]:+5.1f},{r['ci95'][1]:+5.1f}] {r['p']:7.3f}  {v}")
    print("\n  Matching initial states does not remove host or code-revision effects.")
    print("  Comparisons with state-scale arms are cross-collection success contrasts,")
    print("  not controlled method effects. The ms column must not be compared across")
    print("  collections. Within-matrix global/per-cell fits use unequal calibration splits.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

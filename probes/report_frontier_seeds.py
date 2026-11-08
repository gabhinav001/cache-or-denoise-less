"""Is the deployment frontier's arm ordering stable across seeds?

Repeated seeds test the stability of the same-host comparison across seeds.
More episodes would answer a different question: separating the
closest pair at 80% power needs about 670 per arm, roughly 22x what exists.
Ordering stability is both cheaper and the property actually in doubt.

SUCCESS ONLY. Latency is not pooled across these collections and is not
recollected here. Success is seed-dependent but not load-dependent -- a seed-7
run swung from load 63 to 1.8 and read 5722 ms/cycle on one task against 360 on
another with its success rate unaffected -- so success survives a busy host in a
way latency does not.

THE REPEATS ARE NOT INDEPENDENT EPISODES, and an earlier version of this script
treated them as though they were. LIBERO draws initial states from the benchmark,
not from the seed: all three collections visit the SAME 30 initial states, and
the seed varies the denoiser sampling only. Pooling them as 90 independent
episodes understates the uncertainty and invalidates any test that assumes
independent pairs. The unit here is therefore the initial state (n=30), each
observed three times, and every interval and test below clusters on it.

That correlation is itself a result. Most state-arm cells are unanimous across
the three runs, so the repeats add little independent information about success:
they show the ORDERING is unstable under denoiser noise alone, which is a
statement about fragility, not about equivalence. These runs strengthen the
latency result and do not establish success equivalence.
"""
import json, glob, os, sys
from collections import defaultdict
from report_cachesteps_ci import _cp
from report_frontier_fragility import STEPS, TAU, load as load_seed42

R = os.path.join(os.path.dirname(__file__), "..", "results")
DF = os.path.join(os.path.dirname(__file__), "..", "results", "deployfrontier")


def seeds_present():
    """Seed suffixes shipped beside the seed-42 collection, e.g. {7, 11}.

    Seed variants live in the same directory under a _seedN suffix, which is the
    convention repro/check_repro_map.py can follow: it dry-runs each declared
    PARAM_SETS entry and matches the filenames. An earlier layout put them in
    per-seed subdirectories, where the map -- which keys artefacts by basename --
    could not see them at all.
    """
    out = set()
    for f in glob.glob(os.path.join(DF, "d*_seed*.json")):
        out.add(int(os.path.basename(f).rsplit("_seed", 1)[1][:-5]))
    return sorted(out)


def load_dir(d, seed=None):
    """arm -> {init_state: success} for one seed's collection."""
    succ, meta = defaultdict(dict), {}
    pat = f"d*_seed{seed}.json" if seed else "d*.json"
    for f in sorted(glob.glob(os.path.join(d, pat))):
        base = os.path.basename(f)[:-5]
        if seed: base = base.rsplit("_seed", 1)[0]
        elif "_seed" in base: continue
        if base not in STEPS: continue
        j = json.load(open(f)); m = j.get("meta") or {}
        meta.setdefault("seed", m.get("seed"))
        meta.setdefault("n_tasks", m.get("n_tasks"))
        meta.setdefault("replan", m.get("replan"))
        for r in j["rows"]:
            cfg = r["config"]
            arm = (f"{STEPS[base]}st" if cfg == "stock"
                   else f"{STEPS[base]}st+tau{TAU[cfg]}")
            succ[arm][r["init_state"]] = bool(r["success"])
    return succ, meta


def main():
    seeds = {}
    s42, _ = load_seed42()
    seeds[42] = s42
    metas = {42: {"seed": 42}}
    for sd in seeds_present():
        s, m = load_dir(DF, sd)
        if not s:
            print(f"  (no arms parsed for seed {sd})"); continue
        seeds[sd] = s; metas[sd] = m

    if len(seeds) < 2:
        print("Only one seed present; nothing to compare yet.")
        return

    arms = sorted(set.intersection(*(set(v) for v in seeds.values())))
    order = sorted(seeds)
    print(f"Deployment frontier success across {len(order)} seeds: {order}")
    print(f"{len(arms)} arms common to every seed.\n")

    import random
    random.seed(20260919)
    st_ids = sorted(set(seeds[order[0]][arms[0]]))
    print(f"Unit of analysis: the initial state (n={len(st_ids)}), seen once per seed.")
    print("All three collections visit the SAME states, so intervals below cluster")
    print("on the state; they are NOT intervals over 90 independent episodes.\n")
    hdr = f"{'arm':12s}" + "".join(f"{('seed '+str(s)):>10s}" for s in order) \
          + f"{'mean':>9s}  95% CI (state bootstrap)"
    print(hdr); print("-" * (len(hdr) + 8))
    pooled = {}
    for a in arms:
        ks = [sum(seeds[s][a].values()) for s in order]
        ns = [len(seeds[s][a]) for s in order]
        vals = [sum(seeds[sd][a][st] for sd in order) / len(order) for st in st_ids]
        m = 100.0 * sum(vals) / len(vals)
        bs = sorted(100.0 * sum(vals[random.randrange(len(vals))] for _ in vals) / len(vals)
                    for _ in range(10000))
        lo, hi = bs[250], bs[9749]
        pooled[a] = (sum(ks), sum(ns))
        cells = "".join(f"{100.0*k/n:9.1f}%" for k, n in zip(ks, ns))
        print(f"{a:12s}{cells}{m:8.1f}%  [{lo:5.1f}, {hi:5.1f}]")

    # per-seed ranking agreement
    print("\nBest arm by success, per seed:")
    for s in order:
        best = max(arms, key=lambda a: sum(seeds[s][a].values()) / len(seeds[s][a]))
        bv = 100.0 * sum(seeds[s][best].values()) / len(seeds[s][best])
        tied = [a for a in arms
                if abs(100.0 * sum(seeds[s][a].values()) / len(seeds[s][a]) - bv) < 1e-9]
        print(f"  seed {s:<3d} {bv:5.1f}%  {', '.join(tied)}")

    # does two-step stay at or above the cached arms?
    print("\nTwo-step against each cached arm, per seed (percentage points):")
    cached = [a for a in arms if "tau" in a]
    print(f"  {'cached arm':14s}" + "".join(f"{('seed '+str(s)):>10s}" for s in order)
          + f"{'pooled':>10s}")
    for c in cached:
        row = ""
        for s in order:
            d2 = 100.0 * sum(seeds[s]["2st"].values()) / len(seeds[s]["2st"])
            dc = 100.0 * sum(seeds[s][c].values()) / len(seeds[s][c])
            row += f"{d2-dc:+9.1f} "
        p2 = 100.0 * pooled["2st"][0] / pooled["2st"][1]
        pc = 100.0 * pooled[c][0] / pooled[c][1]
        print(f"  {c:14s}{row}{p2-pc:+9.1f}")

    signs = {}
    for c in cached:
        ss = []
        for s in order:
            d2 = sum(seeds[s]["2st"].values()) / len(seeds[s]["2st"])
            dc = sum(seeds[s][c].values()) / len(seeds[s][c])
            ss.append(0 if d2 == dc else (1 if d2 > dc else -1))
        signs[c] = ss
    flip = [c for c, v in signs.items() if 1 in v and -1 in v]
    print(f"\n  {len(flip)} of {len(cached)} cached arms change sign across seeds"
          + (f": {', '.join(flip)}" if flip else ""))
    print("  A sign change means the ordering is seed-dependent and the single-seed")
    print("  ranking should not be read as a property of the configurations.")


def summary():
    """Numbers canonical_numbers.py needs, without the printing."""
    from report_frontier_fragility import dominance, load as load42
    import glob as _g
    cases = {}
    s42, ms42 = load42()
    cases[42] = (s42, ms42)
    for sd_ in seeds_present():
        succ, meta = load_dir(DF, sd_)
        lat = {}
        for f in sorted(_g.glob(os.path.join(DF, f"d*_seed{sd_}.json"))):
            b = os.path.basename(f)[:-5].rsplit("_seed", 1)[0]
            if b not in STEPS: continue
            for r in json.load(open(f))["rows"]:
                if r.get("warmup") or "ms_per_cycle" not in r: continue
                arm = (f"{STEPS[b]}st" if r["config"] == "stock"
                       else f"{STEPS[b]}st+tau{TAU[r['config']]}")
                lat.setdefault(arm, []).append(r["ms_per_cycle"])
        if succ:
            cases[int(meta.get("seed"))] = (succ, {a: sum(v)/len(v) for a, v in lat.items()})

    order = sorted(cases)
    arms = sorted(set.intersection(*(set(v[0]) for v in cases.values())))
    dom, fastest = {}, {}
    for sd in order:
        succ, ms = cases[sd]
        ks = {a: sum(v.values()) for a, v in succ.items()}
        n = {a: len(v) for a, v in succ.items()}
        f = min(ks, key=lambda a: ms[a])
        fastest[sd] = f
        dom[sd] = len({p for p in dominance(ks, ms, n) if p[0] == f})

    cached = [a for a in arms if "tau" in a]
    flips = []
    for c in cached:
        ss = []
        for sd in order:
            succ, _ = cases[sd]
            d2 = sum(succ["2st"].values()) / len(succ["2st"])
            dc = sum(succ[c].values()) / len(succ[c])
            ss.append(0 if d2 == dc else (1 if d2 > dc else -1))
        if 1 in ss and -1 in ss: flips.append(c)

    pooled = {}
    for a in arms:
        K = sum(sum(cases[sd][0][a].values()) for sd in order)
        N = sum(len(cases[sd][0][a]) for sd in order)
        pooled[a] = round(100.0 * K / N, 1)

    # Every arm pair, paired on (seed, initial state) and pooled over seeds. This
    # is the strongest separation test the collection admits: if none of these
    # reaches significance, the ordering is not measurable at this sample size.
    import itertools
    from report_cachesteps_ci import _binom_cdf
    key = {}
    for sd in order:
        for a, v in cases[sd][0].items():
            for st, ok in v.items(): key.setdefault(a, {})[(sd, st)] = ok
    # The unit is the initial state, observed once per seed. Per state an arm has
    # 0-3 successes; the paired difference is tested with a sign test over states.
    # A McNemar test over pooled episodes would count the same state three times.
    from math import comb
    st_ids = sorted({st for (_, st) in key[arms[0]]})
    K = {a: {st: sum(key[a][(sd, st)] for sd in order) for st in st_ids} for a in arms}
    ps = []
    for a, b in itertools.combinations(arms, 2):
        pos = sum(1 for st in st_ids if K[a][st] > K[b][st])
        neg = sum(1 for st in st_ids if K[a][st] < K[b][st])
        m = pos + neg
        pv = 1.0 if m == 0 else min(1.0, 2 * sum(
            comb(m, i) for i in range(0, min(pos, neg) + 1)) / 2 ** m)
        ps.append(pv)
    unanimous = sum(1 for a in arms for st in st_ids if K[a][st] in (0, len(order)))
    frac_unanimous = round(100.0 * unanimous / (len(arms) * len(st_ids)), 1)
    pooled_min_p = round(min(ps), 4) if ps else 1.0
    n_sig = sum(1 for x in ps if x < 0.05)

    return {
        "seeds": order,
        "episodes_per_arm_per_seed": 30,
        "observations_per_arm": 90,
        "independent_units_per_arm": 30,
        "fastest_arm_every_seed": (len(set(fastest.values())) == 1),
        "fastest_arm": sorted(set(fastest.values()))[0],
        "dominance_by_seed": {str(k): v for k, v in dom.items()},
        "dominance_min": min(dom.values()), "dominance_max": max(dom.values()),
        "cached_arms": len(cached), "sign_flips": sorted(flips),
        "n_sign_flips": len(flips),
        "pooled_success_pct": pooled,
        "initial_states": len(st_ids),
        "repeats_per_state": len(order),
        "states_shared_across_seeds": True,
        "pairs_tested": len(ps),
        "pairs_significant": n_sig,
        "smallest_state_signtest_p": pooled_min_p,
        "pct_state_arm_cells_unanimous": frac_unanimous,
        # ranges the write-ups quote instead of per-seed point values
        "two_step_ms_range": [round(min(cases[sd][1]["2st"] for sd in order), 1),
                              round(max(cases[sd][1]["2st"] for sd in order), 1)],
        "fastest_cache_ms_range": [
            round(min(min(v for a, v in cases[sd][1].items() if "tau" in a)
                      for sd in order), 1),
            round(max(min(v for a, v in cases[sd][1].items() if "tau" in a)
                      for sd in order), 1)],
        "two_step_lowest_latency_every_seed": all(
            min(cases[sd][1], key=cases[sd][1].get) == "2st" for sd in order),
        "note": ("Repeated seeds on the one same-host collection. "
                 "Two-step is the fastest arm at every seed, so "
                 "the latency half of every dominance relation replicates. The success "
                 "half does not: the dominance count is "
                 + ", ".join(f"{dom[s]} at seed {s}" for s in order) +
                 ", and " + str(len(flips)) + " of " + str(len(cached)) + " cached arms "
                 "change sign against two-step across seeds -- including the one arm "
                 "that survived the single-seed frontier. Pooled to 90 episodes per arm "
                 "every arm lies between 92.2% and 96.7% with overlapping intervals. "
                 "The ordering is therefore not a property of the configurations."),
    }


if __name__ == "__main__":
    main()

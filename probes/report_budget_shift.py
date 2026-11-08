"""Analysis for the denoising-budget / cache-coverage / noise experiments.

Reads the fresh-process collections and writes one JSON snapshot that the
paper's tables and figures are generated from, plus a text report.

  python3 probes/report_budget_shift.py [--out results/budget_shift/summary.json]
                                        [--results DIR]

--results points the analysis at another results tree with the same layout, for
instance one regenerated on a new host by repro/regenerate_paper.sh; sections
whose runs are absent from that tree are skipped.

Conventions, applied everywhere:
  * Success is paired on (task, initial-state hash). A comparison uses only the
    states both arms visited and refuses to run if the arms visited different
    task sets.
  * p-values are exact two-sided McNemar on discordant pairs. Intervals for a
    success difference are percentile bootstrap over states (10,000 resamples,
    fixed generator), so they respect the pairing.
  * Latency is the per-cycle wall time around one replanning call, excluding the
    first (compile/warm-up) episode of every task. The episode-level mean of
    per-cycle means is what earlier tables reported; per-cycle pooled quantiles
    are what a control loop meets, and refresh and reuse cycles are split.
  * Latency is compared only within one host.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def use_results(results: Path):
    """Point every path below at one results tree (the shipped one by default)."""
    global R, NEW, HOST_A, HOST_B, REPEATS
    R = Path(results).absolute()
    NEW = R / "budget_shift"
    HOST_A = R / "matched50_a6000"       # 50 states, RTX A6000
    HOST_B = R / "matched100_a6000"     # 100 + 50 states, RTX A6000
    REPEATS = NEW / "repeats_a6000"      # second copies of arms that ran on two new hosts


use_results(ROOT / "results")


def rel(path: Path) -> str:
    """A run's path as recorded in the snapshot: relative to the tree holding results/."""
    return str(Path(path).absolute().relative_to(R.parent))
B = 10_000
EQ_MARGIN = 5.0   # percentage points; declared before the new collections were analysed


# ----------------------------------------------------------------- loading
def load(path: Path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    rows = d["rows"]
    meta = d["meta"]
    # fw_libero.py rewrites its artefact after every task, so a file copied off a
    # running host can hold only the early tasks. Use complete arms only.
    expected = int(meta["n_tasks"]) * int(meta["n_episodes"]) * len(meta.get("configs", [None]))
    if len(rows) != expected:
        print(f"  (incomplete: {path.name} has {len(rows)}/{expected} episodes; skipped)")
        return None
    outcome, steps = {}, {}
    for r in rows:
        key = (r["task"], r["init_state"])
        if key in outcome:
            raise ValueError(f"{path}: duplicate state {key}")
        outcome[key] = bool(r["success"])
        steps[key] = int(r["steps"])
    return {"path": rel(path), "meta": meta, "rows": rows,
            "outcome": outcome, "steps": steps}


def arm(dirpath: Path, name: str):
    """One arm by name. The new collection is split into purpose subdirectories
    (frontier100_a6000, latency_a6000, shift_a6000, noise_a6000, long_a6000);
    names are unique across them, so search recursively and refuse duplicates.
    repeats_a6000 holds second copies of arms that ran on two hosts; they enter
    only the determinism check, through dirpath=REPEATS."""
    if dirpath == NEW:
        hits = sorted(h for h in NEW.rglob(f"{name}.json") if REPEATS not in h.parents)
        if len(hits) > 1:
            raise ValueError(f"{name}: found in several subdirectories: {hits}")
        return load(hits[0]) if hits else None
    return load(dirpath / f"{name}.json")


def rate(a):
    v = list(a["outcome"].values())
    return {"k": int(sum(v)), "n": len(v), "pct": round(100.0 * sum(v) / len(v), 1)}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


# ----------------------------------------------------------------- statistics
def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def tango_ci(b01, b10, n, conf=0.90):
    """Tango (1998) score interval for a paired difference of proportions, in points.

    b01 counts states where only the first arm succeeded, b10 where only the
    second did. The score statistic is monotone in the difference, so each bound
    is found by bisection. Non-inferiority at margin m holds when the lower bound
    of the 90% interval exceeds -m (a one-sided test at alpha = 0.05)."""
    z = NormalDist().inv_cdf(0.5 + conf / 2)

    def score(d):
        qa = 2 * n
        qb = -b01 - b10 + (2 * n - b01 + b10) * d
        qc = -b10 * d * (1 - d)
        q21 = (math.sqrt(max(qb * qb - 4 * qa * qc, 0.0)) - qb) / (2 * qa)
        var = n * (2 * q21 + d * (1 - d))
        num = b01 - b10 - n * d
        return num / math.sqrt(var) if var > 0 else math.copysign(math.inf, num)

    def solve(lo, hi, target):
        for _ in range(200):
            mid = (lo + hi) / 2
            if score(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    est = (b01 - b10) / n
    lower = solve(-1 + 1e-12, est, z)
    upper = solve(est, 1 - 1e-12, -z)
    return [round(100 * lower, 1), round(100 * upper, 1)]


def paired(a, b, label=""):
    """Success of a minus success of b on the states both visited."""
    ta = {t for t, _ in a["outcome"]}
    tb = {t for t, _ in b["outcome"]}
    if ta != tb:
        raise ValueError(f"{label}: arms cover different tasks {sorted(ta)} vs {sorted(tb)}")
    keys = sorted(set(a["outcome"]) & set(b["outcome"]))
    if not keys:
        raise ValueError(f"{label}: no shared states")
    x = np.array([a["outcome"][k] for k in keys], dtype=float)
    y = np.array([b["outcome"][k] for k in keys], dtype=float)
    b01 = int(((x == 1) & (y == 0)).sum())
    b10 = int(((x == 0) & (y == 1)).sum())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    boots = (x[idx] - y[idx]).mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5, 95])
    # Equivalence at a declared margin: two one-sided tests at alpha=0.05 are
    # equivalent to the 90% interval lying inside (-margin, +margin).
    return {"label": label, "n": len(keys), "a_pct": round(100 * x.mean(), 1),
            "b_pct": round(100 * y.mean(), 1), "diff": round(100 * (x.mean() - y.mean()), 1),
            "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "ci90": [round(float(lo90), 1), round(float(hi90), 1)],
            "equivalent_within_5pts": bool(lo90 > -EQ_MARGIN and hi90 < EQ_MARGIN),
            "tango90": tango_ci(b01, b10, len(keys)),
            "a_only": b01, "b_only": b10, "p": mcnemar_exact(b01, b10)}


def did(a1, a2, b1, b2, label=""):
    """[a1 - a2] - [b1 - b2] on shared states, bootstrap over states."""
    keys = sorted(set(a1["outcome"]) & set(a2["outcome"]) & set(b1["outcome"]) & set(b2["outcome"]))
    v = {n: np.array([d["outcome"][k] for k in keys], float) for n, d in
         (("a1", a1), ("a2", a2), ("b1", b1), ("b2", b2))}
    est = (v["a1"] - v["a2"]) - (v["b1"] - v["b2"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    boots = est[idx].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    frac_nonpos = float((boots <= 0).mean())
    return {"label": label, "n": len(keys),
            "a_effect": round(100 * float((v["a1"] - v["a2"]).mean()), 1),
            "b_effect": round(100 * float((v["b1"] - v["b2"]).mean()), 1),
            "did": round(100 * float(est.mean()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "boot_p_one_sided": round(frac_nonpos, 4)}


def holm(results, key="p"):
    """Holm step-down adjustment, in place, as p_holm."""
    order = sorted(range(len(results)), key=lambda i: results[i][key])
    m, running = len(results), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i][key])
        running = max(running, adj)
        results[i]["p_holm"] = running
    return results


def determinism(a, b, label=""):
    keys = sorted(set(a["outcome"]) & set(b["outcome"]))
    flips = sum(a["outcome"][k] != b["outcome"][k] for k in keys)
    traj = sum(a["steps"][k] != b["steps"][k] for k in keys)
    pa, pb = a["meta"].get("provenance", {}), b["meta"].get("provenance", {})
    return {"label": label, "n": len(keys), "success_flips": flips, "step_count_differs": traj,
            "gpu_a": pa.get("gpu"), "gpu_b": pb.get("gpu"),
            "host_a": pa.get("host_tag"), "host_b": pb.get("host_tag")}


def latency(a):
    """Episode-level and per-cycle latency, warm-up episodes excluded."""
    ep = [r["ms_per_cycle"] for r in a["rows"] if not r.get("warmup") and "ms_per_cycle" in r]
    cyc, ref, reu = [], [], []
    for r in a["rows"]:
        if r.get("warmup") or "cycle_ms" not in r:
            continue
        for ms, is_ref in zip(r["cycle_ms"], r["cycle_refresh"]):
            cyc.append(ms)
            (ref if is_ref else reu).append(ms)
    out = {"episodes": len(ep)}
    if ep:
        e = np.array(ep)
        out.update(ep_mean=round(float(e.mean()), 1), ep_sd=round(float(e.std(ddof=1)) if len(e) > 1 else 0.0, 1))
    if cyc:
        c = np.sort(np.array(cyc))
        def q(p):  # nearest-rank quantile, the convention used in earlier reports
            return round(float(c[max(0, math.ceil(p * len(c)) - 1)]), 1)
        out.update(cycles=len(c), cyc_mean=round(float(c.mean()), 1), cyc_median=q(0.5), cyc_p95=q(0.95),
                   cyc_max=round(float(c[-1]), 1),
                   refresh_mean=round(float(np.mean(ref)), 1) if ref else None,
                   reuse_mean=round(float(np.mean(reu)), 1) if reu else None,
                   refresh_share=round(len(ref) / len(c), 3))
    return out


def lat_diff(a, b, label=""):
    """Mean per-cycle latency of a minus b (same host), bootstrap over episodes."""
    ea = np.array([r["ms_per_cycle"] for r in a["rows"] if not r.get("warmup") and "ms_per_cycle" in r])
    eb = np.array([r["ms_per_cycle"] for r in b["rows"] if not r.get("warmup") and "ms_per_cycle" in r])
    rng = np.random.default_rng(0)
    ba = ea[rng.integers(0, len(ea), size=(B, len(ea)))].mean(axis=1)
    bb = eb[rng.integers(0, len(eb), size=(B, len(eb)))].mean(axis=1)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return {"label": label, "a_ms": round(float(ea.mean()), 1), "b_ms": round(float(eb.mean()), 1),
            "diff_ms": round(float(ea.mean() - eb.mean()), 1), "ratio": round(float(ea.mean() / eb.mean()), 3),
            "ci95": [round(float(lo), 1), round(float(hi), 1)]}


def config_key(meta):
    """Everything that decides a run's rollouts, normalised across harness revisions.

    Older revisions left out fields they did not vary, and runs from before the
    harness recorded its runtime used upstream's default action shift, 1.0 at the
    pinned Fast-WAM commit. Two runs with equal keys ran the same configuration.
    The attenuation weights are not recorded; every attenuated arm here uses the
    shipped per-cell weights except matrix/m10global.json, which EXCLUDED names.
    """
    rt = meta.get("runtime") or {}
    pv = meta.get("provenance") or {}
    ckpt = pv.get("ckpt") or (rt.get("dataset_stats") or "").replace("_dataset_stats.json", ".pt") \
        or "libero_optional_idm_2cam224.pt"
    k = int(meta.get("num_inference_steps") or 10)
    cfgs = tuple(meta.get("configs") or ())
    caching = any(c not in ("stock", "full") for c in cfgs)
    prefix = meta.get("cache_prefix")
    prefix = 0 if not caching else (k - 2 if prefix is None else int(prefix))
    # the per-episode latent and delayed execution (latent_delay.sh) are recorded
    # only when used; a run without them keeps the key it always had
    return (ckpt, meta.get("suite") or "libero_goal", k, float(rt.get("effective_action_sigma_shift", 1.0)),
            cfgs, prefix, int(meta.get("resample_noise") or 0), meta.get("perturb") or "none",
            int(meta.get("cudagraphs") or 0), int(meta.get("seed", 42)),
            int(meta.get("episode_noise") or 0), tuple(meta.get("delay") or (0, 0)))


# Runs whose configuration the metadata cannot distinguish from another's: the
# pooled single attenuation weight of matrix.sh records the same config label as
# the per-cell weights.
EXCLUDED = {"matrix/m10global.json"}
DETERMINISM_DIRS = ("budget_shift", "matched50_a6000", "matched100_a6000", "composed2step",
                    "matched_composed", "statescale", "matrix", "tau8_fresh")


def exhaustive_determinism():
    """Every pair of single-configuration runs with equal keys, on different machines."""
    runs = []
    for d in DETERMINISM_DIRS:
        for p_ in sorted((R / d).rglob("*.json")) if (R / d).is_dir() else []:
            if rel(p_).split("results/", 1)[-1] in EXCLUDED or p_.name == "summary.json":
                continue
            try:
                j_ = json.loads(p_.read_text())
            except ValueError:
                continue
            if not isinstance(j_, dict) or "rows" not in j_ or len((j_.get("meta") or {}).get("configs") or []) != 1:
                continue
            a = load(p_)
            if a is not None:
                runs.append(a)
    groups = {}
    for a in runs:
        groups.setdefault(config_key(a["meta"]), []).append(a)
    out = {"scope": [f"results/{d}" for d in DETERMINISM_DIRS], "excluded": sorted(EXCLUDED), "runs": len(runs),
           "pairs": 0, "paired_episodes": 0, "outcome_changes": 0, "step_count_changes": 0,
           "same_host_pairs": 0, "same_host_paired_episodes": 0, "same_host_outcome_changes": 0,
           "hosts": set(), "harness_revisions": set(), "changes": []}
    states = set()
    for g_ in groups.values():
        for i in range(len(g_)):
            for j in range(i + 1, len(g_)):
                a, b = g_[i], g_[j]
                r_ = determinism(a, b)
                if not r_["n"]:
                    continue
                if r_["host_a"] == r_["host_b"]:
                    out["same_host_pairs"] += 1
                    out["same_host_paired_episodes"] += r_["n"]
                    out["same_host_outcome_changes"] += r_["success_flips"]
                    continue
                out["pairs"] += 1
                out["paired_episodes"] += r_["n"]
                out["outcome_changes"] += r_["success_flips"]
                out["step_count_changes"] += r_["step_count_differs"]
                out["hosts"].update([r_["host_a"], r_["host_b"]])
                for x in (a, b):
                    c_ = (x["meta"].get("provenance") or {}).get("commit")
                    if c_:
                        out["harness_revisions"].add(c_[:12])
                for k_ in sorted(set(a["outcome"]) & set(b["outcome"])):
                    if a["outcome"][k_] != b["outcome"][k_]:
                        states.add((config_key(a["meta"]), k_))
                        out["changes"].append({"a": a["path"], "b": b["path"], "task": k_[0], "init_state": k_[1],
                                               "success_a": a["outcome"][k_], "success_b": b["outcome"][k_],
                                               "steps_a": a["steps"][k_], "steps_b": b["steps"][k_]})
    out["hosts"] = sorted(h for h in out["hosts"] if h)
    out["harness_revisions"] = sorted(out["harness_revisions"])
    out["states_changed"] = len(states)          # distinct (configuration, initial state)
    return out


def host_of(a):
    p = a["meta"].get("provenance", {})
    return {"gpu": p.get("gpu"), "host_tag": p.get("host_tag"), "commit": p.get("commit"),
            "loadavg": p.get("loadavg_1_5_15"), "cpu_count": p.get("cpu_count")}


# ----------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="default: <results>/budget_shift/summary.json")
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="results tree to analyse (default: the shipped results/)")
    args = ap.parse_args()
    use_results(args.results)
    if args.out is None:
        args.out = str(NEW / "summary.json")
    S = {"conventions": __doc__.strip().splitlines()[0], "bootstrap_resamples": B}

    def need(d, n):
        a = arm(d, n)
        if a is None:
            print(f"  (missing {d.name}/{n}.json)")
        return a

    # ---------- 1. 100-state frontier (LIBERO Goal), one host -----------------
    front = {
        "idm": {"label": "Optional-IDM checkpoint, first-frame mode, s=1.0",
                "arms": {"uncached 1 step": "idm_stock_1step",
                         "uncached 2 steps": "idm_stock_2step",
                         "uncached 4 steps": "idm_stock_4step",
                         "uncached 10 steps": "idm_stock_10step",
                         "10 steps, cache tau=2": "idm_whole_plain_tau2",
                         "10 steps, cache tau=4": "idm_whole_plain_tau4",
                         "10 steps, cache tau=8": "idm_whole_plain_tau8",
                         "10 steps, cache tau=0": "idm_whole_plain_tau0",
                         "10 steps, attenuated cache tau=8": "idm_whole_agg_tau8",
                         "4 steps, cache tau=4": "idm_whole_4step_tau4"}},
        "native": {"label": "original checkpoint, s=5.0",
                   "arms": {"uncached 1 step": "uncond_stock_1step",
                            "uncached 2 steps": "uncond_stock_2step",
                            "uncached 4 steps": "uncond_stock_4step",
                            "uncached 10 steps": "uncond_stock_10step",
                            "10 steps, cache tau=0": "uncond_whole_tau0",
                            "10 steps, cache tau=2": "uncond_whole_tau2",
                            "10 steps, cache tau=4": "uncond_whole_tau4",
                            "10 steps, cache tau=8": "uncond_whole_tau8",
                            "4 steps, cache tau=4": "uncond_whole_4step_tau4"}},
    }
    cg = {"uncached 1 step": "lat_uncond_stock1_cg", "uncached 2 steps": "lat_uncond_stock2_cg",
          "uncached 4 steps": "lat_uncond_stock4_cg", "uncached 10 steps": "lat_uncond_stock10_cg"}
    print("\n== 1. Frontier on 100 LIBERO-Goal states (one host) ==")
    S["frontier"] = {}
    for ck, spec in front.items():
        print(f"-- {spec['label']}")
        rows = {}
        for lab, n in spec["arms"].items():
            a = need(NEW, n)
            if a is None:
                continue
            r = rate(a)
            L = latency(a)
            # CUDA-graph timings exist for the original checkpoint's uncached schedules only
            g = arm(NEW, cg[lab]) if (ck == "native" and lab in cg) else None
            rows[lab] = {"success": r, "wilson95": wilson(r["k"], r["n"]), "source": a["path"],
                         "host": host_of(a), "latency": L,
                         "latency_cudagraphs": latency(g) if g is not None else None}
            gtxt = f"  CUDA graphs {rows[lab]['latency_cudagraphs']['ep_mean']:6.1f}" if g is not None else ""
            print(f"   {lab:36s} {r['k']:3d}/{r['n']}  {r['pct']:5.1f}%  {L.get('ep_mean', float('nan')):6.1f} ms "
                  f"(p95 {L.get('cyc_p95')}, refresh {L.get('refresh_mean')}, reuse {L.get('reuse_mean')}){gtxt}")
        S["frontier"][ck] = {"label": spec["label"], "arms": rows}

    # primary contrasts (declared family, Holm-adjusted)
    print("\n   primary contrasts (Holm over this family):")
    fam = []
    pairs = [("idm", "uncached 2 steps", "uncached 10 steps"),
             ("idm", "uncached 2 steps", "10 steps, cache tau=8"),
             ("idm", "uncached 2 steps", "10 steps, attenuated cache tau=8"),
             ("idm", "uncached 2 steps", "10 steps, cache tau=2"),
             ("idm", "uncached 2 steps", "4 steps, cache tau=4"),
             ("idm", "10 steps, attenuated cache tau=8", "10 steps, cache tau=8"),
             ("native", "uncached 2 steps", "uncached 10 steps"),
             ("native", "uncached 2 steps", "10 steps, cache tau=8"),
             ("native", "uncached 2 steps", "10 steps, cache tau=0"),
             ("native", "uncached 2 steps", "4 steps, cache tau=4"),
             ("native", "uncached 1 step", "uncached 10 steps")]
    for ck, x, y in pairs:
        spec = front[ck]["arms"]
        a, b = arm(NEW, spec[x]), arm(NEW, spec[y])
        if a is None or b is None:
            continue
        fam.append(paired(a, b, f"{ck}: {x} vs {y}"))
    holm(fam)
    for c in fam:
        print(f"   {c['label']:62s} {c['diff']:+5.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] "
              f"{c['a_only']}/{c['b_only']} p={c['p']:.4f} holm={c['p_holm']:.4f}"
              f"{'  equiv<=5pts' if c['equivalent_within_5pts'] else ''}")
    S["primary_contrasts"] = fam

    # Descriptive, added for the round 10 reviews: the successes behind the primary
    # comparison task by task, and its Tango 90% interval with each task left out.
    per_task = {}
    for ck in ("idm", "native"):
        spec = front[ck]["arms"]
        runs_ = {lab: arm(NEW, spec[lab]) for lab in ("uncached 2 steps", "uncached 10 steps", "10 steps, cache tau=8")}
        if any(v is None for v in runs_.values()):
            continue
        two, ten = runs_["uncached 2 steps"], runs_["uncached 10 steps"]
        tasks = sorted({t for t, _ in two["outcome"]})
        succ = {lab: [int(sum(v for (t, _), v in r["outcome"].items() if t == task)) for task in tasks]
                for lab, r in runs_.items()}
        loo = []
        for task in tasks:
            keys = [k for k in two["outcome"] if k[0] != task and k in ten["outcome"]]
            b01 = sum(1 for k in keys if two["outcome"][k] and not ten["outcome"][k])
            b10 = sum(1 for k in keys if ten["outcome"][k] and not two["outcome"][k])
            loo.append(tango_ci(b01, b10, len(keys)))
        names_ = {r["task"]: r.get("task_name") for r in two["rows"]}
        per_task[ck] = {"tasks": tasks, "task_names": [names_.get(task) for task in tasks],
                        "states_per_task": [sum(1 for (t, _) in two["outcome"] if t == task) for task in tasks],
                        "success": succ,
                        "two_minus_ten": [x - y for x, y in zip(succ["uncached 2 steps"], succ["uncached 10 steps"])],
                        "loo_tango90_two_vs_ten": loo,
                        "loo_lowest_bound": min(lo for lo, _ in loo)}
        print(f"   {ck}: two minus ten steps by task {per_task[ck]['two_minus_ten']}; "
              f"lowest Tango 90% bound with one task left out {per_task[ck]['loo_lowest_bound']}")
    S["per_task"] = per_task

    # ---------- 2. cross-host determinism -------------------------------------
    print("\n== 2. Same configuration, different host ==")
    det = []
    same = ["idm_stock_1step", "idm_stock_2step", "idm_stock_4step", "idm_stock_10step",
            "idm_whole_plain_tau2", "idm_whole_agg_tau8", "uncond_stock_1step", "uncond_stock_2step",
            "uncond_stock_4step", "uncond_stock_10step", "uncond_whole_tau8", "uncond_whole_4step_tau4"]
    checks = [((NEW, n), (HOST_B, n)) for n in same] + [
              ((HOST_B, "idm_stock_2step"), (HOST_A, "idm_stock_2step")),
              ((HOST_B, "causal_idm_s1_whole_tau8"), (HOST_A, "idm_whole_plain_tau8")),
              ((NEW, "idm_whole_plain_tau8"), (HOST_A, "idm_whole_plain_tau8")),
              ((NEW, "idm_whole_plain_tau0"), (HOST_A, "idm_whole_plain_tau0")),
              ((NEW, "uncond_whole_tau0"), (HOST_A, "uncond_whole_tau0")),
              ((NEW, "idm_s1_stock10"), (HOST_A, "idm_stock_10step")),
              ((NEW, "uncond_s5_stock10"), (HOST_A, "uncond_stock_10step")),
              ((NEW, "idm_s1_whole_tau8"), (HOST_B, "causal_idm_s1_whole_tau8")),
              ((NEW, "idm_s5_whole_tau8"), (HOST_B, "causal_idm_s5_whole_tau8")),
              ((NEW, "uncond_s1_whole_tau8"), (HOST_B, "causal_uncond_s1_whole_tau8")),
              ((NEW, "uncond_s5_whole_tau8"), (HOST_B, "causal_uncond_s5_whole_tau8")),
              ((NEW, "idm_s1_stock2"), (HOST_B, "causal_idm_s1_stock2")),
              ((NEW, "uncond_s5_stock2"), (HOST_B, "causal_uncond_s5_stock2")),
              ((NEW, "uncond_s5_whole_tau0"), (HOST_A, "uncond_whole_tau0")),
              # the new hosts against each other (all RTX A6000, 50 shared states)
              ((NEW, "idm_s1_stock2"), (REPEATS, "idm_s1_stock2")),
              ((NEW, "uncond_s5_stock2"), (REPEATS, "uncond_s5_stock2")),
              ((NEW, "uncond_s5_whole_tau0"), (REPEATS, "uncond_s5_whole_tau0")),
              ((NEW, "idm_stock_10step"), (NEW, "idm_s1_stock10")),
              ((NEW, "uncond_stock_10step"), (NEW, "uncond_s5_stock10")),
              ((NEW, "idm_stock_2step"), (NEW, "idm_s1_stock2")),
              ((NEW, "uncond_stock_2step"), (NEW, "uncond_s5_stock2")),
              ((NEW, "idm_whole_plain_tau8"), (NEW, "idm_s1_whole_tau8")),
              ((NEW, "uncond_whole_tau8"), (NEW, "uncond_s5_whole_tau8")),
              ((NEW, "uncond_whole_tau0"), (NEW, "uncond_s5_whole_tau0"))]
    for (d1, n1), (d2, n2) in checks:
        a, b = arm(d1, n1), arm(d2, n2)
        if a is None or b is None:
            continue
        r = determinism(a, b, f"{d1.name}/{n1} vs {d2.name}/{n2}")
        det.append(r)
        print(f"   {r['label']:70s} n={r['n']:3d} flips={r['success_flips']:2d} "
              f"traj={r['step_count_differs']:2d}  [{r['host_a']} | {r['host_b']}]")
    S["determinism"] = det
    S["determinism_total"] = {
        "comparisons": len(det), "paired_episodes": sum(r["n"] for r in det),
        "outcome_changes": sum(r["success_flips"] for r in det),
        "step_count_changes": sum(r["step_count_differs"] for r in det),
        "hosts": sorted({h for r in det for h in (r["host_a"], r["host_b"]) if h})}
    print(f"   total: {S['determinism_total']['comparisons']} comparisons, "
          f"{S['determinism_total']['paired_episodes']} paired episodes, "
          f"{S['determinism_total']['outcome_changes']} outcome changes, "
          f"{len(S['determinism_total']['hosts'])} hosts")

    # the same configuration's latency on two hosts: outcomes transfer, timings do not
    xh = []
    for n in same:
        a, b = arm(NEW, n), arm(HOST_B, n)
        if a is None or b is None:
            continue
        la, lb = latency(a), latency(b)
        xh.append({"arm": n, "this_host": host_of(a)["host_tag"], "other_host": host_of(b)["host_tag"],
                   "this_host_ms": la.get("ep_mean"), "other_host_ms": lb.get("ep_mean"),
                   "ratio": round(lb["ep_mean"] / la["ep_mean"], 3) if la.get("ep_mean") and lb.get("ep_mean") else None})
    S["latency_across_hosts"] = xh
    for r in xh:
        print(f"   latency {r['arm']:28s} {r['this_host_ms']} ms here vs {r['other_host_ms']} ms on {r['other_host']} "
              f"(x{r['ratio']})")

    # ---------- 3. shift vs cached share of the flow path (50 states) ---------
    print("\n== 3. Schedule shift and cached share of the path (50 states, tau=8) ==")
    sh = []
    for ck, s, name in [("idm", 1.0, "idm_s1_stock10"), ("idm", 5.0, "idm_s5_stock10"),
                        ("native", 1.0, "uncond_s1_stock10"), ("native", 5.0, "uncond_s5_stock10"),
                        ("idm", 3.0, "idm_s3_stock10")]:
        a = need(NEW, name)
        if a:
            sh.append({"ckpt": ck, "shift": s, "arm": "uncached 10 steps", "prefix": 0, "fraction": 0.0,
                       "success": rate(a), "source": a["path"]})
    cached = [("idm", 1.0, 3, "idm_s1_p3_tau8"), ("idm", 1.0, 4, "idm_s1_p4_tau8"),
              ("idm", 1.0, 5, "idm_s1_p5_tau8"), ("idm", 1.0, 6, "idm_s1_p6_tau8"),
              ("idm", 1.0, 8, "idm_s1_whole_tau8"), ("idm", 5.0, 8, "idm_s5_whole_tau8"),
              ("idm", 5.0, 9, "idm_s5_p9_tau8"), ("native", 1.0, 5, "uncond_s1_p5_tau8"),
              ("native", 1.0, 8, "uncond_s1_whole_tau8"), ("native", 5.0, 8, "uncond_s5_whole_tau8"),
              ("native", 5.0, 9, "uncond_s5_p9_tau8"),
              # follow-up collection (repro/experiments/followup.sh): the 70% share
              # at s=1 and a third shift, where 8 and 9 cached steps cover 57% and 75%
              ("idm", 1.0, 7, "idm_s1_p7_tau8"), ("idm", 3.0, 8, "idm_s3_p8_tau8"),
              ("idm", 3.0, 9, "idm_s3_p9_tau8"), ("native", 1.0, 7, "uncond_s1_p7_tau8")]
    for ck, s, p, name in cached:
        a = need(NEW, name)
        if not a:
            continue
        frac = a["meta"].get("cached_path_fraction")
        if frac is None or frac == 0:
            u = 1 - p / 10
            frac = round(1 - s * u / (1 + (s - 1) * u), 4)
        sh.append({"ckpt": ck, "shift": s, "arm": f"cache tau=8, steps 0-{p - 1}", "prefix": p,
                   "fraction": frac, "success": rate(a), "source": a["path"]})
    for r in sh:
        print(f"   {r['ckpt']:6s} s={r['shift']:.0f} {r['arm']:26s} frac={r['fraction']:.3f} "
              f"{r['success']['k']:2d}/{r['success']['n']} {r['success']['pct']:5.1f}%")
    S["shift_sweep"] = sh

    print("\n   matched-budget interaction: [cache(s5)-cache(s1)] - [10-step(s5)-10-step(s1)]")
    inter = []
    for ck, pre in (("idm", "idm"), ("native", "uncond")):
        c5, c1 = arm(NEW, f"{pre}_s5_whole_tau8"), arm(NEW, f"{pre}_s1_whole_tau8")
        u5, u1 = arm(NEW, f"{pre}_s5_stock10"), arm(NEW, f"{pre}_s1_stock10")
        if None in (c5, c1, u5, u1):
            continue
        r = did(c5, c1, u5, u1, f"{ck}: shift effect on the tau=8 cache minus on uncached 10-step")
        inter.append(r)
        print(f"   {r['label']:66s} cache {r['a_effect']:+.1f}, uncached {r['b_effect']:+.1f}, "
              f"DiD {r['did']:+.1f} [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]")
    S["shift_interaction"] = inter

    print("\n   matched cached share across shifts:")
    mc = []
    for x, y in [("idm_s1_p4_tau8", "idm_s5_whole_tau8"), ("idm_s1_p5_tau8", "idm_s5_whole_tau8"),
                 ("uncond_s1_p5_tau8", "uncond_s5_whole_tau8"),
                 ("idm_s1_whole_tau8", "idm_s1_p4_tau8"), ("uncond_s1_whole_tau8", "uncond_s1_p5_tau8"),
                 ("idm_s5_p9_tau8", "idm_s5_whole_tau8"), ("uncond_s5_p9_tau8", "uncond_s5_whole_tau8")]:
        a, b = arm(NEW, x), arm(NEW, y)
        if a is None or b is None:
            continue
        c = paired(a, b, f"{x} vs {y}")
        mc.append(c)
        print(f"   {c['label']:46s} {c['a_pct']:5.1f} vs {c['b_pct']:5.1f}  {c['diff']:+5.1f} "
              f"[{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] {c['a_only']}/{c['b_only']} p={c['p']:.4f}")
    S["shift_matched"] = mc

    print("\n   checkpoint at the same shift (s=1), cached and uncached:")
    ck1 = []
    for x, y in [("idm_s1_whole_tau8", "uncond_s1_whole_tau8"), ("idm_s1_stock10", "uncond_s1_stock10")]:
        a, b = arm(NEW, x), arm(NEW, y)
        if a is None or b is None:
            continue
        c = paired(a, b, f"{x} vs {y}")
        ck1.append(c)
        print(f"   {c['label']:46s} {c['a_pct']:5.1f} vs {c['b_pct']:5.1f}  {c['diff']:+5.1f} "
              f"[{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] {c['a_only']}/{c['b_only']} p={c['p']:.4f}")
    S["checkpoint_at_s1"] = ck1

    # ---------- 4. fresh noise on every call (50 states) -----------------------
    print("\n== 4. Fixed latent vs fresh noise per call (50 states) ==")
    nz = []
    for fixed, fresh in [("idm_s1_stock2", "idm_s1_stock2_rn"), ("idm_s1_stock10", "idm_s1_stock10_rn"),
                         ("idm_s1_whole_tau8", "idm_s1_whole_tau8_rn"),
                         ("uncond_s5_stock2", "uncond_s5_stock2_rn"),
                         ("uncond_s5_stock10", "uncond_s5_stock10_rn"),
                         ("uncond_s5_whole_tau8", "uncond_s5_whole_tau8_rn"),
                         ("uncond_s5_whole_tau0", "uncond_s5_whole_tau0_rn"),
                         ("idm_s1_p4_tau8", "idm_s1_p4_tau8_rn"), ("idm_s1_p6_tau8", "idm_s1_p6_tau8_rn"),
                         ("idm_s5_whole_tau8", "idm_s5_whole_tau8_rn"),
                         ("idm_s1_p7_tau8", "idm_s1_p7_tau8_rn"), ("idm_s3_p8_tau8", "idm_s3_p8_tau8_rn"),
                         ("idm_s3_p9_tau8", "idm_s3_p9_tau8_rn")]:
        a, b = arm(NEW, fixed), arm(NEW, fresh)
        if a is None or b is None:
            continue
        nz.append({"arm": fixed, "fixed": rate(a), "fresh": rate(b),
                   "fresh_minus_fixed": paired(b, a, f"{fresh} vs {fixed}")})
        c = nz[-1]["fresh_minus_fixed"]
        print(f"   {fixed:22s} fixed {rate(a)['pct']:5.1f}  fresh {rate(b)['pct']:5.1f}  "
              f"{c['diff']:+5.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] p={c['p']:.3f}")
    fresh_c = []
    for x, y in [("idm_s1_stock2_rn", "idm_s1_stock10_rn"), ("idm_s1_stock2_rn", "idm_s1_whole_tau8_rn"),
                 ("idm_s1_stock10_rn", "idm_s1_whole_tau8_rn"),
                 ("uncond_s5_stock2_rn", "uncond_s5_stock10_rn"),
                 ("uncond_s5_stock2_rn", "uncond_s5_whole_tau8_rn"),
                 ("uncond_s5_stock10_rn", "uncond_s5_whole_tau8_rn"),
                 ("uncond_s5_stock10_rn", "uncond_s5_whole_tau0_rn"),
                 ("idm_s1_stock10_rn", "idm_s1_p4_tau8_rn"), ("idm_s1_stock10_rn", "idm_s1_p6_tau8_rn"),
                 ("idm_s1_p4_tau8_rn", "idm_s5_whole_tau8_rn"), ("idm_s1_p6_tau8_rn", "idm_s1_whole_tau8_rn")]:
        a, b = arm(NEW, x), arm(NEW, y)
        if a is None or b is None:
            continue
        c = paired(a, b, f"{x} vs {y}")
        fresh_c.append(c)
        print(f"   {c['label']:50s} {c['diff']:+5.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] "
              f"{c['a_only']}/{c['b_only']} p={c['p']:.4f}")
    fresh_sweep = []
    for ck, s_, p_, name in [("idm", 1.0, 4, "idm_s1_p4_tau8_rn"), ("idm", 1.0, 6, "idm_s1_p6_tau8_rn"),
                             ("idm", 1.0, 8, "idm_s1_whole_tau8_rn"), ("idm", 5.0, 8, "idm_s5_whole_tau8_rn"),
                             ("native", 5.0, 8, "uncond_s5_whole_tau8_rn"),
                             ("idm", 1.0, 7, "idm_s1_p7_tau8_rn"), ("idm", 3.0, 8, "idm_s3_p8_tau8_rn"),
                             ("idm", 3.0, 9, "idm_s3_p9_tau8_rn")]:
        a = arm(NEW, name)
        if a is None:
            continue
        u = 1 - p_ / 10
        fresh_sweep.append({"ckpt": ck, "shift": s_, "prefix": p_,
                            "fraction": round(1 - s_ * u / (1 + (s_ - 1) * u), 4),
                            "success": rate(a), "source": a["path"]})
        print(f"   fresh noise: {ck:6s} s={s_:.0f} steps 0-{p_ - 1} frac={fresh_sweep[-1]['fraction']:.3f} "
              f"{rate(a)['k']}/{rate(a)['n']}")
    S["noise"] = {"fixed_vs_fresh": nz, "contrasts_under_fresh_noise": fresh_c,
                  "fresh_noise_share_sweep": fresh_sweep}

    # ---------- 5. LIBERO-Long (50 states) ------------------------------------
    print("\n== 5. LIBERO-Long (libero_10), 50 states ==")
    lg = {}
    for pre in ("uncond", "idm"):
        arms = {k: arm(NEW, f"long_{pre}_{k}") for k in ("stock2", "stock10", "whole_tau8")}
        if any(v is None for v in arms.values()):
            continue
        lg[pre] = {k: rate(v) for k, v in arms.items()}
        lg[pre]["contrasts"] = [paired(arms["stock2"], arms["stock10"], f"{pre}: 2 vs 10 steps"),
                                paired(arms["stock2"], arms["whole_tau8"], f"{pre}: 2 steps vs tau=8 cache"),
                                paired(arms["stock10"], arms["whole_tau8"], f"{pre}: 10 steps vs tau=8 cache")]
        print(f"   {pre}: " + ", ".join(f"{k} {lg[pre][k]['pct']}%" for k in ("stock2", "stock10", "whole_tau8")))
        for c in lg[pre]["contrasts"]:
            print(f"      {c['label']:34s} {c['diff']:+5.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] p={c['p']:.4f}")
    S["libero_long"] = lg

    # ---------- 6. latency on the frontier host -------------------------------
    print("\n== 6. Latency on the frontier host ==")
    L = {}
    names = [n for n in (list(front["idm"]["arms"].values()) + list(front["native"]["arms"].values()))] + [
        "lat_uncond_stock2_cg", "lat_uncond_stock1_cg", "lat_uncond_stock4_cg", "lat_uncond_stock10_cg",
        "lat_idm_block_skip_tau8", "lat_idm_block_plain_tau8", "lat_uncond_stock10_cg_r2",
        "lat_uncond_stock2_cg_r2", "lat_uncond_whole_tau8_r2", "lat_uncond_whole_4step_tau4_r2",
        "lat_uncond_stock10_r2", "lat_uncond_stock2_r2"]
    for n in names:
        a = arm(NEW, n)
        if a is None:
            continue
        L[n] = latency(a)
        L[n]["host"] = host_of(a)
        x = L[n]
        print(f"   {n:32s} ep {x.get('ep_mean', float('nan')):6.1f}±{x.get('ep_sd', float('nan')):4.1f}  "
              f"median {x.get('cyc_median')}  p95 {x.get('cyc_p95')}  refresh {x.get('refresh_mean')}  "
              f"reuse {x.get('reuse_mean')}  [{x['host']['host_tag']}]")
    S["latency"] = L
    ld = []
    for x, y in [("lat_uncond_stock1_cg", "uncond_stock_1step"), ("lat_uncond_stock2_cg", "uncond_stock_2step"),
                 ("lat_uncond_stock4_cg", "uncond_stock_4step"), ("lat_uncond_stock10_cg", "uncond_stock_10step"),
                 ("uncond_stock_2step", "uncond_stock_10step"), ("lat_uncond_stock2_cg", "lat_uncond_stock10_cg"),
                 ("uncond_stock_2step", "uncond_whole_4step_tau4"),
                 ("lat_uncond_stock2_cg", "uncond_whole_4step_tau4"),
                 ("uncond_stock_4step", "uncond_whole_4step_tau4"),
                 ("uncond_whole_tau8", "uncond_stock_10step"),
                 ("idm_whole_agg_tau8", "lat_idm_block_skip_tau8"),
                 ("idm_whole_plain_tau8", "lat_idm_block_plain_tau8"),
                 ("lat_uncond_stock2_r2", "uncond_stock_2step"), ("lat_uncond_stock10_r2", "uncond_stock_10step"),
                 ("lat_uncond_stock2_cg_r2", "lat_uncond_stock2_cg"),
                 ("lat_uncond_stock10_cg_r2", "lat_uncond_stock10_cg"),
                 ("lat_uncond_whole_4step_tau4_r2", "uncond_whole_4step_tau4"),
                 ("lat_uncond_whole_tau8_r2", "uncond_whole_tau8"),
                 ("idm_whole_plain_tau8", "uncond_whole_tau8")]:
        a, b = arm(NEW, x), arm(NEW, y)
        if a is None or b is None:
            continue
        ld.append(lat_diff(a, b, f"{x} - {y}"))
        c = ld[-1]
        print(f"   {c['label']:62s} {c['diff_ms']:+7.1f} ms [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] ratio {c['ratio']}")
    S["latency_contrasts"] = ld

    # Cost model: latency = F + K * a over the uncached schedules (least squares).
    # F is everything a call does besides denoising (image encoding, the video
    # expert's prefill); a is one pass of the action expert. A C3ache-style reuse
    # call that keeps m fresh steps costs at least F + m * a, and a cache over the
    # first of two steps can save at most (tau - 1) / tau of one pass per call.
    print("\n   cost model F + K*a over uncached schedules (frontier host):")
    lm = {}
    steps = (1, 2, 4, 10)
    for ck, block, key in (("idm", "idm", "latency"), ("native", "native", "latency"),
                           ("native_cudagraphs", "native", "latency_cudagraphs")):
        arms = S["frontier"][block]["arms"]
        ks, ts = [], []
        for k in steps:
            a = arms.get("uncached 1 step" if k == 1 else f"uncached {k} steps")
            if a and a.get(key):
                ks.append(k)
                ts.append(a[key]["ep_mean"])
        if len(ks) < 2:
            continue
        a_step, fixed = np.polyfit(np.array(ks, dtype=float), np.array(ts, dtype=float), 1)
        lm[ck] = {"steps": ks, "ms": ts, "fixed_ms": round(float(fixed), 1),
                  "per_step_ms": round(float(a_step), 2),
                  "fixed_in_step_costs": round(float(fixed / a_step), 2),
                  "share_outside_denoising_at_2_steps": round(float(fixed / (fixed + 2 * a_step)), 3),
                  "composed_tau4_max_saving": round(float(0.75 * a_step / (fixed + 2 * a_step)), 3)}
        print(f"   {ck:18s} F={fixed:6.1f} ms  a={a_step:5.2f} ms/step  F/a={fixed / a_step:4.2f}  "
              f"outside denoising at K=2: {fixed / (fixed + 2 * a_step):.2f}")
    # C3ache's headline: 2.51x at tau=0 with 8 of 10 steps cached, i.e.
    # (F + 10a) / (F + 2a) = 2.51 if reuse costs nothing beyond the fresh steps.
    lm["c3ache_implied_fixed_in_step_costs"] = round((10 - 2 * 2.51) / (2.51 - 1), 2)
    S["latency_model"] = lm

    # ---------- identity gates run on each new host ---------------------------
    gates = []
    for g in sorted(NEW.rglob("gate*.json")):
        if REPEATS in g.parents:
            continue
        d = json.loads(g.read_text())
        rt = d.get("runtime", {})
        gates.append({"path": rel(g), "ckpt": d.get("ckpt"), "pass": d.get("pass"),
                      "shift": rt.get("effective_action_sigma_shift"), "gpu": d.get("gpu"),
                      "identity_err": d.get("identity_err"), "whole_identity_err": d.get("whole_identity_err"),
                      "stale_err": d.get("stale_err"), "whole_stale_err": d.get("whole_stale_err"),
                      "whole_cache_bytes": d.get("whole_cache_bytes")})
        print(f"   gate {gates[-1]['path']}: {d.get('ckpt')} s={rt.get('effective_action_sigma_shift')} "
              f"pass={d.get('pass')} whole identity {d.get('whole_identity_err'):.2e} stale {d.get('whole_stale_err'):.2e}")
    S["gates"] = gates

    # ---------- 7. earlier fresh-process collections, for the paper ------------
    S["earlier"] = {}
    ss = {}
    for seed, suffix in ((42, ""), (7, "_seed7")):
        a2, a10 = load(R / "statescale" / f"s2{suffix}.json"), load(R / "statescale" / f"s10{suffix}.json")
        if a2 and a10:
            ss[seed] = paired(a2, a10, f"500 states seed {seed}: 2 steps vs 10-step blockwise attenuated tau=4")
            ss[seed]["latency_2"] = latency(a2)
            ss[seed]["latency_10cache"] = latency(a10)
    full = load(R / "matrix" / "m10full.json")
    two = load(R / "statescale" / "s2.json")
    if full and two:
        ss["full_vs_2"] = paired(two, full, "500 states seed 42: 2 steps vs 10-step full compute (separate hosts)")
        # The 100 frontier states are a subset of these 500 and use the same fixed
        # noise, so the current runs should reproduce the earlier ones state by state.
        repro = []
        for old, new in ((two, arm(NEW, "idm_stock_2step")), (full, arm(NEW, "idm_stock_10step"))):
            if new is not None:
                repro.append(determinism(new, old, f"{new['path']} vs {old['path']}"))
        S["earlier"]["statescale_reproduced"] = repro
    S["earlier"]["statescale"] = ss
    fr = {}
    for seed in (42, 7):
        p, a = load(R / "tau8_fresh" / f"seed{seed}_plain.json"), load(R / "tau8_fresh" / f"seed{seed}_attenuated.json")
        if p and a:
            fr[seed] = paired(a, p, f"seed {seed}: attenuated vs plain blockwise tau=8")
    S["earlier"]["attenuation_fresh"] = fr
    ha = {}
    for x, y in [("idm_whole_agg_tau8", "idm_whole_plain_tau8"), ("idm_whole_agg_tau4", "idm_whole_plain_tau4"),
                 ("idm_whole_agg_tau8", "idm_block_skip_tau8"), ("idm_block_skip_tau8", "idm_block_plain_tau8")]:
        a, b = arm(HOST_A, x), arm(HOST_A, y)
        if a and b:
            ha[f"{x} vs {y}"] = paired(a, b, f"host A: {x} vs {y}")
            ha[f"{x} vs {y}"]["latency"] = lat_diff(a, b, f"host A: {x} - {y}")
    S["earlier"]["host_a"] = ha

    for k, v in ss.items():
        print(f"\n   {v['label']}: {v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] p={v['p']:.4f}")
    for k, v in fr.items():
        print(f"   {v['label']}: {v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] p={v['p']:.4f}")

    # ---------- 8. composed 2-step schedule (K=2, prefix=1) ---------------------
    COMPOSED = R / "composed2step"
    comp = {}
    if COMPOSED.is_dir():
        print("\n== 8. Composed 2-step schedule (step 0 cached, prefix=1) ==")
        for name in ["idm_whole_2step_p1_tau2", "idm_whole_2step_p1_tau4",
                     "uncond_whole_2step_p1_tau2", "uncond_whole_2step_p1_tau4"]:
            a = load(COMPOSED / f"{name}.json")
            if a is None:
                continue
            r = rate(a)
            L = latency(a)
            L["host"] = host_of(a)
            ck = "idm" if "idm" in name else "native"
            u2_name = "idm_stock_2step" if ck == "idm" else "uncond_stock_2step"
            u2 = arm(NEW, u2_name)
            p_res = paired(a, u2, f"{name} vs {u2_name}") if u2 else None
            comp[name] = {
                "rate": r, "latency": L,
                "vs_uncached_2step": p_res
            }
            p_val = p_res['p'] if p_res else float('nan')
            print(f"   {name:30s} {r['k']:3d}/{r['n']} {r['pct']:5.1f}%  {L.get('ep_mean', float('nan')):6.1f} ms "
                  f"(refresh {L.get('refresh_mean')}, reuse {L.get('reuse_mean')}) vs 2step p={p_val:.4f}")
        S["composed2step"] = comp

    # ---------- 9. matched same-host composed & uncached baselines --------------
    MATCHED = R / "matched_composed"
    if MATCHED.is_dir():
        print("\n== 9. Matched same-host uncached & composed 2-step caching ==")
        mc = {"arms": {}, "contrasts": {}}
        gate_f = MATCHED / "gate.json"
        if gate_f.exists():
            mc["gate"] = json.loads(gate_f.read_text())
        m_arms = {}
        for name in [
            "idm_stock_1step", "idm_stock_2step",
            "idm_whole_2step_p1_tau2", "idm_whole_2step_p1_tau4",
            "uncond_stock_1step", "uncond_stock_2step",
            "uncond_whole_2step_p1_tau2", "uncond_whole_2step_p1_tau4",
        ]:
            a = load(MATCHED / f"{name}.json")
            if a is None:
                continue
            r = rate(a)
            L = latency(a)
            L["host"] = host_of(a)
            m_arms[name] = a
            mc["arms"][name] = {
                "rate": r, "latency": L,
                "wilson95": wilson(r["k"], r["n"]),
            }
            print(f"   {name:30s} {r['k']:3d}/{r['n']} {r['pct']:5.1f}%  {L.get('ep_mean', float('nan')):6.1f} ms "
                  f"(refresh {L.get('refresh_mean')}, reuse {L.get('reuse_mean')})")

        contrast_pairs = [
            ("idm_tau4_vs_2step", "idm_whole_2step_p1_tau4", "idm_stock_2step"),
            ("idm_tau4_vs_1step", "idm_whole_2step_p1_tau4", "idm_stock_1step"),
            ("idm_tau2_vs_2step", "idm_whole_2step_p1_tau2", "idm_stock_2step"),
            ("native_tau4_vs_2step", "uncond_whole_2step_p1_tau4", "uncond_stock_2step"),
            ("native_tau4_vs_1step", "uncond_whole_2step_p1_tau4", "uncond_stock_1step"),
            ("native_tau2_vs_2step", "uncond_whole_2step_p1_tau2", "uncond_stock_2step"),
        ]
        for cname, a1_name, a2_name in contrast_pairs:
            if a1_name in m_arms and a2_name in m_arms:
                p_res = paired(m_arms[a1_name], m_arms[a2_name], cname)
                lat1 = mc["arms"][a1_name]["latency"].get("ep_mean", 0)
                lat2 = mc["arms"][a2_name]["latency"].get("ep_mean", 0)
                speedup = round(lat2 / lat1, 2) if lat1 > 0 else float("nan")
                p_res["speedup"] = speedup
                mc["contrasts"][cname] = p_res
                print(f"   Contrast {cname:24s}: {p_res['diff']:+.1f} pts [{p_res['ci95'][0]:+.1f}, {p_res['ci95'][1]:+.1f}] p={p_res['p']:.4f}, speedup {speedup}x")
        S["matched_composed"] = mc

    # ---------- 10. follow-up collection (repro/experiments/followup.sh) -------
    FU = NEW / "followup_a6000"
    if FU.is_dir():
        print("\n== 10. Follow-up: interleaved timing, share threshold, camera jitter ==")
        fu = {}
        # Interleaved timing rounds. Latency only: the success of these
        # configurations comes from the 100-state runs. Ratios are taken within a
        # round, so drift in the host between rounds cancels.
        rounds = {}
        for r in (1, 2, 3):
            row = {}
            for k in ("stock1", "stock2", "comp4", "stock10", "tau8"):
                a = arm(NEW, f"tm{r}_idm_{k}")
                if a is not None:
                    row[k] = latency(a)
                    row[k]["host"] = host_of(a)
            if row:
                rounds[str(r)] = row
        ratios = {}
        for num, den in (("comp4", "stock2"), ("stock1", "stock2"), ("comp4", "stock1"),
                         ("stock10", "stock2"), ("tau8", "stock10")):
            vals = [rows_[num]["ep_mean"] / rows_[den]["ep_mean"] for rows_ in rounds.values()
                    if num in rows_ and den in rows_]
            if vals:
                ratios[f"{num}/{den}"] = {"per_round": [round(v, 3) for v in vals],
                                          "mean": round(float(np.mean(vals)), 3),
                                          "min": round(min(vals), 3), "max": round(max(vals), 3)}
                print(f"   {num}/{den}: " + ", ".join(f"{v:.3f}" for v in vals))
        comp_cycles = {}
        for rows_ in rounds.values():
            if "comp4" in rows_:
                for key in ("refresh_mean", "reuse_mean"):
                    comp_cycles.setdefault(key, []).append(rows_["comp4"][key])
        fu["timing"] = {"rounds": rounds, "ratios": ratios,
                        "comp4_refresh_reuse_ms": {k: [round(v, 1) for v in vs] for k, vs in comp_cycles.items()}}
        # Reuse against refresh calls of the composed cache WITHIN a run: the two
        # kinds of call alternate every few calls, so the host's load hits both
        # alike. This is the per-call saving of skipping the first step, on every
        # host that ran the composed cache.
        rr = []
        for label, d, name in (("quiet A6000", R / "composed2step", "idm_whole_2step_p1_tau4"),
                               ("quiet A6000", R / "composed2step", "uncond_whole_2step_p1_tau4"),
                               ("shared A6000", R / "matched_composed", "idm_whole_2step_p1_tau4"),
                               ("shared A6000", R / "matched_composed", "uncond_whole_2step_p1_tau4"),
                               ("follow-up A6000", NEW, "tm1_idm_comp4"), ("follow-up A6000", NEW, "tm2_idm_comp4"),
                               ("follow-up A6000", NEW, "tm3_idm_comp4")):
            a = arm(d, name)
            if a is None:
                continue
            L_ = latency(a)
            ratio = L_["reuse_mean"] / L_["refresh_mean"]
            rr.append({"host": label, "host_tag": host_of(a)["host_tag"], "arm": name,
                       "refresh_ms": L_["refresh_mean"], "reuse_ms": L_["reuse_mean"],
                       "reuse_over_refresh": round(ratio, 3),
                       "tau4_mean_saving": round(1 - (1 + 3 * ratio) / 4, 3)})
            print(f"   {label:16s} {name:28s} reuse/refresh {ratio:.3f}  tau=4 saving {rr[-1]['tau4_mean_saving']:.3f}")
        fu["composed_reuse_over_refresh"] = rr
        # Per-host cost of a call. The frontier host's fit is in latency_model; on
        # the follow-up host the 1-step arm of round 1 switched load regime half
        # way (bimodal episode means), so its fixed and per-step costs come from
        # the 2- and 10-step arms of that round alone.
        r1 = rounds.get("1", {})
        if "stock2" in r1 and "stock10" in r1:
            t2, t10 = r1["stock2"]["ep_mean"], r1["stock10"]["ep_mean"]
            a_ = (t10 - t2) / 8.0
            fu["cost_two_point"] = {"host_tag": r1["stock2"]["host"]["host_tag"], "ms_2": t2, "ms_10": t10,
                                    "per_step_ms": round(a_, 2), "fixed_ms": round(t2 - 2 * a_, 1),
                                    "fixed_in_step_costs": round((t2 - 2 * a_) / a_, 2),
                                    "ten_over_two": round(t10 / t2, 3)}
        # How much slower the same uncached configurations ran on the other hosts
        # than on the frontier host (latency_across_hosts covers one more host).
        xr = []
        for d, host_label, pairs in ((R / "matched_composed", "shared A6000",
                                      (("idm_stock_1step", "idm_stock_1step"), ("idm_stock_2step", "idm_stock_2step"),
                                       ("uncond_stock_1step", "uncond_stock_1step"), ("uncond_stock_2step", "uncond_stock_2step"))),
                                     (NEW, "follow-up A6000",
                                      (("tm1_idm_stock1", "idm_stock_1step"), ("tm1_idm_stock2", "idm_stock_2step"),
                                       ("tm1_idm_stock10", "idm_stock_10step")))):
            for other, ref in pairs:
                a, b = arm(d, other), arm(NEW, ref)
                if a is None or b is None:
                    continue
                la, lb = latency(a)["ep_mean"], latency(b)["ep_mean"]
                xr.append({"host": host_label, "host_tag": host_of(a)["host_tag"], "arm": other,
                           "frontier_arm": ref, "ms": la, "frontier_ms": lb, "ratio": round(la / lb, 3)})
        fu["latency_vs_frontier_host"] = xr
        print("   latency vs frontier host: " + ", ".join(f"{x['arm']} x{x['ratio']}" for x in xr))
        # The timing arms replay the first three initial states of every task, which
        # the frontier and the composed-cache runs covered on other A6000s, so they
        # also test whether this host reproduces outcomes.
        det = []
        refs = {"stock1": (NEW, "idm_stock_1step"), "stock2": (NEW, "idm_stock_2step"),
                "stock10": (NEW, "idm_stock_10step"), "tau8": (NEW, "idm_whole_plain_tau8"),
                "comp4": (R / "composed2step", "idm_whole_2step_p1_tau4")}
        for r in (1, 2, 3):
            for k, (d, ref) in refs.items():
                a = arm(NEW, f"tm{r}_idm_{k}")
                b = arm(d, ref)
                if a is not None and b is not None:
                    det.append(determinism(a, b, f"tm{r}_idm_{k} vs {ref}"))
        fu["determinism"] = det
        fu["determinism_total"] = {"comparisons": len(det), "paired_episodes": sum(x["n"] for x in det),
                                   "outcome_changes": sum(x["success_flips"] for x in det),
                                   "step_count_changes": sum(x["step_count_differs"] for x in det)}
        print(f"   determinism vs earlier hosts: {fu['determinism_total']}")
        # Everything together: the budget_shift re-runs plus this host's replays.
        hosts = set(S["determinism_total"]["hosts"])
        commits = set()
        for x in S["determinism"] + det:
            hosts.update(h for h in (x.get("host_a"), x.get("host_b")) if h)
        for d_ in (NEW, R / "matched100_a6000", R / "matched50_a6000", R / "composed2step"):
            for p_ in sorted(d_.rglob("*.json")):
                try:
                    j_ = json.loads(p_.read_text())
                except Exception:
                    continue
                if isinstance(j_, dict) and "meta" in j_ and (j_["meta"].get("provenance") or {}).get("host_tag") in hosts:
                    c_ = (j_["meta"].get("provenance") or {}).get("commit")
                    if c_:
                        commits.add(c_[:12])
        fu["determinism_all"] = {
            "comparisons": S["determinism_total"]["comparisons"] + fu["determinism_total"]["comparisons"],
            "paired_episodes": S["determinism_total"]["paired_episodes"] + fu["determinism_total"]["paired_episodes"],
            "outcome_changes": S["determinism_total"]["outcome_changes"] + fu["determinism_total"]["outcome_changes"],
            "step_count_changes": S["determinism_total"]["step_count_changes"] + fu["determinism_total"]["step_count_changes"],
            "hosts": sorted(hosts), "harness_revisions": sorted(commits)}
        print(f"   determinism, all hosts: {fu['determinism_all']}")
        # Cached share between 64% and 80%, and the third shift.
        share = []
        for x, y in (("idm_s1_p7_tau8", "idm_s1_stock10"), ("uncond_s1_p7_tau8", "uncond_s1_stock10"),
                     ("idm_s3_p8_tau8", "idm_s3_stock10"), ("idm_s3_p9_tau8", "idm_s3_stock10"),
                     ("idm_s3_stock10", "idm_s1_stock10"), ("idm_s3_p9_tau8", "idm_s1_p7_tau8"),
                     ("idm_s3_p8_tau8", "idm_s1_p6_tau8"),
                     # the same eight cached steps at s=3 and at s=1 (57% vs 80% of the path)
                     ("idm_s3_p8_tau8", "idm_s1_whole_tau8")):
            a, b = arm(NEW, x), arm(NEW, y)
            if a is None or b is None:
                continue
            c = paired(a, b, f"{x} vs {y}")
            share.append(c)
            print(f"   {c['label']:40s} {c['a_pct']:5.1f} vs {c['b_pct']:5.1f}  {c['diff']:+5.1f} "
                  f"[{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] {c['a_only']}/{c['b_only']} p={c['p']:.4f}")
        fu["share_contrasts"] = share
        # Camera jitter: a pilot to choose the magnitude, then the comparison.
        fu["camera_pilot"] = {p_.stem: rate(load(p_)) for p_ in sorted(FU.glob("campilot_*.json"))
                              if load(p_) is not None}
        cam = {}
        for p_ in sorted(FU.glob("cam*_idm_stock10.json")):
            if p_.name.startswith("campilot"):
                continue
            tag = p_.name.split("_")[0]
            arms_ = {k: arm(NEW, f"{tag}_{k}") for k in ("idm_stock10", "idm_stock2", "idm_stock1", "idm_tau8",
                                                        "uncond_stock10", "uncond_stock2")}
            entry = {"perturb": arms_["idm_stock10"]["meta"].get("perturb"),
                     "rates": {k: rate(v) for k, v in arms_.items() if v is not None}, "contrasts": []}
            for x, y in (("idm_stock2", "idm_stock10"), ("idm_stock1", "idm_stock10"),
                         ("idm_tau8", "idm_stock10"), ("idm_stock2", "idm_tau8"),
                         ("uncond_stock2", "uncond_stock10")):
                if arms_.get(x) is not None and arms_.get(y) is not None:
                    entry["contrasts"].append(paired(arms_[x], arms_[y], f"{tag}: {x} vs {y}"))
            cam[tag] = entry
            print(f"   {tag} ({entry['perturb']}): " + ", ".join(f"{k} {v['pct']}%" for k, v in entry["rates"].items()))
            for c in entry["contrasts"]:
                print(f"      {c['label']:40s} {c['diff']:+5.1f} tango90 {c['tango90']} p={c['p']:.4f}")
        fu["camera"] = cam
        S["followup"] = fu

    # ---------- 12. every same-configuration pair on different machines -------
    # Sections 2 and 10 compare the re-runs the collection was designed around.
    # This compares every pair of single-configuration runs that share a
    # configuration and not a machine, across all the collections read above.
    ex = exhaustive_determinism()
    S["determinism_exhaustive"] = ex
    print(f"\n== 12. Every same-configuration pair across machines: {ex['pairs']} pairs, "
          f"{ex['paired_episodes']} paired episodes, {ex['outcome_changes']} outcome changes "
          f"on {ex['states_changed']} initial states, {ex['step_count_changes']} length changes, "
          f"{len(ex['hosts'])} hosts, {len(ex['harness_revisions'])} revisions")
    for c in ex["changes"]:
        print(f"   {c['a']} vs {c['b']}: task {c['task']} state {c['init_state']} "
              f"{c['success_a']}/{c['success_b']} after {c['steps_a']}/{c['steps_b']} steps")

    # ---------- 13. idle-host follow-up (repro/experiments/idle_host.sh) ------
    # RTX A6000s in one machine rented whole. The composed cache and two uncached
    # steps alternate in interleaved rounds that ran alone on the machine, so the
    # ratio is taken within a round and drift between rounds cancels; the
    # interval is a t interval over rounds. The camera-jitter arms ran before
    # them, side by side on four GPUs, and measure success only.
    IDLE = NEW / "idle_a6000"
    if IDLE.is_dir():
        print("\n== 13. Idle host: composed-cache timing and camera-jitter non-inferiority ==")
        idle = {}
        rounds = []
        for r_ in range(1, 50):
            a, b = arm(NEW, f"it{r_}_idm_stock2"), arm(NEW, f"it{r_}_idm_comp4")
            if a is None or b is None:
                continue
            la, lb = latency(a), latency(b)
            rounds.append({"round": r_, "stock2_ms": la["ep_mean"], "comp4_ms": lb["ep_mean"],
                           "ratio": round(lb["ep_mean"] / la["ep_mean"], 4),
                           "reuse_ms": lb.get("reuse_mean"), "refresh_ms": lb.get("refresh_mean"),
                           "reuse_over_refresh": round(lb["reuse_mean"] / lb["refresh_mean"], 4),
                           "loadavg_1min": [host_of(x)["loadavg"][0] if host_of(x)["loadavg"] else None
                                            for x in (a, b)],
                           "host_tag": host_of(a)["host_tag"], "cpu_count": host_of(a)["cpu_count"]})
            print(f"   round {r_}: two steps {la['ep_mean']:.1f} ms, composed {lb['ep_mean']:.1f} ms, "
                  f"ratio {rounds[-1]['ratio']:.3f} (reuse/refresh {rounds[-1]['reuse_over_refresh']:.3f}, "
                  f"1-min load {rounds[-1]['loadavg_1min']})")
        if len(rounds) >= 2:
            ratios = [x["ratio"] for x in rounds]
            m_ = float(np.mean(ratios))
            sd_ = float(np.std(ratios, ddof=1))
            t_ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                  10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}.get(len(ratios) - 1, 1.96)
            half = t_ * sd_ / math.sqrt(len(ratios))
            rr_ = [x["reuse_over_refresh"] for x in rounds]
            loads = [v for x in rounds for v in x["loadavg_1min"] if v is not None]
            idle["timing"] = {
                "rounds": rounds, "n_rounds": len(rounds),
                "states_per_arm": latency(arm(NEW, f"it{rounds[0]['round']}_idm_stock2"))["episodes"],
                "ratio_mean": round(m_, 4), "ratio_ci95": [round(m_ - half, 4), round(m_ + half, 4)],
                "saving_pct": round(100 * (1 - m_), 1),
                "saving_ci95": [round(100 * (1 - (m_ + half)), 1), round(100 * (1 - (m_ - half)), 1)],
                "reuse_over_refresh": [round(min(rr_), 3), round(max(rr_), 3)],
                "reuse_saving_at_tau4_pct": [round(100 * 0.75 * (1 - max(rr_)), 1), round(100 * 0.75 * (1 - min(rr_)), 1)],
                "max_loadavg_1min": round(max(loads), 2) if loads else None,
                "stock2_ms": [round(min(x["stock2_ms"] for x in rounds), 1), round(max(x["stock2_ms"] for x in rounds), 1)],
                "comp4_ms": [round(min(x["comp4_ms"] for x in rounds), 1), round(max(x["comp4_ms"] for x in rounds), 1)],
                "stock2_ms_mean": round(float(np.mean([x["stock2_ms"] for x in rounds])), 1),
                "comp4_ms_mean": round(float(np.mean([x["comp4_ms"] for x in rounds])), 1)}
            print(f"   composed over two steps: {m_:.3f} [{m_ - half:.3f}, {m_ + half:.3f}] over {len(rounds)} rounds, "
                  f"saving {100 * (1 - m_):.1f}%; max 1-min load at the end of an arm {idle['timing']['max_loadavg_1min']}")
        jit = {}
        for ck in ("idm", "uncond"):
            a, b = arm(NEW, f"cam0015n250_{ck}_stock2"), arm(NEW, f"cam0015n250_{ck}_stock10")
            if a is None or b is None:
                continue
            c_ = paired(a, b, f"cam0015n250: {ck} 2 vs 10 steps")
            # The first five states per task repeat the 50-state runs of followup_a6000.
            rep = []
            for x_, k_ in ((a, "stock2"), (b, "stock10")):
                y_ = arm(NEW, f"cam0015_{ck}_{k_}")
                if y_ is not None:
                    rep.append(determinism(x_, y_, f"cam0015n250_{ck}_{k_} vs cam0015_{ck}_{k_}"))
            jit[ck] = {"stock2": rate(a), "stock10": rate(b), "contrast": c_,
                       "noninferior_within_5pts": bool(c_["tango90"][0] > -EQ_MARGIN),
                       "reproduces_50": rep,
                       "reproduced_outcome_changes": sum(x["success_flips"] for x in rep),
                       "reproduced_episodes": sum(x["n"] for x in rep)}
            print(f"   camera jitter, {ck}: two steps {rate(a)['pct']}% vs ten {rate(b)['pct']}% on {c_['n']} states, "
                  f"Tango 90% {c_['tango90']}, non-inferior within 5: {jit[ck]['noninferior_within_5pts']}; "
                  f"first 50 states repeat the 50-state runs with {jit[ck]['reproduced_outcome_changes']} changes "
                  f"in {jit[ck]['reproduced_episodes']}")
        idle["jitter"] = jit
        S["idle_host"] = idle

    # ---------- 14. held-out shift and a second noise stream (repro/experiments/heldout.sh)
    # The predictions P1-P3 were committed and pushed before these arms ran, in
    # repro/experiments/heldout.sh; they are tested here exactly as stated there.
    HO = NEW / "heldout_a6000"
    if HO.is_dir():
        print("\n== 14. Held-out shift (s = 2) and a second noise stream (seed 7) ==")
        ho = {}
        st_, p7_, p9_ = (arm(NEW, n) for n in ("idm_s2_stock10", "idm_s2_p7_tau8", "idm_s2_p9_tau8"))
        if st_ is not None and p7_ is not None and p9_ is not None:
            def _sigma(m, s_, k=10):                   # Eq. 1
                u = 1 - m / k
                return s_ * u / (1 + (s_ - 1) * u)
            c7 = paired(p7_, st_, "s=2: first 7 of 10 steps cached vs uncached")
            c9 = paired(p9_, st_, "s=2: first 9 of 10 steps cached vs uncached")
            ho["shift"] = {
                "shift": 2.0, "stock10": rate(st_), "p7": rate(p7_), "p9": rate(p9_),
                "sigma_p7": round(_sigma(7, 2.0), 4), "share_p7": round(1 - _sigma(7, 2.0), 4),
                "sigma_p9": round(_sigma(9, 2.0), 4), "share_p9": round(1 - _sigma(9, 2.0), 4),
                "contrast_p7": c7, "contrast_p9": c9,
                "P1_within_4_points": bool(abs(c7["diff"]) <= 4),
                "P2_loss_at_least_20": bool(c9["diff"] <= -20)}
            for lab, c_, ok in (("P1 first 7 cached (53.8%)", c7, ho["shift"]["P1_within_4_points"]),
                                ("P2 first 9 cached (81.8%)", c9, ho["shift"]["P2_loss_at_least_20"])):
                print(f"   {lab}: {c_['a_pct']}% vs {c_['b_pct']}% uncached, diff {c_['diff']:+.1f}, "
                      f"Tango 90% {c_['tango90']}, McNemar p={c_['p']:.3g}; prediction met: {ok}")
        f7, r7 = arm(NEW, "idm_s1_p6_tau8_seed7"), arm(NEW, "idm_s1_p6_tau8_rn_seed7")
        if f7 is not None and r7 is not None:
            c_7 = paired(r7, f7, "seed 7: 60% cache, fresh vs fixed noise")
            strm = {"seed7": {"fixed": rate(f7), "fresh": rate(r7), "contrast": c_7},
                    "P3_loss_at_least_5": bool(c_7["diff"] <= -5)}
            print(f"   P3 seed 7: fresh {c_7['a_pct']}% vs fixed {c_7['b_pct']}%, diff {c_7['diff']:+.1f}, "
                  f"McNemar p={c_7['p']:.3g}; prediction met: {strm['P3_loss_at_least_5']}")
            f42, r42 = arm(NEW, "idm_s1_p6_tau8"), arm(NEW, "idm_s1_p6_tau8_rn")
            if f42 is not None and r42 is not None:
                c_42 = paired(r42, f42, "seed 42: 60% cache, fresh vs fixed noise")
                b01, b10 = c_7["a_only"] + c_42["a_only"], c_7["b_only"] + c_42["b_only"]
                n_ = c_7["n"] + c_42["n"]
                strm["seed42"] = {"fixed": rate(f42), "fresh": rate(r42), "contrast": c_42}
                strm["pooled"] = {"n": n_, "fresh_only": b01, "fixed_only": b10,
                                  "diff": round(100 * (b01 - b10) / n_, 1), "p": mcnemar_exact(b01, b10)}
                # Both seeds visit the same 50 initial states, so the state is the
                # sampling unit: sum each state's fresh-minus-fixed differences over
                # the two seeds and flip that sum's sign as one, exactly over all
                # sign patterns (a paired permutation test clustered by state).
                dsum = {}
                for fr_, fx_ in ((r7, f7), (r42, f42)):
                    for k_ in set(fr_["outcome"]) & set(fx_["outcome"]):
                        dsum[k_] = dsum.get(k_, 0) + int(fr_["outcome"][k_]) - int(fx_["outcome"][k_])
                vals = [abs(v) for v in dsum.values() if v]
                dist = {0: 1}
                for v in vals:
                    nd = {}
                    for t_, c_ in dist.items():
                        nd[t_ + v] = nd.get(t_ + v, 0) + c_
                        nd[t_ - v] = nd.get(t_ - v, 0) + c_
                    dist = nd
                tobs = abs(sum(dsum.values()))
                strm["pooled"]["states"] = len(dsum)
                strm["pooled"]["states_changed"] = len(vals)
                strm["pooled"]["p_by_state"] = sum(c_ for t_, c_ in dist.items() if abs(t_) >= tobs) / 2 ** len(vals)
                print(f"   pooled over seeds 42 and 7: {n_} paired states, diff {strm['pooled']['diff']:+.1f}, "
                      f"discordant {b01} vs {b10}, exact McNemar p={strm['pooled']['p']:.3g}; with the "
                      f"{strm['pooled']['states']} initial states as the unit, exact sign-flip "
                      f"p={strm['pooled']['p_by_state']:.3g}")
            ho["streams"] = strm
        S["heldout"] = ho

    # ---------- 15. cache screen and held-out closed-loop test (repro/experiments/screen.sh)
    # The predictions P4-P8 were committed and pushed with probes/report_screen.py
    # before any screening run; report_screen.analyze tests them as stated there.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import report_screen as _RS
    sc_ = _RS.analyze(NEW, load=load)
    if sc_ is not None:
        print("\n== 15. Cache screen on uncached rollouts, and a held-out closed-loop test ==")
        r_ = sc_.get("rho_primary")
        if r_:
            print(f"   {r_['n']} tau = 8 configurations: Spearman rho(D, success) = {r_['D']}, "
                  f"rho(share, success) = {r_['share']}, bootstrap 90% of the difference {r_.get('diff_boot90')}")
        for name_, v_ in sc_["predictions"].items():
            print(f"   {name_}: met = {v_['met']}  {dict((k, v) for k, v in v_.items() if k != 'met')}")
        for v_ in sc_.get("validation", []):
            print(f"   held-out {v_['cell']} p{v_['m']}: predicted {v_['pred_D']}% (D) / {v_['pred_share']}% (share), "
                  f"observed {v_.get('observed', {}).get('pct')}")
        S["screen"] = sc_

    # ---------- 16. episode-latent control and delayed execution (repro/experiments/latent_delay.sh)
    # The predictions P9-P11 were committed and pushed with probes/report_latent_delay.py
    # before any of these runs; report_latent_delay.analyze tests them as stated there.
    import report_latent_delay as _LD
    ld_ = _LD.analyze(NEW, load, paired, rate)
    if ld_ is not None:
        print("\n== 16. Episode-latent control and delayed execution ==")
        for k_, v_ in ld_.get("latent", {}).items():
            print(f"   {k_}: episode latent {v_['episode']['pct']}% vs fixed {v_['fixed']['pct']}% vs fresh {v_['fresh']['pct']}%")
        if "delay" in ld_:
            print("   delay:", {k_: v_["pct"] for k_, v_ in ld_["delay"]["rates"].items()})
        for name_, v_ in ld_["predictions"].items():
            print(f"   {name_}: met = {v_['met']}  {dict((k, v) for k, v in v_.items() if k != 'met')}")
        S["latent_delay"] = ld_
    # delay_graphs.sh: the delays of CUDA-graph execution, committed with this
    # analysis before its runs and reported without a prediction
    gr_ = _LD.analyze_graphs(NEW, load, paired, rate)
    if gr_:
        print("\n== 16b. Delays of CUDA-graph execution ==")
        for ck_, v_ in gr_.items():
            c_ = v_["two_d2_vs_ten_d3"]
            print(f"   {ck_}: two steps at delay 2 {c_['a_pct']}% vs ten at delay 3 {c_['b_pct']}% "
                  f"(diff {c_['diff']:+.0f}, Tango 90% {c_['tango90']}, p={c_['p']:.3f})")
        S["delay_graphs"] = gr_

    # ---------- 11. process-sharing pilot (results/rngdrift, repro/experiments/rngdrift.sh)
    # The same configurations run in one process, once in a three-arm and once in
    # a five-arm list, on 2 tasks x 10 episodes. With the per-episode reseed on
    # (the current protocol), any difference comes from what else ran in the
    # process. probes/report_rngdrift.py prints the full comparison.
    RD = R / "rngdrift"
    ps = {}
    for tag, x, y in (("reseed_on", "arms3_reseed", "arms5_reseed"), ("reseed_off", "arms3_drift", "arms5_drift")):
        fx, fy = RD / f"{x}.json", RD / f"{y}.json"
        if not (fx.is_file() and fy.is_file()):
            continue
        rx_, ry_ = (json.loads(f.read_text())["rows"] for f in (fx, fy))
        kx = {(r["config"], r["task"], r["episode"]): r for r in rx_}
        ky = {(r["config"], r["task"], r["episode"]): r for r in ry_}
        order_x = list(dict.fromkeys(r["config"] for r in rx_))
        order_y = list(dict.fromkeys(r["config"] for r in ry_))
        arms_ = []
        for c_ in [c for c in order_x if c in order_y]:
            ks = [k for k in kx if k[0] == c_ and k in ky]
            arms_.append({"arm": c_, "episodes": len(ks),
                          "steps_changed": sum(kx[k]["steps"] != ky[k]["steps"] for k in ks),
                          "outcomes_changed": sum(kx[k]["success"] != ky[k]["success"] for k in ks),
                          "position": [order_x.index(c_), order_y.index(c_)]})
        ps[tag] = {"runs": [rel(fx), rel(fy)], "arms": arms_}
        print(f"   process sharing ({tag}): " + ", ".join(
            f"{a['arm']} steps changed {a['steps_changed']}/{a['episodes']}, outcomes {a['outcomes_changed']}"
            for a in arms_))
    if ps:
        S["process_sharing"] = ps

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(S, indent=1, default=float) + "\n")
    print(f"\nwrote {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()

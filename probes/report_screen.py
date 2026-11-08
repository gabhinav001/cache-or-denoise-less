"""Section 15 of the analysis: does a screen of uncached rollouts predict closed-loop cache failure?

    python3 probes/report_screen.py [results/budget_shift]

The predictions tested here (P4-P8) are stated in repro/experiments/screen.sh,
which was committed and pushed with this file before any screening run. Every
quantity below is defined here as it was then; report_budget_shift.py calls
`analyze` and stores the result under `screen` in the snapshot.

Definitions
  D(config)   the screen's deviation: for each of the 50 states, the mean over
              the uncached rollout's reuse calls of d (RMS difference between
              the configuration's action chunk and the uncached one over the ten
              executed actions, normalised action space; probes/screen_engine.py),
              then the mean over states
  S(config)   closed-loop success of the cached configuration on the same states
              (the shipped run named in CLOSED), or, for VALIDATION, on the other
              five initial states of every task
  share       1 - sigma_m from Eq. 1, the cached share of the flow path
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

K_STEPS = 10

# screening cell -> (checkpoint, shift, fresh noise); every cell is an uncached
# rollout of the first 5 initial states of each LIBERO-Goal task with seed 42
CELLS = {
    "scr_idm_s1": ("idm", 1.0, False), "scr_idm_s1_rn": ("idm", 1.0, True),
    "scr_idm_s2": ("idm", 2.0, False), "scr_idm_s2_rn": ("idm", 2.0, True),
    "scr_idm_s3": ("idm", 3.0, False), "scr_idm_s3_rn": ("idm", 3.0, True),
    "scr_idm_s5": ("idm", 5.0, False), "scr_idm_s5_rn": ("idm", 5.0, True),
    "scr_uncond_s1": ("uncond", 1.0, False), "scr_uncond_s1_rn": ("uncond", 1.0, True),
    "scr_uncond_s5": ("uncond", 5.0, False), "scr_uncond_s5_rn": ("uncond", 5.0, True),
}

# Every tau = 8 configuration with a closed-loop run on these 50 states before
# the screen ran: (cell, cached steps m, tau, run file under results/budget_shift)
CLOSED = [
    ("scr_idm_s1", 3, 8, "shift_a6000/idm_s1_p3_tau8.json"),
    ("scr_idm_s1", 4, 8, "shift_a6000/idm_s1_p4_tau8.json"),
    ("scr_idm_s1", 5, 8, "shift_a6000/idm_s1_p5_tau8.json"),
    ("scr_idm_s1", 6, 8, "shift_a6000/idm_s1_p6_tau8.json"),
    ("scr_idm_s1", 7, 8, "followup_a6000/idm_s1_p7_tau8.json"),
    ("scr_idm_s1", 8, 8, "shift_a6000/idm_s1_whole_tau8.json"),
    ("scr_idm_s1_rn", 4, 8, "noise_a6000/idm_s1_p4_tau8_rn.json"),
    ("scr_idm_s1_rn", 6, 8, "noise_a6000/idm_s1_p6_tau8_rn.json"),
    ("scr_idm_s1_rn", 7, 8, "followup_a6000/idm_s1_p7_tau8_rn.json"),
    ("scr_idm_s1_rn", 8, 8, "noise_a6000/idm_s1_whole_tau8_rn.json"),
    ("scr_idm_s2", 7, 8, "heldout_a6000/idm_s2_p7_tau8.json"),
    ("scr_idm_s2", 9, 8, "heldout_a6000/idm_s2_p9_tau8.json"),
    ("scr_idm_s3", 8, 8, "followup_a6000/idm_s3_p8_tau8.json"),
    ("scr_idm_s3", 9, 8, "followup_a6000/idm_s3_p9_tau8.json"),
    ("scr_idm_s3_rn", 9, 8, "followup_a6000/idm_s3_p9_tau8_rn.json"),
    ("scr_idm_s5", 8, 8, "shift_a6000/idm_s5_whole_tau8.json"),
    ("scr_idm_s5", 9, 8, "shift_a6000/idm_s5_p9_tau8.json"),
    ("scr_idm_s5_rn", 8, 8, "noise_a6000/idm_s5_whole_tau8_rn.json"),
    ("scr_uncond_s1", 5, 8, "shift_a6000/uncond_s1_p5_tau8.json"),
    ("scr_uncond_s1", 7, 8, "followup_a6000/uncond_s1_p7_tau8.json"),
    ("scr_uncond_s1", 8, 8, "shift_a6000/uncond_s1_whole_tau8.json"),
    ("scr_uncond_s5", 8, 8, "shift_a6000/uncond_s5_whole_tau8.json"),
    ("scr_uncond_s5", 9, 8, "shift_a6000/uncond_s5_p9_tau8.json"),
    ("scr_uncond_s5_rn", 8, 8, "noise_a6000/uncond_s5_whole_tau8_rn.json"),
]
# Other refresh intervals at eight cached steps (secondary: the share cannot
# tell them apart). The 100-state runs contribute their first 50 states.
TAU_VARIANTS = [
    ("scr_idm_s1", 8, 2, "frontier100_a6000/idm_whole_plain_tau2.json"),
    ("scr_idm_s1", 8, 4, "frontier100_a6000/idm_whole_plain_tau4.json"),
    ("scr_idm_s1", 8, 0, "frontier100_a6000/idm_whole_plain_tau0.json"),
    ("scr_uncond_s5", 8, 2, "frontier100_a6000/uncond_whole_tau2.json"),
    ("scr_uncond_s5", 8, 4, "frontier100_a6000/uncond_whole_tau4.json"),
    ("scr_uncond_s5", 8, 0, "shift_a6000/uncond_s5_whole_tau0.json"),
    ("scr_uncond_s5_rn", 8, 0, "noise_a6000/uncond_s5_whole_tau0_rn.json"),
]
# P8: configurations never run in closed loop before, run on the other 50
# states (initial states 5-9 of every task), and their uncached baselines
VALIDATION = [
    ("scr_uncond_s1_rn", 6, 8, "screen_a6000/val_uncond_s1_p6_tau8_rn.json"),
    ("scr_uncond_s1_rn", 8, 8, "screen_a6000/val_uncond_s1_p8_tau8_rn.json"),
    ("scr_idm_s2_rn", 7, 8, "screen_a6000/val_idm_s2_p7_tau8_rn.json"),
]
VAL_BASELINES = {"scr_uncond_s1_rn": "screen_a6000/val_uncond_s1_stock10_rn.json",
                 "scr_idm_s2_rn": "screen_a6000/val_idm_s2_stock10_rn.json"}
# shipped uncached runs of the same policy on the same 50 states: the screen's
# own rollouts are uncached and should repeat their outcomes
STOCK = {"scr_idm_s1": "shift_a6000/idm_s1_stock10.json",
         "scr_idm_s1_rn": "noise_a6000/idm_s1_stock10_rn.json",
         "scr_idm_s2": "heldout_a6000/idm_s2_stock10.json",
         "scr_idm_s3": "followup_a6000/idm_s3_stock10.json",
         "scr_idm_s5": "shift_a6000/idm_s5_stock10.json",
         "scr_uncond_s1": "shift_a6000/uncond_s1_stock10.json",
         "scr_uncond_s5": "shift_a6000/uncond_s5_stock10.json",
         "scr_uncond_s5_rn": "noise_a6000/uncond_s5_stock10_rn.json"}

BOOT, BOOT_SEED = 2000, 20261002


def share(m, s, k=K_STEPS):
    u = 1.0 - m / k
    return 1.0 - s * u / (1.0 + (s - 1.0) * u)


def _load(path):
    """A run file, or None if it does not exist (report_budget_shift.load's contract)."""
    path = Path(path)
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    if not isinstance(d, dict) or "rows" not in d:
        raise ValueError(f"{path}: not a run file")
    return d


def smoke(SD, load=_load):
    """The smoke check of probes/check_screen_smoke.py, as numbers for the snapshot."""
    sc, wh = load(SD / "smoke_screen.json"), load(SD / "smoke_whole8.json")
    if sc is None or wh is None:
        return None
    a, b = sc["rows"][0], wh["rows"][0]
    if (a["task"], a["init_state"]) != (b["task"], b["init_state"]):
        raise ValueError("smoke runs did not start from the same state")
    cyc = a["screen"]["cycle"]["p8_tau8"]
    if not cyc or cyc[0] != 1:
        raise ValueError("smoke: no replay at the first reuse call")
    return {"first_call_max_abs_diff": float(np.abs(np.array(a["chunks"][0]) - np.array(b["chunks"][0])).max()),
            "first_reuse_max_abs_diff": float(np.abs(np.array(a["screen"]["exec"]["p8_tau8"][0])
                                                     - np.array(b["chunks"][1])).max()),
            "uncached_vs_cache_at_first_reuse": round(float(np.abs(np.array(a["chunks"][1])
                                                                   - np.array(b["chunks"][1])).max()), 4),
            # after the first reuse call the closed-loop cache has executed its own
            # actions, so the replays (which follow the uncached rollout) diverge from it
            "later_calls": [c for c in cyc if c < len(b["chunks"])],
            "later_max_abs_diff": [round(float(np.abs(np.array(e) - np.array(b["chunks"][c])).max()), 4)
                                   for c, e in zip(cyc, a["screen"]["exec"]["p8_tau8"]) if c < len(b["chunks"])]}


def _key(r):
    return (int(r["task"]), str(r["init_state"]))


def _check(d, path, *, config, noise, shift, m=None, sha=None, first_episode=0):
    """Refuse a run that is not the configuration the table says it is.

    config is "screen", "stock" or "whole_tauT"; a whole-stack cache must also
    have cached exactly its first m steps.
    """
    meta = d["meta"]
    ok_cfg = meta.get("configs") == [config]
    if config.startswith("whole_tau"):
        ok_cfg = ok_cfg and int(meta.get("cache_prefix", -1)) == m
    probs = []
    if not ok_cfg:
        probs.append(f"config {meta.get('configs')} prefix {meta.get('cache_prefix')}, want {config} m={m}")
    if int(meta.get("resample_noise", 0)) != int(noise):
        probs.append(f"resample_noise {meta.get('resample_noise')}")
    if abs(float(meta["runtime"]["effective_action_sigma_shift"]) - shift) > 1e-9:
        probs.append(f"shift {meta['runtime']['effective_action_sigma_shift']}")
    if int(meta.get("seed", 42)) != 42:
        probs.append(f"seed {meta.get('seed')}")
    if int(meta.get("num_inference_steps") or K_STEPS) != K_STEPS:
        probs.append(f"steps {meta.get('num_inference_steps')}")
    if sha is not None and meta["runtime"].get("checkpoint_sha256") != sha:
        probs.append("checkpoint differs from the screening cell's")
    eps = {int(r["episode"]) for r in d["rows"]}
    if not eps <= set(range(first_episode, first_episode + 10)):
        probs.append(f"episodes {sorted(eps)}")
    if probs:
        raise ValueError(f"{path}: " + "; ".join(probs))


def _success(d, keys=None):
    out = {_key(r): int(bool(r["success"])) for r in d["rows"]}
    return out if keys is None else {k: out[k] for k in keys if k in out}


def _state_dev(d, label):
    """state -> mean d over that rollout's reuse calls, for one screened configuration."""
    out = {}
    for r in d["rows"]:
        v = r["screen"]["d"][label]
        if v:
            out[_key(r)] = float(np.mean(v))
    return out


def _state_metric(d, field, label):
    out = {}
    for r in d["rows"]:
        v = r["screen"][field][label]
        if v:
            out[_key(r)] = float(np.mean(v))
    return out


def _ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def logistic_fit(x, k, n, ridge=1e-6, iters=100):
    """Binomial logistic regression logit(p) = a + b x by Newton's method."""
    x, k, n = (np.asarray(v, float) for v in (x, k, n))
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (k - n * p) - ridge * w
        H = (X * (n * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.abs(step).max() < 1e-10:
            break
    return [float(w[0]), float(w[1])]


def logistic_predict(w, x):
    return 1.0 / (1.0 + math.exp(-(w[0] + w[1] * float(x))))


def analyze(NEW, load=None):
    """The section-15 analysis. `load` reads a run file or returns None when it is
    absent; report_budget_shift passes its own, so the files read here are traced
    like every other input of the snapshot."""
    load = load or _load
    NEW = Path(NEW)
    SD = NEW / "screen_a6000"
    if not SD.is_dir():
        return None
    runs, out = {}, {"cells": {}}
    for cell, (ck, s, noise) in CELLS.items():
        p = SD / f"{cell}.json"
        d = load(p)
        if d is None:
            continue
        _check(d, p, config="screen", noise=noise, shift=s)
        if len(d["rows"]) != 50:
            raise ValueError(f"{p}: {len(d['rows'])} episodes, expected 50")
        if not all(r["screen"]["hits_ok"] for r in d["rows"]):
            raise ValueError(f"{p}: a replay did not substitute every cached step")
        runs[cell] = d
        labels = d["meta"]["screen"]
        drift = [row for r in d["rows"] for row in r["screen"]["drift8"]]
        cinfo = {"ckpt": ck, "shift": s, "fresh_noise": noise,
                 "uncached": {"k": sum(r["success"] for r in d["rows"]), "n": len(d["rows"])},
                 "D": {}, "grip": {}, "e": {},
                 "drift8_by_step": [round(float(np.mean([row[k] for row in drift])), 5)
                                    for k in range(K_STEPS)] if drift else None}
        # descriptive: replay time over the uncached call's own time, per call,
        # on episodes after the first of each task (the first pays compilation)
        _ok = [r for r in d["rows"] if not r.get("warmup") and r["screen"].get("screen_ms") and r.get("ms_per_cycle")]
        if _ok:
            _sm = float(np.mean([np.mean(r["screen"]["screen_ms"]) for r in _ok]))
            _tot = float(np.mean([r["ms_per_cycle"] for r in _ok]))
            cinfo["replay_cost_ratio"] = round(_sm / (_tot - _sm), 2)
        cinfo["grip_all"] = {}
        for lb in labels:
            sd = _state_dev(d, lb)
            cinfo["D"][lb] = round(float(np.mean(list(sd.values()))), 5)
            # descriptive (not a prediction): share of reuse calls whose replay
            # commands the opposite gripper state on all ten executed actions
            g_ = [x for r in d["rows"] for x in r["screen"]["grip"][lb]]
            cinfo["grip_all"][lb] = round(sum(x == 10 for x in g_) / len(g_), 4) if g_ else None
            cinfo["grip"][lb] = round(float(np.mean(list(_state_metric(d, "grip", lb).values()))), 4)
            cinfo["e"][lb] = round(float(np.mean(list(_state_metric(d, "e", lb).values()))), 5)
        st = load(NEW / STOCK[cell]) if cell in STOCK else None
        if st is not None:
            a, b = _success(d), _success(st)
            common = sorted(set(a) & set(b))
            cinfo["vs_shipped_uncached"] = {"source": STOCK[cell], "paired": len(common),
                                            "outcome_changes": sum(a[k] != b[k] for k in common)}
        out["cells"][cell] = cinfo
    if not runs:
        return None

    def table(rows):
        res = []
        for cell, m, tau, rel in rows:
            cl = load(NEW / rel) if cell in runs else None
            if cl is None:
                continue
            d = runs[cell]; ck, s, noise = CELLS[cell]
            _check(cl, rel, config=f"whole_tau{tau}", m=m, noise=noise, shift=s,
                   sha=d["meta"]["runtime"]["checkpoint_sha256"])
            lb = f"p{m}_tau{tau}"
            dev = _state_dev(d, lb)
            succ = _success(cl, dev.keys())
            keys = sorted(succ)
            if len(keys) != 50:
                raise ValueError(f"{rel}: {len(keys)} states shared with {cell}, expected 50")
            res.append({"cell": cell, "ckpt": ck, "shift": s, "fresh_noise": noise, "m": m, "tau": tau,
                        "share": round(share(m, s), 4), "source": rel,
                        "D": round(float(np.mean([dev[k] for k in keys])), 5),
                        "success": {"k": sum(succ[k] for k in keys), "n": len(keys),
                                    "pct": round(100.0 * sum(succ[k] for k in keys) / len(keys), 1)},
                        "_dev": [dev[k] for k in keys], "_succ": [succ[k] for k in keys]})
        return res

    prim = table(CLOSED)
    taus = table(TAU_VARIANTS)
    out["n_primary"] = len(prim)

    def rho(rows):
        if len(rows) < 3:
            return None
        Dv = [r["D"] for r in rows]; Sv = [r["success"]["pct"] for r in rows]; Hv = [r["share"] for r in rows]
        return {"D": round(spearman(Dv, Sv), 4), "share": round(spearman(Hv, Sv), 4), "n": len(rows)}

    out["rho_primary"] = rho(prim)
    out["rho_all"] = rho(prim + taus)
    out["rho_fixed_noise"] = rho([r for r in prim if not r["fresh_noise"]])
    # bootstrap over the 50 states: the same resample for every configuration
    if len(prim) >= 3:
        rng = np.random.default_rng(BOOT_SEED)
        devs = np.array([r["_dev"] for r in prim]); succs = np.array([r["_succ"] for r in prim])
        H = [r["share"] for r in prim]; diffs = []
        for _ in range(BOOT):
            idx = rng.integers(0, devs.shape[1], devs.shape[1])
            Dv = devs[:, idx].mean(1); Sv = succs[:, idx].mean(1)
            diffs.append(spearman(Dv, Sv) - spearman(H, Sv))
        diffs = np.array([x for x in diffs if not math.isnan(x)])
        out["rho_primary"]["diff_boot90"] = [round(float(np.quantile(diffs, 0.05)), 4),
                                             round(float(np.quantile(diffs, 0.95)), 4)]

    def get(cell, lb):
        return out["cells"].get(cell, {}).get("D", {}).get(lb)

    P = {}
    a, b = get("scr_idm_s1", "p8_tau8"), get("scr_uncond_s1", "p8_tau8")
    if a is not None and b is not None:
        P["P4"] = {"idm": a, "original": b, "met": bool(a > b)}
    pairs = [(get("scr_idm_s1_rn", f"p{m}_tau8"), get("scr_idm_s1", f"p{m}_tau8")) for m in (4, 6, 7, 8)]
    if all(x is not None and y is not None for x, y in pairs):
        P["P5"] = {"fresh": [x for x, _ in pairs], "fixed": [y for _, y in pairs],
                   "met": bool(all(x > y for x, y in pairs))}
    chain = [get(f"scr_idm_s{s}", "p8_tau8") for s in (1, 2, 3, 5)]
    if all(x is not None for x in chain):
        P["P6"] = {"s1_s2_s3_s5": chain, "met": bool(all(chain[i] > chain[i + 1] for i in range(3)))}
    if out["rho_primary"] and len(prim) == len(CLOSED):
        r = out["rho_primary"]
        P["P7"] = {"rho_D": r["D"], "rho_share": r["share"], "met": bool(r["D"] <= -0.7 and r["D"] < r["share"])}

    # P8: predictions for the validation configurations, fixed by the rule above
    if len(prim) == len(CLOSED):
        k = [r["success"]["k"] for r in prim]; n = [r["success"]["n"] for r in prim]
        wD = logistic_fit([math.log(r["D"]) for r in prim], k, n)
        wH = logistic_fit([r["share"] for r in prim], k, n)
        val = []
        for cell, m, tau, rel in VALIDATION:
            lb = f"p{m}_tau{tau}"
            Dv = get(cell, lb)
            if Dv is None:
                continue
            ck, s, noise = CELLS[cell]
            row = {"cell": cell, "m": m, "tau": tau, "share": round(share(m, s), 4), "D": Dv,
                   "pred_D": round(100 * logistic_predict(wD, math.log(Dv)), 1),
                   "pred_share": round(100 * logistic_predict(wH, share(m, s)), 1), "source": rel}
            cl = load(NEW / rel)
            if cl is not None:
                _check(cl, rel, config=f"whole_tau{tau}", m=m, noise=noise, shift=s, first_episode=5,
                       sha=runs[cell]["meta"]["runtime"]["checkpoint_sha256"] if cell in runs else None)
                eps = {int(r_["episode"]) for r_ in cl["rows"]}
                if eps != set(range(5, 10)) or len(cl["rows"]) != 50:
                    raise ValueError(f"{rel}: expected initial states 5-9 of every task")
                kk = sum(r_["success"] for r_ in cl["rows"])
                row["observed"] = {"k": kk, "n": 50, "pct": round(100.0 * kk / 50, 1)}
            val.append(row)
        out["logistic"] = {"D": {"x": "log D", "w": [round(v, 5) for v in wD]},
                           "share": {"x": "share", "w": [round(v, 5) for v in wH]}}
        out["validation"] = val
        base = {}
        for cell, rel in VAL_BASELINES.items():
            bl = load(NEW / rel)
            if bl is not None:
                ck, s, noise = CELLS[cell]
                _check(bl, rel, config="stock", noise=noise, shift=s, first_episode=5,
                       sha=runs[cell]["meta"]["runtime"]["checkpoint_sha256"] if cell in runs else None)
                if {int(r_["episode"]) for r_ in bl["rows"]} != set(range(5, 10)) or len(bl["rows"]) != 50:
                    raise ValueError(f"{rel}: expected initial states 5-9 of every task")
                kk = sum(r_["success"] for r_ in bl["rows"])
                base[cell] = {"k": kk, "n": len(bl["rows"]), "pct": round(100.0 * kk / len(bl["rows"]), 1),
                              "source": rel}
        out["validation_uncached"] = base
        if val and all("observed" in v for v in val) and len(val) == len(VALIDATION):
            maeD = float(np.mean([abs(v["pred_D"] - v["observed"]["pct"]) for v in val]))
            maeH = float(np.mean([abs(v["pred_share"] - v["observed"]["pct"]) for v in val]))
            P["P8"] = {"mae_D": round(maeD, 1), "mae_share": round(maeH, 1), "met": bool(maeD < maeH)}
    out["predictions"] = P
    allc = prim + taus
    lo_ = [r for r in allc if r["D"] < 0.13]; hi_ = [r for r in allc if r["D"] > 0.37]
    out["bands"] = {"below_0.13": {"n": len(lo_), "min_success_pct": min((r["success"]["pct"] for r in lo_), default=None)},
                    "above_0.37": {"n": len(hi_), "max_success_pct": max((r["success"]["pct"] for r in hi_), default=None)},
                    "between": {"n": len(allc) - len(lo_) - len(hi_)},
                    "note": "descriptive, chosen after the runs; not among the predictions"}
    out["smoke"] = smoke(SD, load)
    for r in prim + taus:
        r.pop("_dev"); r.pop("_succ")
    out["primary"] = prim
    out["tau_variants"] = taus
    return out


def main(argv):
    NEW = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "results/budget_shift"
    res = analyze(NEW)
    if res is None:
        print("no screening runs under", NEW / "screen_a6000"); return 0
    print(json.dumps({k: v for k, v in res.items() if k not in ("primary", "tau_variants")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

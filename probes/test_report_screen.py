"""Run probes/report_screen.py end to end on synthetic screening runs.

The closed-loop runs are the shipped ones; the screening and validation runs are
fabricated in a temporary directory with the shipped runs' state keys, so the
analysis is exercised on exactly the joins it will make on real data. Checks:
the joins cover all 24 + 7 configurations, a deviation built to track success
outranks the cached share, the predictions P4-P8 are evaluated, and a
mislabelled run is refused.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_screen as RS   # noqa: E402

NEW = HERE.parent / "results/budget_shift"
fails = 0
def check(cond, msg):
    global fails
    if not cond:
        print(f"  FAIL: {msg}"); fails += 1


def rows_of(rel):
    return json.loads((NEW / rel).read_text())["rows"]


def sha_of(rel):
    return json.loads((NEW / rel).read_text())["meta"]["runtime"]["checkpoint_sha256"]


SHA = {"idm": sha_of("shift_a6000/idm_s1_stock10.json"), "uncond": sha_of("shift_a6000/uncond_s5_stock10.json")}
# state keys: episodes 0-4 from a 50-state run, 5-9 from a 100-state run of each checkpoint
KEYS = {ck: [(r["task"], r["episode"], r["init_state"]) for r in rows_of(f"frontier100_a6000/{ck}_stock_10step.json")]
        for ck in ("idm", "uncond")}
SUCC = {}
for cell, m, tau, rel in RS.CLOSED + RS.TAU_VARIANTS:
    rows = rows_of(rel)
    SUCC[(cell, m, tau)] = sum(r["success"] for r in rows if r["episode"] < 5) / 50.0

rng = random.Random(7)


def fake_screen(cell, labels):
    ck, s, noise = RS.CELLS[cell]
    rows = []
    for task, ep, key in KEYS[ck]:
        if ep >= 5:
            continue
        sc = {"configs": labels, "hits_ok": True, "d": {}, "grip": {}, "e": {}, "cycle": {},
              "drift8": [[0.1] * 10] * 3, "screen_ms": []}
        for lb in labels:
            m, tau = int(lb[1:lb.index("_")]), int(lb.split("tau")[1])
            # built to track the shipped closed-loop success where it exists
            p = SUCC.get((cell, m, tau), 1.0 - RS.share(m, s) ** 3)
            base = 0.02 + 0.5 * (1.0 - p) + (0.05 if noise else 0.0)
            sc["d"][lb] = [base * math.exp(rng.gauss(0, 0.1)) for _ in range(4)]
            sc["grip"][lb] = [0, 1, 0, 0]; sc["e"][lb] = [0.1] * 4; sc["cycle"][lb] = [1, 2, 3, 4]
        rows.append({"task": task, "episode": ep, "init_state": key, "success": rng.random() < 0.95,
                     "config": "screen", "screen": sc})
    meta = {"configs": ["screen"], "resample_noise": int(noise), "seed": 42, "num_inference_steps": 10,
            "cache_prefix": 0, "screen": labels,
            "runtime": {"effective_action_sigma_shift": s, "checkpoint_sha256": SHA[ck]}}
    return {"meta": meta, "rows": rows}


def fake_closed(cell, m, tau, first_episode, config=None):
    ck, s, noise = RS.CELLS[cell]
    rows = [{"task": t, "episode": e, "init_state": k, "success": rng.random() < 0.7, "config": "x"}
            for t, e, k in KEYS[ck] if first_episode <= e < first_episode + 5]
    meta = {"configs": [config or f"whole_tau{tau}"], "cache_prefix": m, "resample_noise": int(noise), "seed": 42,
            "num_inference_steps": 10,
            "runtime": {"effective_action_sigma_shift": s, "checkpoint_sha256": SHA[ck]}}
    return {"meta": meta, "rows": rows}


tmp = Path(tempfile.mkdtemp())
try:
    for sub in ("shift_a6000", "noise_a6000", "followup_a6000", "heldout_a6000", "frontier100_a6000"):
        (tmp / sub).symlink_to(NEW / sub)
    (tmp / "screen_a6000").mkdir()
    labels = [RS.label(m, t) if hasattr(RS, "label") else f"p{m}_tau{t}" for m, t in
              ((3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (8, 2), (8, 4), (8, 0))]
    for cell in RS.CELLS:
        (tmp / "screen_a6000" / f"{cell}.json").write_text(json.dumps(fake_screen(cell, labels)))
    for cell, m, tau, rel in RS.VALIDATION:
        (tmp / rel).write_text(json.dumps(fake_closed(cell, m, tau, 5)))
    for cell, rel in RS.VAL_BASELINES.items():
        (tmp / rel).write_text(json.dumps(fake_closed(cell, 0, None, 5, config="stock")))

    res = RS.analyze(tmp)
    check(res is not None, "no result")
    check(res["n_primary"] == 24, f"{res['n_primary']} primary configurations joined, expected 24")
    check(len(res["tau_variants"]) == 7, f"{len(res['tau_variants'])} refresh-interval variants")
    r = res["rho_primary"]
    print(f"  synthetic: rho(D, success) = {r['D']}, rho(share, success) = {r['share']}, "
          f"bootstrap 90% of the difference {r['diff_boot90']}")
    check(r["D"] < r["share"], "a deviation built to track success should outrank the share")
    P = res["predictions"]
    check(set(P) == {"P4", "P5", "P6", "P7", "P8"}, f"predictions evaluated: {sorted(P)}")
    for v in res["validation"]:
        check(0 <= v["pred_D"] <= 100 and 0 <= v["pred_share"] <= 100, f"prediction out of range: {v}")
        check(v["observed"]["n"] == 50, "validation states")
    check(set(res["validation_uncached"]) == set(RS.VAL_BASELINES), "validation baselines")
    check(all("vs_shipped_uncached" in res["cells"][c] for c in RS.STOCK), "screen vs shipped uncached runs")

    # a run that is not what the table says is refused
    bad = fake_closed("scr_uncond_s1_rn", 7, 8, 5)
    (tmp / RS.VALIDATION[0][3]).write_text(json.dumps(bad))
    try:
        RS.analyze(tmp); check(False, "a mislabelled validation run was accepted")
    except ValueError as e:
        print(f"  mislabelled run refused: {str(e)[:90]}")
finally:
    shutil.rmtree(tmp)

if fails:
    print(f"report_screen: {fails} FAILED"); sys.exit(1)
print("report_screen: joins, predictions and refusals as specified")

"""Run probes/report_latent_delay.py end to end on synthetic runs.

The comparison runs are the shipped ones; the new runs are fabricated in a
temporary directory with the shipped runs' states and the metadata the harness
writes, so the analysis is exercised on exactly the joins it will make on real
data. Checks: every comparison joins on 50 states, P9-P11 are evaluated, the
screen's latent check and D work, and a run with the wrong noise policy is refused.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_budget_shift as rb      # noqa: E402
import report_latent_delay as LDm     # noqa: E402

NEW = HERE.parent / "results/budget_shift"
# the fabricated runs live outside the repository; report_budget_shift labels
# loaded runs by their path inside it, which the joins tested here do not use
rb.rel = lambda path: str(path)
fails = 0
def check(cond, msg):
    global fails
    if not cond:
        print(f"  FAIL: {msg}"); fails += 1

rng = random.Random(3)


def fake(base_rel, *, configs, steps, prefix=None, episode=0, delay=(0, 0), p=0.8, screen=False):
    d = json.loads((NEW / base_rel).read_text())
    meta = copy.deepcopy(d["meta"])
    meta.update(configs=configs, num_inference_steps=steps, seed=42, n_tasks=10, n_episodes=5, resample_noise=0)
    if prefix is not None:
        meta["cache_prefix"] = prefix
    if episode:
        meta["episode_noise"] = 1
    if any(delay):
        meta["delay"] = list(delay)
    if screen:
        meta["screen"] = ["8/8"]
    rows = []
    for r in d["rows"]:
        if r["episode"] >= 5:
            continue
        r = dict(r, success=rng.random() < p, config=configs[0])
        if episode:
            r["episode_latent"] = f"lat{r['task']}_{r['episode']}"
        if screen:
            r["screen"] = {"drift8": [[0.6] * 10] * 3, "latent": [r["episode_latent"]] * 4, "d": {"8/8": [0.4, 0.5]}}
        rows.append(r)
    return {"meta": meta, "rows": rows}


tmp = Path(tempfile.mkdtemp())
try:
    for sub in ("shift_a6000", "noise_a6000"):
        (tmp / sub).symlink_to(NEW / sub)
    (tmp / LDm.LD).mkdir()
    def put(rel, run):
        (tmp / rel).write_text(json.dumps(run))
    put(LDm.LATENT[6]["episode"], fake("shift_a6000/idm_s1_p6_tau8.json", configs=["whole_tau8"], steps=10, prefix=6, episode=1, p=0.94))
    put(LDm.LATENT[8]["episode"], fake("shift_a6000/idm_s1_whole_tau8.json", configs=["whole_tau8"], steps=10, prefix=8, episode=1, p=0.55))
    for (name, d_), rel in LDm.DELAY.items():
        if d_:
            put(rel, fake("shift_a6000/idm_s1_stock10.json", configs=["stock"], steps=LDm.STEPS[name], delay=(d_, d_), p=0.9))
    put(LDm.CACHE_DELAY["two_d2"], fake("shift_a6000/uncond_s5_stock2.json", configs=["stock"], steps=2, delay=(2, 2), p=0.9))
    put(LDm.CACHE_DELAY["cache_d52"], fake("shift_a6000/uncond_s5_whole_tau8.json", configs=["whole_tau8"], steps=10, prefix=8, delay=(5, 2), p=0.9))
    put(LDm.SCREEN_EP, fake("shift_a6000/idm_s1_stock10.json", configs=["screen"], steps=10, episode=1, p=0.95, screen=True))

    res = LDm.analyze(tmp, rb.load, rb.paired, rb.rate)
    check(res is not None and set(res["predictions"]) == {"P9", "P10", "P11"}, f"predictions: {sorted((res or {}).get('predictions', {}))}")
    for k in ("p6", "p8"):
        check(res["latent"][k]["episode_vs_fresh"]["n"] == 50, f"{k}: joined states")
    check(res["delay"]["two_d2_vs_ten_d5"]["n"] == 50, "delay join")
    check(res["cache_delay"]["cache_d52_vs_two_d2"]["n"] == 50, "cache delay join")
    check(res["screen_episode"]["one_latent_per_episode"] and res["screen_episode"]["distinct_latents"] == 50, "screen latent check")
    check(res["screen_episode"].get("D") == {"8/8": 0.45}, f"screen D: {res['screen_episode'].get('D')}")
    print(f"  synthetic: P9 {res['predictions']['P9']}, P11 {res['predictions']['P11']}")

    # delay_graphs.sh: ten steps at the delays of CUDA-graph execution
    (tmp / LDm.GRAPHS).mkdir()
    put(LDm.DELAY_GRAPHS[("idm", 3)], fake("shift_a6000/idm_s1_stock10.json", configs=["stock"], steps=10, delay=(3, 3), p=0.9))
    put(LDm.DELAY_GRAPHS[("orig", 3)], fake("shift_a6000/uncond_s5_stock10.json", configs=["stock"], steps=10, delay=(3, 3), p=0.9))
    put(LDm.DELAY_GRAPHS[("idm", 4)], fake("shift_a6000/idm_s1_stock10.json", configs=["stock"], steps=10, delay=(4, 4), p=0.9))
    gr = LDm.analyze_graphs(tmp, rb.load, rb.paired, rb.rate)
    check(gr is not None and set(gr) == {"idm", "orig"}, f"graphs: {sorted(gr or {})}")
    check(all(gr[c]["two_d2_vs_ten_d3"]["n"] == 50 and gr[c]["ten_d3_vs_ten_d0"]["n"] == 50 for c in gr), "graphs joins")
    check(sorted(gr["idm"]["curve_ten"]) == [0, 2, 3, 4, 5], f"curve: {gr['idm'].get('curve_ten')}")
    put(LDm.DELAY_GRAPHS[("orig", 3)], fake("shift_a6000/uncond_s5_stock10.json", configs=["stock"], steps=10, delay=(5, 5), p=0.9))
    try:
        LDm.analyze_graphs(tmp, rb.load, rb.paired, rb.rate); check(False, "a graph run with the wrong delay was accepted")
    except ValueError as e:
        print(f"  wrong delay refused: {str(e)[:70]}")

    bad = fake("shift_a6000/idm_s1_p6_tau8.json", configs=["whole_tau8"], steps=10, prefix=6, episode=0, p=0.9)
    put(LDm.LATENT[6]["episode"], bad)
    try:
        LDm.analyze(tmp, rb.load, rb.paired, rb.rate); check(False, "a run without episode noise was accepted")
    except ValueError as e:
        print(f"  wrong noise policy refused: {str(e)[:80]}")
finally:
    shutil.rmtree(tmp)

if fails:
    print(f"report_latent_delay: {fails} FAILED"); sys.exit(1)
print("report_latent_delay: joins, predictions and refusals as specified")

"""Section 16 of the analysis: an episode-latent control for the fresh-noise loss, and delayed execution.

    python3 probes/report_latent_delay.py [results/budget_shift]

The predictions tested here (P9-P11) are stated in repro/experiments/latent_delay.sh,
which was committed and pushed with this file before any of its runs. All runs
use Optional-IDM unless named otherwise, on the first five initial states of
every LIBERO-Goal task (50 states), run seed 42.

  latent  the tau = 8 cache over the first 6 and 8 of ten steps at s = 1 with
          each episode's first fresh latent held for the whole episode
          (episode_noise), against the same caches with the run's fixed latent
          and with fresh noise on every call (shipped runs)
  delay   uncached two and ten steps executed with a delay of 2 or 5 control
          steps (50 ms each at 20 Hz): the robot keeps executing the previous
          chunk until a call's result arrives (fw_libero.delayed_actions)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LD = "latent_delay_a6000"
LATENT = {
    6: {"episode": f"{LD}/ld_idm_s1_p6_tau8_ep.json", "fixed": "shift_a6000/idm_s1_p6_tau8.json",
        "fresh": "noise_a6000/idm_s1_p6_tau8_rn.json"},
    8: {"episode": f"{LD}/ld_idm_s1_p8_tau8_ep.json", "fixed": "shift_a6000/idm_s1_whole_tau8.json",
        "fresh": "noise_a6000/idm_s1_whole_tau8_rn.json"},
}
DELAY = {("two", 0): "shift_a6000/idm_s1_stock2.json", ("ten", 0): "shift_a6000/idm_s1_stock10.json",
         ("two", 2): f"{LD}/ld_idm_s1_stock2_d2.json", ("two", 5): f"{LD}/ld_idm_s1_stock2_d5.json",
         ("ten", 2): f"{LD}/ld_idm_s1_stock10_d2.json", ("ten", 5): f"{LD}/ld_idm_s1_stock10_d5.json"}
STEPS = {"two": 2, "ten": 10}
# descriptive: the original checkpoint's tau = 8 cache, whose refresh calls take
# about 210 ms (5 steps) and reuse calls about 100 ms (2), against two steps
CACHE_DELAY = {"two_sync": "shift_a6000/uncond_s5_stock2.json", "cache_sync": "shift_a6000/uncond_s5_whole_tau8.json",
               "two_d2": f"{LD}/ld_uncond_s5_stock2_d2.json", "cache_d52": f"{LD}/ld_uncond_s5_p8_tau8_d52.json"}
SCREEN_EP = f"{LD}/ld_scr_idm_s1_ep.json"


def _meta_ok(d, rel, *, configs, steps, shift, prefix=None, noise=0, episode=0, delay=(0, 0)):
    m = d["meta"]; probs = []
    if m.get("configs") != configs:
        probs.append(f"configs {m.get('configs')}")
    if int(m.get("num_inference_steps") or 0) != steps:
        probs.append(f"steps {m.get('num_inference_steps')}")
    if abs(float(m["runtime"]["effective_action_sigma_shift"]) - shift) > 1e-9:
        probs.append("shift")
    if prefix is not None and int(m.get("cache_prefix", -1)) != prefix:
        probs.append(f"prefix {m.get('cache_prefix')}")
    if int(m.get("resample_noise", 0)) != noise or int(m.get("episode_noise", 0)) != episode:
        probs.append("noise policy")
    if tuple(m.get("delay", [0, 0])) != tuple(delay):
        probs.append(f"delay {m.get('delay')}")
    if int(m.get("seed", 42)) != 42 or int(m.get("first_episode", 0)) != 0 or int(m.get("n_episodes", 0)) != 5:
        probs.append("seed or states")
    if probs:
        raise ValueError(f"{rel}: " + "; ".join(probs))


def analyze(NEW, load, paired, rate):
    """`load`, `paired` and `rate` are report_budget_shift's, so the files read are traced
    and the contrasts are the paper's: a minus b on shared states, Tango 90% and exact McNemar."""
    NEW = Path(NEW)
    if not (NEW / LD).is_dir():
        return None
    out, P = {}, {}

    lat = {}
    for m_, files in LATENT.items():
        runs = {k: load(NEW / rel) for k, rel in files.items()}
        if any(v is None for v in runs.values()):
            continue
        _meta_ok(runs["episode"], files["episode"], configs=["whole_tau8"], steps=10, shift=1.0, prefix=m_, episode=1)
        lat[m_] = {k: rate(v) for k, v in runs.items()}
        lat[m_]["episode_vs_fresh"] = paired(runs["episode"], runs["fresh"], f"p{m_}: episode latent vs fresh noise")
        lat[m_]["episode_vs_fixed"] = paired(runs["episode"], runs["fixed"], f"p{m_}: episode latent vs fixed latent")
        lat[m_]["fixed_vs_fresh"] = paired(runs["fixed"], runs["fresh"], f"p{m_}: fixed vs fresh")
    if lat:
        out["latent"] = {f"p{m_}": v for m_, v in lat.items()}
    if len(lat) == len(LATENT):
        P["P9"] = {"diff_vs_fresh": [lat[m_]["episode_vs_fresh"]["diff"] for m_ in LATENT],
                   "met": bool(all(lat[m_]["episode_vs_fresh"]["diff"] >= 8 for m_ in LATENT))}
        P["P10"] = {"diff_vs_fixed": [lat[m_]["episode_vs_fixed"]["diff"] for m_ in LATENT],
                    "met": bool(all(abs(lat[m_]["episode_vs_fixed"]["diff"]) <= 10 for m_ in LATENT))}

    runs = {k: load(NEW / rel) for k, rel in DELAY.items()}
    if all(v is not None for v in runs.values()):
        for (name, d_), r in runs.items():
            if d_:
                _meta_ok(r, DELAY[(name, d_)], configs=["stock"], steps=STEPS[name], shift=1.0, delay=(d_, d_))
        out["delay"] = {
            "rates": {f"{n}_d{d_}": rate(r) for (n, d_), r in runs.items()},
            "two_d2_vs_ten_d5": paired(runs[("two", 2)], runs[("ten", 5)], "two steps at delay 2 vs ten at delay 5"),
            "two_d5_vs_two_d2": paired(runs[("two", 5)], runs[("two", 2)], "two steps: delay 5 vs 2"),
            "ten_d5_vs_ten_d2": paired(runs[("ten", 5)], runs[("ten", 2)], "ten steps: delay 5 vs 2"),
            "two_d2_vs_two_d0": paired(runs[("two", 2)], runs[("two", 0)], "two steps: delay 2 vs synchronous"),
            # descriptive, added after the runs for the round 10 reviews
            "two_d5_vs_two_d0": paired(runs[("two", 5)], runs[("two", 0)], "two steps: delay 5 vs synchronous"),
            "ten_d5_vs_ten_d0": paired(runs[("ten", 5)], runs[("ten", 0)], "ten steps: delay 5 vs synchronous"),
            "ten_d2_vs_two_d2": paired(runs[("ten", 2)], runs[("two", 2)], "ten vs two steps at delay 2"),
        }
        P["P11"] = {"two_d2": rate(runs[("two", 2)])["pct"], "ten_d5": rate(runs[("ten", 5)])["pct"],
                    "met": bool(rate(runs[("two", 2)])["pct"] >= rate(runs[("ten", 5)])["pct"])}

    cd = {k: load(NEW / rel) for k, rel in CACHE_DELAY.items()}
    if all(v is not None for v in cd.values()):
        _meta_ok(cd["two_d2"], CACHE_DELAY["two_d2"], configs=["stock"], steps=2, shift=5.0, delay=(2, 2))
        _meta_ok(cd["cache_d52"], CACHE_DELAY["cache_d52"], configs=["whole_tau8"], steps=10, shift=5.0, prefix=8, delay=(5, 2))
        out["cache_delay"] = {"rates": {k: rate(v) for k, v in cd.items()},
                              "cache_d52_vs_two_d2": paired(cd["cache_d52"], cd["two_d2"], "original: tau=8 cache (delays 5/2) vs two steps (2)"),
                              "cache_d52_vs_cache_sync": paired(cd["cache_d52"], cd["cache_sync"], "original tau=8 cache: delayed vs synchronous"),
                              "two_d2_vs_two_sync": paired(cd["two_d2"], cd["two_sync"], "original two steps: delayed vs synchronous")}

    se = load(NEW / SCREEN_EP)
    if se is not None:
        _meta_ok(se, SCREEN_EP, configs=["screen"], steps=10, shift=1.0, episode=1)
        rows = se["rows"]
        drift = [row for r in rows for row in r["screen"]["drift8"]]
        same = all(len(set(r["screen"]["latent"])) == 1 and r["screen"]["latent"][0] == r["episode_latent"] for r in rows)
        out["screen_episode"] = {"uncached": rate(se),
                                 "drift8_steps0_7": round(sum(sum(row[:8]) / 8 for row in drift) / len(drift), 4),
                                 "one_latent_per_episode": bool(same),
                                 "distinct_latents": len({r["episode_latent"] for r in rows})}
        # descriptive, added after the runs: D per screened configuration, defined as
        # in report_screen (the mean over states of each rollout's mean over reuse calls)
        D = {}
        for lb in se["meta"].get("screen") or []:
            per_state = [sum(v) / len(v) for r in rows for v in [r["screen"].get("d", {}).get(lb)] if v]
            if per_state:
                D[lb] = round(sum(per_state) / len(per_state), 5)
        if D:
            out["screen_episode"]["D"] = D
    out["predictions"] = P
    return out


# delay_graphs.sh: the delays of CUDA-graph execution (two steps 86 ms -> 2 control
# steps, ten steps 143 ms -> 3), reported without a prediction
GRAPHS = "delay_graphs_a6000"
DELAY_GRAPHS = {("idm", 3): f"{GRAPHS}/ld_idm_s1_stock10_d3.json",
                ("orig", 3): f"{GRAPHS}/ld_uncond_s5_stock10_d3.json",
                ("idm", 4): f"{GRAPHS}/ld_idm_s1_stock10_d4.json"}
GRAPH_REFS = {"idm": {"two_d2": DELAY[("two", 2)], "ten_d0": DELAY[("ten", 0)], "ten_d2": DELAY[("ten", 2)],
                      "ten_d5": DELAY[("ten", 5)]},
              "orig": {"two_d2": CACHE_DELAY["two_d2"], "ten_d0": "shift_a6000/uncond_s5_stock10.json"}}
SHIFT = {"idm": 1.0, "orig": 5.0}


def analyze_graphs(NEW, load, paired, rate):
    """The comparisons delay_graphs.sh states, on the 50 states, for each checkpoint."""
    NEW = Path(NEW)
    if not (NEW / GRAPHS).is_dir():
        return None
    runs = {k: load(NEW / rel) for k, rel in DELAY_GRAPHS.items()}
    for (ck, d_), r in runs.items():
        if r is not None:
            _meta_ok(r, DELAY_GRAPHS[(ck, d_)], configs=["stock"], steps=10, shift=SHIFT[ck], delay=(d_, d_))
    out = {}
    for ck, refs in GRAPH_REFS.items():
        r3 = runs[(ck, 3)]
        ref = {k: load(NEW / rel) for k, rel in refs.items()}
        if r3 is None or any(v is None for v in ref.values()):
            continue
        o = {"ten_d3": rate(r3), "two_d2": rate(ref["two_d2"]), "ten_d0": rate(ref["ten_d0"]),
             "two_d2_vs_ten_d3": paired(ref["two_d2"], r3, f"{ck}: two steps at delay 2 vs ten at delay 3"),
             "ten_d3_vs_ten_d0": paired(r3, ref["ten_d0"], f"{ck}: ten steps at delay 3 vs no delay")}
        if ck == "idm":
            o["ten_d3_vs_ten_d2"] = paired(r3, ref["ten_d2"], "idm: ten steps at delay 3 vs 2")
            r4 = runs[("idm", 4)]
            if r4 is not None:
                o["ten_d4"] = rate(r4)
                o["ten_d4_vs_ten_d2"] = paired(r4, ref["ten_d2"], "idm: ten steps at delay 4 vs 2")
            o["curve_ten"] = {d_: rate(r)["pct"] for d_, r in
                              ((0, ref["ten_d0"]), (2, ref["ten_d2"]), (3, r3), (4, r4), (5, ref["ten_d5"])) if r is not None}
        out[ck] = o
    return out


def main(argv):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import report_budget_shift as rb
    NEW = Path(argv[1]) if len(argv) > 1 else here.parent / "results/budget_shift"
    res = analyze(NEW, rb.load, rb.paired, rb.rate)
    print(json.dumps(res, indent=1, default=str) if res else f"no runs under {NEW / LD}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

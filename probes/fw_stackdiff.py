"""Does per-block substitution actually equal whole-stack substitution?

THE QUESTION THIS SETTLES. C$^3$ache records one residual across the whole
transformer stack and adds it once. Our cache records thirty and adds thirty.
They telescope in exact arithmetic -- `residual_x` is the block input, so
sum_b (x_{b+1} - x_b) = x_30 - x_0 -- and the repository has been citing that
identity as though it established equivalence. It does not: the model runs in
BF16, where thirty adds round thirty times and one add rounds once.
probes/test_cache_semantics.py shows the two forms differing by ~2e-3 on
synthetic tensors of the right shape while self-replay still passes exactly. So
the existing gate cannot exclude the difference, and a task-success gap against
the published number cannot be attributed to the method.

This measures it on the real model instead of arguing about it. No simulator and
no success sweep: fixed recorded observations, a cache filled at one cycle and
reused at a later one, and a direct comparison of what each substitution
produces -- against full compute, and against full compute run twice, which is
the noise floor any claimed difference has to clear.

ARMS, all reusing the SAME residuals recorded at the anchor cycle:
  full          no cache (the reference)
  full_repeat   the reference again -- noise floor
  perblock      30 residuals, substituted, block still computed and discarded
  perblock_skip 30 residuals, block genuinely not called
  wholestack    1 residual for the stack, added once (the published form)
  wholestack32  the same, accumulated in fp32 and rounded once

Reported per (episode, anchor, age, arm): max and RMS error in the final
denoise-step hidden state and in the emitted action, plus OBSERVED cache hits
and misses -- not the requested count, which is what gate_cache conflated.

usage (on a host prepared by setup/onstart_fw_families.sh):
  ./repro/run_experiment.sh stackdiff
"""
from __future__ import annotations

import importlib.util, inspect, json, os, sys, time
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import torch

_ld = torch.load
torch.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fw_libero_mod", os.path.join(_HERE, "fw_libero.py"))
_fw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fw)
CacheHarness, StackHarness = _fw.CacheHarness, _fw.StackHarness
placements, N_BLOCKS = _fw.placements, _fw.N_BLOCKS

sys.path.insert(0, _HERE)
from provenance import provenance                                  # noqa: E402
from stack_probe_common import (SCHEMA_VERSION, inference_runtime, load_probe_model,
                                requested_sigma_shift, tensor_errors)


def _err(a, b):
    """Direct max and RMS difference; never subtract distances to a reference."""
    return tensor_errors(a, b)


def _flat_tap(tap):
    """Concatenate the per-step hidden states so one number covers the schedule."""
    if not tap: return None
    return torch.cat([tap[k].reshape(-1).cpu() for k in sorted(tap)])


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pr = OmegaConf.merge(OmegaConf.create(
        {"manifest": "/workspace/episodes24_idm.json",
         "video_root": "/workspace/libero",
         "out": "/workspace/fw_stackdiff.json",
         "stride": 10, "max_cycles": 12, "n_episodes": 6,
         "ages": "1,4,7", "anchors": "0", "proprio_seed": 0,
         # Compilation is the other candidate the audit could not rule out, so it
         # is a dimension here rather than a fixed background condition.
         "compile": "0,1"}),
        cfg.get("PROBE", {}))

    steps = int(cfg.EVALUATION.get("num_inference_steps", 10))
    # EVALUATION.action_horizon is null in the shipped config, so .get(k, 32)
    # returns None rather than the default. The horizon is derived from the data
    # config, exactly as fw_libero.py does it.
    ah = int(cfg.data.train.num_frames) - 1
    # The frame size comes from the data config, not EVALUATION -- which has no
    # input_h/input_w at all, so a .get default here would silently decode at the
    # wrong resolution instead of failing.
    h, w = (int(v) for v in cfg.data.train.video_size)
    model, checkpoint_load = load_probe_model(cfg, instantiate)
    eps = json.load(open(str(pr.manifest)))[:int(pr.n_episodes)]
    # The manifest stores from_ts/fps/length, not frame indices; fw_families.py
    # derives the window the same way. Reading e["f0"] directly is a KeyError,
    # not a wrong number, so this fails loudly -- but only after the model has
    # loaded, which is three minutes in.
    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]

    _g = torch.Generator().manual_seed(int(pr.proprio_seed))
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim), generator=_g)
    sig = inspect.signature(model.infer_action).parameters

    def infer(img, prompt, compiled):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=42,
                  sigma_shift=requested_sigma_shift(cfg),
                  compile_action_infer=bool(compiled))
        if "num_video_frames" in sig:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in sig:
            kw["action_infer_mode"] = str(cfg.EVALUATION.get("action_infer_mode", "idm"))
        with torch.no_grad(): return model.infer_action(**kw)

    SH = StackHarness(model.mot)
    CH = CacheHarness(model.mot)
    dn = model._denoise_action_with_video_cache

    def traced(*a, **k):
        CH.begin()
        try: return dn(*a, **k)
        finally: CH.end()
    model._denoise_action_with_video_cache = traced

    cells = placements(None, steps)["c3ache"]
    stack_steps = {k for k, _ in cells}
    assert len(cells) == len(stack_steps) * N_BLOCKS, "cells must be whole steps"

    # reuse the replay decoder rather than a second copy of it
    from fw_families import decode_suite                            # noqa: E402

    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)

    ages = [int(x) for x in str(pr.ages).split(",") if x.strip()]
    anchors = [int(x) for x in str(pr.anchors).split(",") if x.strip()]
    compiles = [bool(int(x)) for x in str(pr.compile).split(",") if x.strip()]

    rows, comparisons, t0 = [], [], time.time()
    for ei, ep in enumerate(eps):
        fr = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"])
              if i in pool[ep["suite"]]]
        prompt = str(ep["task"]).strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(fr) // int(pr.stride))

        def image(c):
            return (torch.from_numpy(fr[c * int(pr.stride)]).permute(2, 0, 1)
                    .float().unsqueeze(0) / 127.5 - 1.0)

        for comp in compiles:
            for a0 in anchors:
                if a0 >= ncyc: continue
                # ---- fill the cache at the anchor cycle, both granularities ----
                blk, stk, tap0 = {}, {}, {}
                SH.reset(record=stk, steps=None, tap=tap0)
                CH.reset(record=blk); SH.on = True
                infer(image(a0), prompt, comp)
                SH.on = False
                if set(stk) != set(range(steps)) or not cells.issubset(blk):
                    raise RuntimeError(f"[stackdiff] ep{ei} anchor{a0}: recorded "
                          f"{len(stk)} stack / {len(blk)} block entries; expected "
                          f"{steps} / {steps * N_BLOCKS}")

                for age in ages:
                    c = a0 + age
                    if c >= ncyc: continue
                    img = image(c)

                    def run(label, setup):
                        SH.zero_counts()
                        tap = {}
                        setup(tap)
                        SH.on = True
                        act = infer(img, prompt, comp)
                        SH.on = False
                        CH.reset(); SH.reset()
                        a = act["action"] if isinstance(act, dict) else act
                        return label, _flat_tap(tap), a.detach().float().cpu(), SH.counts()

                    def _ref(tap):
                        SH.reset(steps=None, tap=tap); CH.reset()
                    res = [run("full", _ref), run("full_repeat", _ref)]

                    def _pb(skip):
                        def f(tap):
                            SH.reset(steps=None, tap=tap)
                            CH.reset(apply=blk, cells=cells, staleness=age, skip=skip)
                        return f
                    res.append(run("perblock", _pb(False)))
                    res.append(run("perblock_skip", _pb(True)))

                    def _ws(fp32):
                        def f(tap):
                            CH.reset()
                            SH.reset(apply=stk, steps=stack_steps, fp32=fp32, tap=tap)
                        return f
                    res.append(run("wholestack", _ws(False)))
                    res.append(run("wholestack32", _ws(True)))

                    # Compute the desired pairwise distances while the tensors
                    # still exist. Differences of distances to full compute are
                    # only lower bounds and cannot establish a small discrepancy.
                    outputs = {r[0]: r for r in res}
                    for left, right in (("perblock", "wholestack"),
                                        ("perblock_skip", "wholestack"),
                                        ("perblock", "perblock_skip"),
                                        ("wholestack", "wholestack32")):
                        lhs, rhs = outputs[left], outputs[right]
                        hmax, hrms = _err(lhs[1], rhs[1])
                        amax, arms = _err(lhs[2], rhs[2])
                        comparisons.append({
                            "episode": ei, "suite": ep["suite"], "anchor": a0,
                            "age": age, "compiled": comp, "left": left, "right": right,
                            "hidden_max": hmax, "hidden_rms": hrms,
                            "action_max": amax, "action_rms": arms})

                    ref_hid, ref_act = res[0][1], res[0][2]
                    for label, hid, act, counts in res:
                        if label.startswith("wholestack") and (
                                counts["hits"] != len(stack_steps) or counts["misses"]):
                            raise RuntimeError(f"Incomplete whole-stack substitution: {counts}")
                        hmax, hrms = _err(hid, ref_hid)
                        amax, arms = _err(act, ref_act)
                        rows.append({
                            "episode": ei, "suite": ep["suite"], "anchor": a0,
                            "age": age, "compiled": comp, "arm": label,
                            "hidden_max": hmax, "hidden_rms": hrms,
                            "action_max": amax, "action_rms": arms,
                            "stack_hits": counts["hits"], "stack_misses": counts["misses"]})
                    print(f"[stackdiff] ep{ei} anchor{a0} age{age} "
                          f"compiled={int(comp)}  " +
                          "  ".join(f"{r[0]}:{_err(r[2], ref_act)[0]:.2e}" for r in res[1:]),
                          flush=True)

    out = {"meta": {"schema_version": SCHEMA_VERSION, "collector": "stackdiff",
                    "checkpoint_load": checkpoint_load,
                    "runtime": {**inference_runtime(cfg, model),
                                "compile_action_infer": [int(c) for c in compiles]},
                    "manifest": os.path.basename(str(pr.manifest)),
                    "ckpt": os.path.basename(str(cfg.ckpt)),
                    "steps": steps, "stride": int(pr.stride),
                    "cached_steps": sorted(stack_steps), "n_blocks": N_BLOCKS,
                    "ages": ages, "anchors": anchors,
                    "compile_variants": [int(c) for c in compiles],
                    "inference_mode": inference_runtime(cfg, model)["action_infer_mode"],
                    "inference_seed": 42, "proprio_seed": int(pr.proprio_seed),
                    "n_episodes": len(eps),
                    "wholestack32_precision": "FP32 addition of a model-dtype recorded residual; model unchanged",
                    "provenance": provenance(ckpt=str(cfg.ckpt))},
           "rows": rows, "comparisons": comparisons}
    os.makedirs(os.path.dirname(str(pr.out)) or ".", exist_ok=True)
    json.dump(out, open(str(pr.out), "w"), indent=1)
    print(f"\n[stackdiff] {len(rows)} rows -> {pr.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()

"""Fit the attenuation weight for a WHOLE-STACK residual, on the training split.

The per-cell coefficients in inputs/weights/shrink_idm.json are fit on 18 of 24
replay episodes with six held out. A whole-stack cache has no cells to index, so
its weight is one scalar per cache age, and it has to be fit the SAME way or the
comparison between granularities is a comparison of calibration budgets.

Two rules this enforces rather than documents:
  * zero-based episodes 0, 4, 8, ... are held out, exactly as the per-cell fit
    does, and the training coefficient is scored on that held-out split;
  * evaluation rollouts are never read. This probe takes a replay manifest and
    recorded video, and there is no code path here that opens a results file.

The weight at lag k is Equation 1 at stack granularity, pooled over the training
pairs: beta_k = sum <T_i, C_i> / sum ||C_i||^2, with T the residual at cycle c
and C the residual at cycle c-k.

usage: ./repro/run_experiment.sh stackbeta
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
StackHarness, placements = _fw.StackHarness, _fw.placements

sys.path.insert(0, _HERE)
from provenance import provenance                                  # noqa: E402
from stack_probe_common import (SCHEMA_VERSION, add_step_stats, episode_split,
                                fitted_beta, inference_runtime, load_probe_model, new_stats,
                                requested_sigma_shift, score_beta)


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pr = OmegaConf.merge(OmegaConf.create(
        {"manifest": "/workspace/episodes24_idm.json",
         "video_root": "/workspace/libero",
         "out": "/workspace/stack_shrink.json",
         "stride": 10, "max_cycles": 16, "max_age": 7, "proprio_seed": 0,
         "holdout_every": 4}),
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
    cached_steps = sorted({k for k, _ in placements(None, steps)["c3ache"]})
    if not cached_steps:
        raise ValueError("No cacheable denoising steps: a whole-stack fit is undefined.")
    model, checkpoint_load = load_probe_model(cfg, instantiate)
    eps = json.load(open(str(pr.manifest)))
    # from_ts/fps/length -> frame window, as fw_families.py does it
    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]

    _g = torch.Generator().manual_seed(int(pr.proprio_seed))
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim), generator=_g)
    sig = inspect.signature(model.infer_action).parameters

    def infer(img, prompt):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=42,
                  sigma_shift=requested_sigma_shift(cfg),
                  compile_action_infer=False)
        if "num_video_frames" in sig:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in sig:
            kw["action_infer_mode"] = str(cfg.EVALUATION.get("action_infer_mode", "idm"))
        with torch.no_grad(): return model.infer_action(**kw)

    SH = StackHarness(model.mot); SH.on = True
    from fw_families import decode_suite                            # noqa: E402

    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)

    maxk = int(pr.max_age)
    # Store all sufficient statistics, including the target norm needed to score
    # a coefficient fitted on training data against held-out reconstruction MSE.
    acc = {sp: {k: new_stats() for k in range(1, maxk + 1)} for sp in ("train", "held")}
    t0 = time.time()

    for ei, ep in enumerate(eps):
        split = episode_split(ei, int(pr.holdout_every))
        fr = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"])
              if i in pool[ep["suite"]]]
        prompt = str(ep["task"]).strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(fr) // int(pr.stride))
        hist = []                        # per cycle: {denoise step -> residual}
        for c in range(ncyc):
            img = (torch.from_numpy(fr[c * int(pr.stride)]).permute(2, 0, 1)
                   .float().unsqueeze(0) / 127.5 - 1.0)
            rec = {}
            SH.reset(record=rec, steps=None)
            infer(img, prompt)
            if not set(cached_steps).issubset(rec):
                raise RuntimeError("Calibration failed to record every cacheable step.")
            hist.append({k: v.float().cpu() for k, v in rec.items()})
            for k in range(1, maxk + 1):
                if c - k < 0: continue
                T, C = hist[c], hist[c - k]
                add_step_stats(acc[split][k], T, C, cached_steps)
            if len(hist) > maxk + 1: hist[c - maxk - 1] = {}
        print(f"[stackbeta] ep{ei} ({split}) {ncyc} cycles  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

    def betas(sp):
        return {str(k): fitted_beta(acc[sp][k]) for k in range(1, maxk + 1)}

    tr = betas("train")
    if any(v is None for v in tr.values()):
        raise RuntimeError("Insufficient training data for one or more requested cache ages.")
    held_scores = {str(k): score_beta(acc["held"][k], tr[str(k)])
                   for k in range(1, maxk + 1)}
    if any(v is None for v in held_scores.values()):
        raise RuntimeError("Insufficient held-out data to validate each requested age.")
    out = {"beta_by_age": {k: v for k, v in tr.items() if v is not None},
           "beta": tr.get("1"),
           "fit_on": "train",
           "held_out_refit_beta_by_age": betas("held"),
           "held_out_scores": held_scores,
           "sufficient_statistics": acc,
           "pairs": {sp: {k: s["n_pairs"] for k, s in acc[sp].items()} for sp in acc},
           "episodes_total": len(eps),
           "episodes_train": sum(1 for i in range(len(eps))
                                 if episode_split(i, int(pr.holdout_every)) == "train"),
           "episode_ids": {sp: [i for i in range(len(eps))
                                 if episode_split(i, int(pr.holdout_every)) == sp]
                           for sp in ("train", "held")},
           "holdout_every": int(pr.holdout_every),
           "granularity": "whole-stack (h_L - h_0 per denoise step)",
           "meta": {"schema_version": SCHEMA_VERSION, "collector": "stackbeta",
                    "checkpoint_load": checkpoint_load,
                    "runtime": {**inference_runtime(cfg, model), "compile_action_infer": False},
                    "manifest": os.path.basename(str(pr.manifest)),
                    "ckpt": os.path.basename(str(cfg.ckpt)),
                    "steps": steps, "cached_steps": cached_steps, "stride": int(pr.stride),
                    "inference_mode": inference_runtime(cfg, model)["action_infer_mode"],
                    "inference_seed": 42, "proprio_seed": int(pr.proprio_seed),
                    "compiled": False,
                    "provenance": provenance(ckpt=str(cfg.ckpt))}}
    os.makedirs(os.path.dirname(str(pr.out)) or ".", exist_ok=True)
    json.dump(out, open(str(pr.out), "w"), indent=1)
    print(f"\n[stackbeta] train betas {out['beta_by_age']}")
    print(f"[stackbeta] held-out scores with TRAIN coefficients {held_scores}")
    print(f"[stackbeta] -> {pr.out}")


if __name__ == "__main__":
    main()

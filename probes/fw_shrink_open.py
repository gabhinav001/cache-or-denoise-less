"""Does the residual-level gain survive into the action?

The trace study says the least-squares weight on a cached residual is ~0.77, not
1. That minimises residual error, but it is not obviously the right thing to do
to the network: attenuating 240 of 300 substituted residuals moves every one of
them toward zero, i.e. toward the block contributing nothing at all, and those
errors could accumulate coherently through depth even as each individual one
shrinks.

So this scores the arms on the model's actual output rather than on its
internals. At one placement and one budget, identical in every respect except
the substitution weight:

    hold    R_hat = R[c-1]           (the published rule)
    shrink  R_hat = beta[cell] R[c-1] (weights fit on other episodes)

reporting relative L2 of each arm's action chunk against the full-compute chunk
for the same observation. Paired: both arms see the same cycle, same cache, same
seed, so the difference is the substitution rule and nothing else.
"""
from __future__ import annotations

import inspect, json, os, time
import numpy as np, torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

N_BLOCKS, N_STEPS = 30, 10


class Harness:
    def __init__(self, mot):
        self.mot = mot; self.orig = mot._apply_expert_post_block_tensor
        self.on = False; self.step = -1; self.layer = 0
        self.record = self.apply = self.cells = self.scale = None
        mot._apply_expert_post_block_tensor = self._wrap(self.orig)

    def _wrap(self, fn):
        def w(*a, **k):
            out = fn(*a, **k)
            if not self.on: return out
            key = (self.step, self.layer); self.layer += 1
            rx = k.get("residual_x")
            if rx is None: return out
            if self.record is not None: self.record[key] = (out - rx).detach().clone()
            if self.cells is not None and key in self.cells:
                c = self.apply.get(key) if self.apply else None
                if c is not None and c.shape == rx.shape:
                    s = self.scale.get(key, 1.0) if self.scale else 1.0
                    return rx + (c if s == 1.0 else s * c)
            return out
        return w

    def begin(self): self.step += 1; self.layer = 0; self.on = True
    def end(self): self.on = False
    def reset(self, record=None, apply=None, cells=None, scale=None):
        self.step = -1; self.layer = 0
        self.record, self.apply, self.cells, self.scale = record, apply, cells, scale
    def close(self): self.mot._apply_expert_post_block_tensor = self.orig


def rel_l2(a, b):
    n = torch.linalg.vector_norm(b.float())
    return (torch.linalg.vector_norm(a.float() - b.float()) / n).item() if n > 0 else float("nan")


def decode_suite(main, wrist, ranges, h, w):
    import imageio.v3 as iio, cv2
    half, last = w // 2, max(f1 for _, f1 in ranges)
    want = set()
    for f0, f1 in ranges: want |= set(range(f0, f1))
    def grab(p):
        out = {}
        for i, f in enumerate(iio.imiter(p)):
            if i >= last: break
            if i in want: out[i] = cv2.resize(f, (half, h), interpolation=cv2.INTER_AREA)
        return out
    a, b = grab(main), grab(wrist)
    return {i: np.concatenate([a[i], b[i]], axis=1) for i in sorted(a) if i in b}


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pr = OmegaConf.merge(OmegaConf.create(
        {"manifest": "/workspace/episodes50.json", "video_root": "/workspace/libero",
         "shrink": "/workspace/shrink.json", "out": "/workspace/fw_shrink_open.json",
         "stride": 10, "max_cycles": 20, "skip_steps": 8}), cfg.get("PROBE", {}))
    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    model.load_checkpoint(str(cfg.ckpt)); model.eval()
    h, w = (int(v) for v in cfg.data.train.video_size)
    ah, steps = int(cfg.data.train.num_frames) - 1, int(cfg.eval_num_inference_steps)

    d = json.load(open(str(pr.shrink)))
    SC = {(int(k.split(",")[0]), int(k.split(",")[1])): float(v[0]) for k, v in d["coef"].items()}
    fit_eps = set(d.get("fit_episodes", []))
    cells = {(k, b) for k in range(int(pr.skip_steps)) for b in range(N_BLOCKS)}
    print(f"[shrink] {len(cells)} of {N_STEPS*N_BLOCKS} cells cached | {len(SC)} coefficients "
          f"| median {sorted(SC.values())[len(SC)//2]:.3f}", flush=True)

    eps = json.load(open(str(pr.manifest)))
    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]
    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)
        print(f"[shrink] decoded {suite[7:14]}: {len(pool[suite])} frames", flush=True)

    H = Harness(model.mot)
    dn = model._denoise_action_with_video_cache
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    model._denoise_action_with_video_cache = traced
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))
    sig = inspect.signature(model.infer_action).parameters

    def infer(img, prompt):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=42,
                  compile_action_infer=False)
        if "num_video_frames" in sig:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in sig: kw["action_infer_mode"] = "idm"
        with torch.no_grad(): return model.infer_action(**kw)["action"]

    rows, t0 = [], time.time()
    for ei, ep in enumerate(eps):
        frames = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"]) if i in pool[ep["suite"]]]
        prompt = str(ep["task"]).strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(frames) // int(pr.stride))
        prev = None
        for c in range(ncyc):
            img = torch.from_numpy(frames[c * int(pr.stride)]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            fresh = {}; H.reset(record=fresh)
            a_full = infer(img, prompt)
            if prev is not None:
                H.reset(apply=prev, cells=cells)
                a_hold = infer(img, prompt)
                H.reset(apply=prev, cells=cells, scale=SC)
                a_shrink = infer(img, prompt)
                rows.append({"ep": ei, "suite": ep["suite"], "episode": int(ep["episode"]),
                             "cycle": c, "in_fit": ei in fit_eps,
                             "hold": rel_l2(a_hold, a_full),
                             "shrink": rel_l2(a_shrink, a_full)})
            prev = fresh
        held = [r for r in rows if not r["in_fit"]]
        if held:
            mh = float(np.mean([r["hold"] for r in held])); ms = float(np.mean([r["shrink"] for r in held]))
            win = 100 * np.mean([r["shrink"] < r["hold"] for r in held])
            print(f"[shrink] ep {ei+1}/{len(eps)} | held-out n={len(held)} "
                  f"hold {mh:.4f} shrink {ms:.4f} ({100*(1-ms/mh):+.1f}%) shrink wins {win:.0f}% "
                  f"| {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump({"meta": {"gpu": torch.cuda.get_device_name(0), "cells": len(cells),
                            "stride": int(pr.stride), "skip_steps": int(pr.skip_steps),
                            "shrink_median": sorted(SC.values())[len(SC)//2],
                            "fit_episodes": sorted(fit_eps)}, "rows": rows},
                  open(str(pr.out), "w"))
    model._denoise_action_with_video_cache = dn; H.close()


if __name__ == "__main__":
    main()

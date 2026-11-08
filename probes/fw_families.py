"""How much is left in the predictor families the alpha sweep did not close?

The alpha result bounds one family: R[c-1] + a(R[c-1]-R[c-2]), 2.9% over a flat
hold with oracle per-cell a. It says nothing about more taps, mean reversion to
a level, EMAs, per-channel rescaling, or conditioning on the observation. This
records the sufficient statistics for all of them at once (see famlib), over 52
LIBERO episodes spanning 15 tasks and all four suites -- against 8 episodes of
one suite in the original trace.

Everything here runs at full compute. No cache is ever applied, so the result is
a property of the model's own residuals and is independent of the cache
reimplementation whose fidelity is still open.
"""
from __future__ import annotations

import inspect, json, os, time
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from famlib import MultiRecorder, HIST, NCOL
from provenance import provenance


class Harness:
    """mot.py inlines the block body, so no module hook fires; wrap the post-block
    call instead, whose residual_x kwarg is exactly the block input."""

    def __init__(self, mot):
        self.mot = mot; self.orig = mot._apply_expert_post_block_tensor
        self.on = False; self.step = -1; self.layer = 0; self.cur = {}
        mot._apply_expert_post_block_tensor = self._wrap(self.orig)

    def _wrap(self, fn):
        def w(*a, **k):
            out = fn(*a, **k)
            if self.on:
                rx = k.get("residual_x")
                if rx is not None:
                    self.cur[(self.step, self.layer)] = (out - rx).detach()
                self.layer += 1
            return out
        return w

    def begin(self): self.step += 1; self.layer = 0; self.on = True
    def end(self): self.on = False
    def new_cycle(self): self.step = -1; self.layer = 0; self.cur = {}
    def close(self): self.mot._apply_expert_post_block_tensor = self.orig


def decode_suite(main, wrist, ranges, h, w):
    """One sequential pass per video collecting every requested range; episodes sit
    tens of thousands of frames in, so per-episode seeking would dominate."""
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
         "out": "/workspace/fw_families.npz", "stride": 10, "max_cycles": 32,
         "proprio_seed": 0}),
        cfg.get("PROBE", {}))
    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    model.load_checkpoint(str(cfg.ckpt)); model.eval()

    h, w = (int(v) for v in cfg.data.train.video_size)
    ah, steps = int(cfg.data.train.num_frames) - 1, int(cfg.eval_num_inference_steps)
    eps = json.load(open(str(pr.manifest)))
    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]
    print(f"[fam] {len(eps)} episodes, {len({e['task'] for e in eps})} tasks, "
          f"{steps} steps x {len(model.mot.mixtures['action'].blocks)} blocks", flush=True)

    H = Harness(model.mot)
    dn = model._denoise_action_with_video_cache
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    model._denoise_action_with_video_cache = traced
    # seeded: an unseeded draw is constant within a run but differs across runs,
    # which silently makes the study irreproducible. PROBE.proprio_seed varies it
    # deliberately when testing sensitivity to this input.
    _g = torch.Generator().manual_seed(int(pr.proprio_seed))
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim), generator=_g)
    sig = inspect.signature(model.infer_action).parameters

    def infer(img, prompt):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=42,
                  compile_action_infer=False)
        if "num_video_frames" in sig:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in sig: kw["action_infer_mode"] = "idm"
        with torch.no_grad(): return model.infer_action(**kw)

    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        t0 = time.time()
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)
        print(f"[fam] decoded {suite[7:14]}: {len(pool[suite])} frames in "
              f"{time.time()-t0:.0f}s", flush=True)

    acc = {k: [] for k in ("gram", "chan1", "tok1", "chan2", "ep", "cycle",
                           "step", "block", "obs_delta")}
    t0 = time.time()
    for ei, ep in enumerate(eps):
        frames = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"])
                  if i in pool[ep["suite"]]]
        prompt = str(ep["task"]).strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(frames) // int(pr.stride))
        rec, prev_img, kept = MultiRecorder(dev), None, 0
        for c in range(ncyc):
            fr = frames[c * int(pr.stride)].astype(np.float32)
            d = float(np.linalg.norm(fr - prev_img) / max(np.linalg.norm(prev_img), 1e-6)) \
                if prev_img is not None else float("nan")
            prev_img = fr
            img = torch.from_numpy(frames[c * int(pr.stride)]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            H.new_cycle(); infer(img, prompt)
            for shape, (keys, r) in rec.push(H.cur).items():
                n = len(keys)
                acc["gram"].append(r["gram"])
                for k in ("chan1", "tok1", "chan2"): acc[k].append(r[k])
                acc["ep"].append(np.full(n, ei, np.int16))
                acc["cycle"].append(np.full(n, c, np.int16))
                acc["step"].append(np.array([k[0] for k in keys], np.int8))
                acc["block"].append(np.array([k[1] for k in keys], np.int8))
                acc["obs_delta"].append(np.full(n, d, np.float32))
                kept += n
        el = (time.time() - t0) / 60
        print(f"[fam] ep {ei+1}/{len(eps)} {ep['suite'][7:14]} #{ep['episode']} "
              f"{ncyc}cyc -> {kept} cells | {el:.1f} min | \"{prompt[:38]}\"", flush=True)
        np.savez_compressed(str(pr.out),
            meta=np.array([json.dumps({"gpu": torch.cuda.get_device_name(0),
                "episodes": eps[:ei+1], "stride": int(pr.stride), "hist": HIST,
                "ncol": NCOL, "steps": steps, "seed": 42,
                "ckpt": os.path.basename(str(cfg.ckpt)),
                "manifest": os.path.basename(str(pr.manifest)),
                # WHICH CODE PRODUCED THIS. Two collections of this experiment
                # agreed on every recorded setting -- checkpoint, manifest,
                # episode windows, stride, hist, seed, cells, cycles -- and still
                # disagreed on the residuals by 9% at the median. Nothing in the
                # artefact could say why, because the model version was never
                # written down. It is now.
                "provenance": provenance(ckpt=str(cfg.ckpt)),
                "note": "full compute throughout; no cache applied"})]),
            **{k: np.concatenate(v) for k, v in acc.items() if v})
    model._denoise_action_with_video_cache = dn; H.close()
    print(f"[fam] wrote {sum(len(x) for x in acc['ep'])} cell-observations", flush=True)


if __name__ == "__main__":
    main()

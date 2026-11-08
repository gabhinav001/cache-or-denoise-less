"""Scaled replication: 10 LIBERO episodes, 6 tasks, 2 suites, with repeats.

Runs both headline studies per episode so the model loads once:

  A. placement -- five cache placements at a matched budget of 240 of 300
     skipped block evaluations, repeated R times per episode so the
     bfloat16/fused-attention nondeterminism becomes an error bar rather than
     an unmeasured confound.

  B. forecast  -- zero-order hold vs damped first-order extrapolation, scored
     on held-out cycles.

Each episode uses its own real task instruction from the dataset metadata
rather than a fixed prompt, so the language conditioning varies as it would in
evaluation.
"""
from __future__ import annotations

import inspect, json, os, random, statistics as st, time
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

N_BLOCKS, N_STEPS, SKIP = 30, 10, 240
ALPHAS = [0.25, 0.5, 0.75, 1.0]


class Harness:
    """Records fresh residuals and/or substitutes cached ones, per (step, layer)."""
    def __init__(self, mot):
        self.mot = mot; self.orig = mot._apply_expert_post_block_tensor
        self.on = False; self.step = -1; self.layer = 0
        self.record = None; self.apply = None; self.cells = None
        mot._apply_expert_post_block_tensor = self._wrap(self.orig)

    def _wrap(self, fn):
        def w(*a, **k):
            out = fn(*a, **k)
            if not self.on: return out
            key = (self.step, self.layer); self.layer += 1
            rx = k.get("residual_x")
            if rx is None: return out
            if self.record is not None:
                self.record[key] = (out - rx).detach().clone()
            if self.cells is not None and key in self.cells:
                c = self.apply.get(key) if self.apply else None
                if c is not None and c.shape == rx.shape: return rx + c
            return out
        return w

    def begin(self): self.step += 1; self.layer = 0; self.on = True
    def end(self): self.on = False
    def reset(self, record=None, apply=None, cells=None):
        self.step = -1; self.layer = 0
        self.record, self.apply, self.cells = record, apply, cells
    def close(self): self.mot._apply_expert_post_block_tensor = self.orig


def placements(oracle_path, seed=0):
    rng = random.Random(seed)
    allc = [(k, b) for k in range(N_STEPS) for b in range(N_BLOCKS)]
    P = {"c3ache": {(k, b) for k in range(8) for b in range(N_BLOCKS)},
         "depth":  {(k, b) for k in range(N_STEPS) for b in range(24)},
         "random": set(rng.sample(allc, SKIP)),
         "late":   {(k, b) for k in range(2, N_STEPS) for b in range(N_BLOCKS)}}
    # The oracle placement needs a cell set computed elsewhere. If it is absent
    # this silently produced four placements where the shipped artefact has five,
    # and the analysis then failed on the missing column -- a run that looks
    # successful and is not comparable. Say so instead.
    if oracle_path and os.path.exists(oracle_path):
        cells = {tuple(x) for x in json.load(open(oracle_path))["oracle"]}
        if len(cells) == SKIP:
            P["oracle"] = cells
        else:
            print(f"[scale] WARNING: {oracle_path} has {len(cells)} cells, expected "
                  f"{SKIP}; oracle placement omitted", flush=True)
    else:
        print(f"[scale] WARNING: no oracle cell set at {oracle_path}. Collecting FOUR "
              f"placements; the shipped artefact has five. The oracle arm is not "
              f"reproducible from this repository -- see repro/experiments/placement.sh.",
              flush=True)
    for n, s in P.items(): assert len(s) == SKIP, (n, len(s))
    return P


def rel_l2(a, b):
    n = torch.linalg.vector_norm(b)
    return (torch.linalg.vector_norm(a - b) / n).item() if n > 0 else float("nan")


def cos(a, b):
    x, y = a.flatten().float(), b.flatten().float()
    nx, ny = x.norm(), y.norm()
    return (torch.dot(x, y) / (nx * ny)).item() if nx > 0 and ny > 0 else float("nan")


def decode_suite(main, wrist, ranges, h, w):
    """One sequential pass per video, collecting every requested frame range.

    Episodes sit far into the concatenated file (index 200+ is ~36k frames in),
    so decoding from the start once per episode would dominate runtime.
    """
    import imageio.v3 as iio, cv2
    half = w // 2
    last = max(f1 for _, f1 in ranges)
    want = {}
    for f0, f1 in ranges: want.update({i: None for i in range(f0, f1)})
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
        {"manifest": "/workspace/episodes.json", "video_root": "/workspace/libero",
         "oracle": "/workspace/oracle_cells.json", "out": "/workspace/fw_scale.json",
         "stride": 10, "repeats": 3, "max_cycles": 18}), cfg.get("PROBE", {}))
    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    model.load_checkpoint(str(cfg.ckpt)); model.eval()

    h, w = (int(v) for v in cfg.data.train.video_size)
    ah, steps = int(cfg.data.train.num_frames) - 1, int(cfg.eval_num_inference_steps)
    eps = json.load(open(str(pr.manifest)))
    PL = placements(str(pr.oracle))
    print(f"[scale] {len(eps)} episodes | placements {sorted(PL)} | "
          f"repeats={int(pr.repeats)} stride={int(pr.stride)}", flush=True)

    H = Harness(model.mot)
    dn = model._denoise_action_with_video_cache
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    model._denoise_action_with_video_cache = traced
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))

    def infer(img, prompt, seed):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=seed,
                  compile_action_infer=False)
        p = inspect.signature(model.infer_action).parameters
        if "num_video_frames" in p:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in p: kw["action_infer_mode"] = "idm"
        with torch.no_grad(): return model.infer_action(**kw)["action"]

    # decode each suite's video exactly once
    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]
    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        t0 = time.time()
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)
        print(f"[scale] decoded {suite[7:14]}: {len(pool[suite])} frames "
              f"in {time.time()-t0:.0f}s", flush=True)

    place_rows, fc_rows = [], []
    t_start = time.time()
    for ei, ep in enumerate(eps):
        frames = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"])
                  if i in pool[ep["suite"]]]
        prompt = ep["task"].strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(frames) // int(pr.stride))
        print(f"[scale] ep {ei+1}/{len(eps)} {ep['suite'][7:14]} #{ep['episode']} "
              f"{len(frames)}f -> {ncyc} cycles | \"{prompt[:44]}\"", flush=True)

        for rep in range(int(pr.repeats)):
            prev = prev2 = None
            for c in range(ncyc):
                img = torch.from_numpy(frames[c * int(pr.stride)]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                fresh = {}
                H.reset(record=fresh)
                a_full = infer(img, prompt, 42)
                if prev is not None:
                    row = {"ep": ei, "suite": ep["suite"], "episode": ep["episode"],
                           "rep": rep, "cycle": c}
                    for name, cells in PL.items():
                        H.reset(apply=prev, cells=cells)
                        row[name] = rel_l2(infer(img, prompt, 42), a_full)
                    place_rows.append(row)
                    if rep == 0:                      # forecast study once per episode
                        for key, true in fresh.items():
                            p1 = prev.get(key)
                            p2 = prev2.get(key) if prev2 else None
                            if p1 is None or p2 is None or p1.shape != true.shape: continue
                            k_, b_ = key
                            r = {"ep": ei, "cycle": c, "step": k_, "block": b_,
                                 "zero": rel_l2(p1, true), "cos": cos(p1, true)}
                            for al in ALPHAS: r[f"a{al}"] = rel_l2(p1 + al * (p1 - p2), true)
                            fc_rows.append(r)
                prev2 = prev if c > 0 else None
                prev = fresh
            el = time.time() - t_start
            print(f"[scale]   rep {rep+1}/{int(pr.repeats)} done, {len(place_rows)} rows, "
                  f"{el/60:.1f} min elapsed", flush=True)

        json.dump({"meta": {"gpu": torch.cuda.get_device_name(0), "episodes": eps,
                            "placements": sorted(PL), "repeats": int(pr.repeats),
                            "stride": int(pr.stride), "skip_cells": SKIP, "alphas": ALPHAS},
                   "placement": place_rows, "forecast": fc_rows},
                  open(str(pr.out), "w"))

    model._denoise_action_with_video_cache = dn; H.close()
    print(f"[scale] DONE {len(place_rows)} placement rows, {len(fc_rows)} forecast rows, "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Does the placement ranking survive a stale cache?

Every placement result so far uses a one-cycle-old cache -- the most favourable
setting that exists. The published LIBERO configuration is tau=0, i.e. no
refresh at all, so its cache age grows without bound across an episode, and the
same paper reports 34% degradation past 12 chunks. This sweeps cache age over
{1,2,4,8} for three placements.

The mechanism claim makes a prediction: if what matters is leaving the final
denoise steps free to correct against the current observation, then placements
that keep steps 8-9 fresh should degrade more gracefully as the cache goes
stale, and the advantage should widen with age.

Repeats are omitted deliberately: within a single process this pipeline is
bit-reproducible (verified over 120 episode/cycle pairs, max difference 0), so
repeating inside one run measures nothing.
"""
from __future__ import annotations

import inspect, json, math, os, random, statistics as st, time
import hydra, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

N_BLOCKS, N_STEPS, SKIP = 30, 10, 240
AGES = [1, 2, 4, 8]


class Harness:
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
            if self.record is not None: self.record[key] = (out - rx).detach().clone()
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


def placements(oracle_path):
    rng = random.Random(0)
    allc = [(k, b) for k in range(N_STEPS) for b in range(N_BLOCKS)]
    P = {"c3ache": {(k, b) for k in range(8) for b in range(N_BLOCKS)},
         "random": set(rng.sample(allc, SKIP))}
    if oracle_path and os.path.exists(oracle_path):
        cells = {tuple(x) for x in json.load(open(oracle_path))["oracle"]}
        if len(cells) == SKIP: P["oracle"] = cells
    for n, s in P.items(): assert len(s) == SKIP, (n, len(s))
    return P


def rel_l2(a, b):
    n = torch.linalg.vector_norm(b)
    return (torch.linalg.vector_norm(a - b) / n).item() if n > 0 else float("nan")


def decode_suite(main, wrist, ranges, h, w):
    import imageio.v3 as iio, cv2
    half = w // 2; last = max(f1 for _, f1 in ranges)
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
        {"manifest": "/workspace/episodes.json", "video_root": "/workspace/libero",
         "oracle": "/workspace/oracle_cells.json", "out": "/workspace/fw_age.json",
         "stride": 10, "max_cycles": 16}), cfg.get("PROBE", {}))
    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    model.load_checkpoint(str(cfg.ckpt)); model.eval()

    h, w = (int(v) for v in cfg.data.train.video_size)
    ah, steps = int(cfg.data.train.num_frames) - 1, int(cfg.eval_num_inference_steps)
    eps = json.load(open(str(pr.manifest)))
    PL = placements(str(pr.oracle))
    print(f"[age] placements {sorted(PL)} | ages {AGES}", flush=True)

    for e in eps:
        e["f0"] = int(round(e["from_ts"] * e["fps"])); e["f1"] = e["f0"] + e["length"]
    pool = {}
    for suite in sorted({e["suite"] for e in eps}):
        rs = [(e["f0"], e["f1"]) for e in eps if e["suite"] == suite]
        base = os.path.join(str(pr.video_root), suite)
        pool[suite] = decode_suite(f"{base}/main.mp4", f"{base}/wrist.mp4", rs, h, w)
        print(f"[age] decoded {suite[7:14]}: {len(pool[suite])} frames", flush=True)

    H = Harness(model.mot)
    dn = model._denoise_action_with_video_cache
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    model._denoise_action_with_video_cache = traced
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))

    def infer(img, prompt):
        kw = dict(prompt=prompt, input_image=img, action_horizon=ah, proprio=proprio,
                  num_inference_steps=steps, rand_device="cpu", seed=42,
                  compile_action_infer=False)
        p = inspect.signature(model.infer_action).parameters
        if "num_video_frames" in p:
            kw["num_video_frames"] = ah // int(cfg.data.train.action_video_freq_ratio) + 1
        if "action_infer_mode" in p: kw["action_infer_mode"] = "idm"
        with torch.no_grad(): return model.infer_action(**kw)["action"]

    rows = []; t0 = time.time()
    for ei, ep in enumerate(eps):
        frames = [pool[ep["suite"]][i] for i in range(ep["f0"], ep["f1"]) if i in pool[ep["suite"]]]
        if not frames: continue
        prompt = ep["task"].strip("[]'\" ")
        ncyc = min(int(pr.max_cycles), len(frames) // int(pr.stride))
        hist = []                                    # full-compute residuals, newest last
        for c in range(ncyc):
            img = torch.from_numpy(frames[c * int(pr.stride)]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            fresh = {}
            H.reset(record=fresh)
            a_full = infer(img, prompt)
            for age in AGES:
                if len(hist) < age: continue
                src = hist[-age]
                for name, cells in PL.items():
                    H.reset(apply=src, cells=cells)
                    rows.append({"ep": ei, "cycle": c, "age": age, "placement": name,
                                 "dev": rel_l2(infer(img, prompt), a_full)})
            hist.append(fresh)
            if len(hist) > max(AGES): hist.pop(0)
        print(f"[age] ep {ei+1}/{len(eps)} done, {len(rows)} rows, {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump({"meta": {"gpu": torch.cuda.get_device_name(0), "ages": AGES,
                            "placements": sorted(PL), "skip_cells": SKIP,
                            "stride": int(pr.stride), "episodes": eps}, "rows": rows},
                  open(str(pr.out), "w"))

    model._denoise_action_with_video_cache = dn; H.close()
    print("\n=== deviation by cache age ===")
    for age in AGES:
        line = f"  age {age}: "
        for name in sorted(PL):
            v = [r["dev"] for r in rows if r["age"] == age and r["placement"] == name]
            if v: line += f"{name}={st.mean(v):.4f}  "
        print(line, flush=True)
    print(f"[age] DONE {len(rows)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

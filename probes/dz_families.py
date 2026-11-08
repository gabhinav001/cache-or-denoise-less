"""The same family bound on DreamZero: 20 DROID episodes instead of one.

Fast-WAM and DreamZero are both Wan-derived, but DreamZero is 14B, carries real
persistent KV across cycles, and denoises video tokens rather than 32 action
tokens. If mean reversion is a property of denoising residuals rather than a
quirk of one checkpoint, it should appear here too.

Observations come from GEAR-Dreams/DreamZero-DROID-Data, the authors' own
training set, so the replayed footage is in distribution. Twenty episodes are
drawn evenly across the index range, hence across collection sessions and
scenes, rather than from one contiguous block.

Scale forces two subsamplings that Fast-WAM did not need. Video residuals are
~3.6k tokens x 5120 channels, so an eight-cycle history over all 640 cells would
not fit: this tracks every 8th block at every 2nd denoise step (40 cells) and a
fixed random 256 tokens per cell. The token subset is drawn once and reused for
every cycle and every predictor, so all families are compared on identical
coordinates and the comparison between them is exact -- it is an analysis of a
sub-tensor, not an estimate of the whole one.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
import torch

from famlib import MultiRecorder, HIST, NCOL

CKPT = os.environ.get("DZ_CKPT", "/workspace/ckpt")
EPS_JSON = os.environ.get("DZ_EPISODES", "/workspace/dz_episodes20.json")
VIDROOT = os.environ.get("DZ_VIDROOT", "/workspace/droid")
OUT = os.environ.get("DZ_OUT", "/workspace/dz_families.npz")
CYCLES = int(os.environ.get("DZ_CYCLES", "28"))
BLOCK_STRIDE = int(os.environ.get("DZ_BLOCK_STRIDE", "8"))
STEP_STRIDE = int(os.environ.get("DZ_STEP_STRIDE", "2"))
NTOK = int(os.environ.get("DZ_NTOK", "256"))
FRAMES_PER_CHUNK = 4


class Tracer:
    """Captures R[c,k,b] = h_out - h_in for a subsampled grid of cells."""

    def __init__(self, blocks, bstride, sstride, ntok, seed=0):
        self.tracked = list(range(0, len(blocks), bstride))
        self.first, self.sstride, self.ntok = self.tracked[0], sstride, ntok
        self.step, self.cur, self.enabled = -1, {}, False
        self.tok_idx: dict[int, torch.Tensor] = {}
        self.gen = torch.Generator().manual_seed(seed)
        self.handles = [blocks[b].register_forward_hook(self._mk(b), with_kwargs=True)
                        for b in self.tracked]

    def _mk(self, bi):
        def hook(module, args, kwargs, output):
            if not self.enabled: return
            if bi == self.first: self.step += 1
            if self.step % self.sstride: return
            x_in = kwargs.get("x", args[0] if args else None)
            if x_in is None: return
            x_out = output[0] if isinstance(output, tuple) else output
            r = (x_out - x_in).detach().reshape(-1, x_out.shape[-1])
            n = r.shape[0]
            if n > self.ntok:
                idx = self.tok_idx.get(n)
                if idx is None:                       # drawn once, reused every cycle
                    idx = torch.randperm(n, generator=self.gen)[:self.ntok].sort().values
                    self.tok_idx[n] = idx
                r = r[idx.to(r.device)]
            self.cur[(self.step, bi)] = r.float()
        return hook

    def new_cycle(self): self.step, self.cur, self.enabled = -1, {}, True
    def end_cycle(self): self.enabled = False; return self.cur
    def close(self):
        for h in self.handles: h.remove()


def fetch(ep):
    """Per-episode mp4s for the three DROID camera streams."""
    from huggingface_hub import hf_hub_download
    import imageio.v3 as iio
    names = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    out = {}
    for n in names:
        rel = f"videos/chunk-{ep['chunk']:03d}/observation.images.{n}/episode_{ep['episode']:06d}.mp4"
        local = os.path.join(VIDROOT, rel)
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            import shutil
            shutil.copy(hf_hub_download("GEAR-Dreams/DreamZero-DROID-Data", rel,
                                        repo_type="dataset"), local)
        out[n] = np.stack([f for f in iio.imiter(local)])
    return out


def main():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not dist.is_initialized():
        for k, v in [("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29513"),
                     ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
            os.environ.setdefault(k, v)
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    sys.path.insert(0, "/dz")
    import eval_utils.serve_dreamzero_wan22 as S

    mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    t0 = time.time()
    policy = S.GrootSimPolicy(embodiment_tag=S.EmbodimentTag("oxe_droid"),
                              model_path=CKPT, device="cuda", device_mesh=mesh)
    h, w = S._get_expected_video_resolution(policy)
    wrapper = S.DreamZeroWan225BPolicy(groot_policy=policy, image_height=h,
                                       image_width=w, embodiment_tag="oxe_droid")
    dit = policy.trained_model.action_head.model
    print(f"[dzfam] loaded in {time.time()-t0:.0f}s | {len(dit.blocks)} blocks "
          f"dim={getattr(dit,'dim','?')} local_attn={dit.local_attn_size}", flush=True)

    tracer = Tracer(dit.blocks, BLOCK_STRIDE, STEP_STRIDE, NTOK)
    eps = json.load(open(EPS_JSON))
    print(f"[dzfam] blocks {tracer.tracked} | every {STEP_STRIDE} steps | "
          f"{NTOK} tokens | {len(eps)} episodes", flush=True)

    acc = {k: [] for k in ("gram", "chan1", "tok1", "chan2", "ep", "cycle",
                           "step", "block", "obs_delta")}
    dev, t_start = torch.device("cuda"), time.time()
    for ei, ep in enumerate(eps):
        try:
            streams = fetch(ep)
        except Exception as e:
            print(f"[dzfam] ep {ei} fetch failed: {e}", flush=True); continue
        n_frames = min(len(v) for v in streams.values())
        ncyc = min(CYCLES, (n_frames - 1) // FRAMES_PER_CHUNK)
        rec, prev_obs, kept = MultiRecorder(dev), None, 0
        for c in range(ncyc):
            s, e = c * FRAMES_PER_CHUNK, (c + 1) * FRAMES_PER_CHUNK
            ext1 = streams["exterior_image_1_left"][s:e]
            obs = {"observation/exterior_image_0_left": ext1,
                   "observation/exterior_image_1_left": streams["exterior_image_2_left"][s:e],
                   "observation/wrist_image_left": streams["wrist_image_left"][s:e],
                   "observation/joint_position": np.zeros(7, dtype=np.float64),
                   "observation/gripper_position": np.zeros(1, dtype=np.float64),
                   "prompt": str(ep["task"]), "session_id": f"fam{ei}"}
            f = ext1.astype(np.float32)
            d = float(np.linalg.norm(f - prev_obs) / max(np.linalg.norm(prev_obs), 1e-6)) \
                if prev_obs is not None else float("nan")
            prev_obs = f

            tracer.new_cycle()
            with torch.no_grad(): wrapper.infer(obs)
            cur = tracer.end_cycle()
            for shape, (keys, r) in rec.push(cur).items():
                n = len(keys)
                acc["gram"].append(r["gram"])
                for k in ("chan1", "tok1", "chan2"): acc[k].append(r[k])
                acc["ep"].append(np.full(n, ei, np.int16))
                acc["cycle"].append(np.full(n, c, np.int16))
                acc["step"].append(np.array([k[0] for k in keys], np.int8))
                acc["block"].append(np.array([k[1] for k in keys], np.int8))
                acc["obs_delta"].append(np.full(n, d, np.float32))
                kept += n
        rec.reset(); del rec
        torch.cuda.empty_cache()          # release the episode's history before the next
        print(f"[dzfam] ep {ei+1}/{len(eps)} #{ep['episode']} {ncyc}cyc -> {kept} cells | "
              f"{(time.time()-t_start)/60:.1f} min | \"{str(ep['task'])[:38]}\"", flush=True)
        np.savez_compressed(OUT,
            meta=np.array([json.dumps({"gpu": torch.cuda.get_device_name(0),
                "episodes": eps[:ei+1], "hist": HIST, "ncol": NCOL,
                "block_stride": BLOCK_STRIDE, "step_stride": STEP_STRIDE, "ntok": NTOK,
                "frames_per_chunk": FRAMES_PER_CHUNK,
                "note": "full compute throughout; no cache applied"})]),
            **{k: np.concatenate(v) for k, v in acc.items() if v})
    tracer.close()
    print(f"[dzfam] wrote {sum(len(x) for x in acc['ep'])} cell-observations", flush=True)


if __name__ == "__main__":
    main()

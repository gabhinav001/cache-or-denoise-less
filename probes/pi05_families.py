"""Third backbone: pi0.5, to separate "denoising residual" from "video world model".

Fast-WAM and DreamZero are both Wan-derived video DiTs, so agreement between them
could be an artifact of shared lineage. pi0.5 shares none of it: a PaliGemma VLM
with a Gemma-300M flow-matching action expert, no video generation anywhere. What
it does share is the structure that matters here -- the prefix (images, language,
state) is encoded once into a KV cache, then an expert runs the same layers over
action tokens at every denoise step. Cells are the same object: (step, layer).

It is also finetuned on LIBERO, so the observations are the same benchmark
Fast-WAM is measured on, and the LeRobot dataset supplies real proprioceptive
state rather than the fixed dummy vector the other two probes use.

Two strides are run: 10, matching the Fast-WAM probe so the cross-backbone
comparison holds observation spacing fixed, and 50, pi0.5's own chunk length,
which is what its deployment cadence would actually be.
"""
from __future__ import annotations

import json, os, time
import numpy as np
import torch

from famlib import MultiRecorder, HIST, NCOL

MODEL = os.environ.get("PI_MODEL", "lerobot/pi05_libero_finetuned_v044")
DATASET = os.environ.get("PI_DATASET", "lerobot/libero")
OUT = os.environ.get("PI_OUT", "/workspace/pi05_families.npz")
N_EP = int(os.environ.get("PI_EPISODES", "52"))
STRIDES = [int(s) for s in os.environ.get("PI_STRIDES", "10,50").split(",")]
MAX_CYC = int(os.environ.get("PI_MAX_CYCLES", "32"))
# The checkpoint's processor points at google/paligemma-3b-pt-224 for its tokenizer,
# which is gated. This mirror's tokenizer files are byte-identical -- sha256 matches
# on tokenizer.model and tokenizer.json, exact sizes on the three small JSONs -- so
# tokenization is unchanged and no credential has to reach the rented machine.
TOKENIZER = os.environ.get("PI_TOKENIZER", "leo009/paligemma-3b-pt-224")


class Tracer:
    """R[c,k,b] on the action expert's decoder layers, active only inside denoise_step."""

    def __init__(self, layers):
        self.n = len(layers)
        self.step, self.cur, self.on = -1, {}, False
        self.handles = [l.register_forward_hook(self._mk(i), with_kwargs=True)
                        for i, l in enumerate(layers)]

    def _mk(self, li):
        def hook(module, args, kwargs, output):
            if not self.on: return
            x_in = kwargs.get("hidden_states", args[0] if args else None)
            if x_in is None: return
            x_out = output[0] if isinstance(output, (tuple, list)) else output
            if x_out.shape != x_in.shape: return
            self.cur[(self.step, li)] = (x_out - x_in).detach().reshape(-1, x_out.shape[-1]).float()
        return hook

    def new_cycle(self): self.step, self.cur = -1, {}
    def close(self):
        for h in self.handles: h.remove()


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    dev = torch.device("cuda")
    policy = PI05Policy.from_pretrained(MODEL).to(dev).eval()
    preprocess, _ = make_pre_post_processors(
        policy.config, MODEL,
        preprocessor_overrides={"device_processor": {"device": str(dev)},
                                "tokenizer_processor": {"tokenizer_name": TOKENIZER}})
    # torchcodec needs system ffmpeg shared libraries the image does not ship;
    # pyav bundles its own, and both decode the same frames
    ds = LeRobotDataset(DATASET, video_backend=os.environ.get("PI_VIDEO_BACKEND", "pyav"))
    layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
    print(f"[pi05] {len(layers)} expert layers, {policy.config.num_inference_steps} steps, "
          f"chunk {policy.config.chunk_size} | dataset {len(ds)} frames", flush=True)

    tracer = Tracer(layers)
    dstep = policy.model.denoise_step
    def traced(*a, **k):
        tracer.step += 1; tracer.on = True
        try: return dstep(*a, **k)
        finally: tracer.on = False
    policy.model.denoise_step = traced

    ep_meta = ds.meta.episodes
    n_ep = min(N_EP, len(ep_meta["dataset_from_index"]))
    acc = {k: [] for k in ("gram", "chan1", "tok1", "chan2", "ep", "cycle",
                           "step", "block", "obs_delta", "stride")}
    t0 = time.time()
    for stride in STRIDES:
        for ei in range(n_ep):
            f0 = int(ep_meta["dataset_from_index"][ei]); f1 = int(ep_meta["dataset_to_index"][ei])
            ncyc = min(MAX_CYC, (f1 - f0) // stride)
            if ncyc <= HIST + 1: continue
            rec, prev_img, kept = MultiRecorder(dev), None, 0
            for c in range(ncyc):
                frame = dict(ds[f0 + c * stride])
                batch = preprocess(frame)
                img = frame.get("observation.images.image")
                d = float("nan")
                if img is not None:
                    a = np.asarray(img, dtype=np.float32)
                    if prev_img is not None:
                        d = float(np.linalg.norm(a - prev_img) / max(np.linalg.norm(prev_img), 1e-6))
                    prev_img = a
                tracer.new_cycle()
                with torch.inference_mode():
                    policy.predict_action_chunk(batch)
                for shape, (keys, r) in rec.push(tracer.cur).items():
                    n = len(keys)
                    acc["gram"].append(r["gram"])
                    for k in ("chan1", "tok1", "chan2"): acc[k].append(r[k])
                    acc["ep"].append(np.full(n, ei, np.int16))
                    acc["cycle"].append(np.full(n, c, np.int16))
                    acc["step"].append(np.array([k[0] for k in keys], np.int8))
                    acc["block"].append(np.array([k[1] for k in keys], np.int8))
                    acc["obs_delta"].append(np.full(n, d, np.float32))
                    acc["stride"].append(np.full(n, stride, np.int16))
                    kept += n
            print(f"[pi05] stride {stride} ep {ei+1}/{n_ep} {ncyc}cyc -> {kept} cells | "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            np.savez_compressed(OUT,
                meta=np.array([json.dumps({"gpu": torch.cuda.get_device_name(0),
                    "model": MODEL, "dataset": DATASET, "hist": HIST, "ncol": NCOL,
                    "steps": policy.config.num_inference_steps, "layers": len(layers),
                    "strides": STRIDES, "episodes_done": ei + 1, "stride_running": stride,
                    "note": "full compute throughout; no cache applied"})]),
                **{k: np.concatenate(v) for k, v in acc.items() if v})
    tracer.close(); policy.model.denoise_step = dstep
    print(f"[pi05] wrote {sum(len(x) for x in acc['ep'])} cell-observations", flush=True)


if __name__ == "__main__":
    main()

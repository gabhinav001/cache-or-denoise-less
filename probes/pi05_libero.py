"""Closed-loop LIBERO for pi0.5, with and without per-cell attenuation.

The residual geometry holds on three backbones, but the closed-loop consequence
has so far been shown on one. pi0.5 shares no lineage with the Wan-derived WAMs
-- PaliGemma backbone, Gemma-300M flow-matching action expert, no video
generation -- so if the correction transfers here it is a property of denoising
policies rather than of one model family.

Cycles. pi0.5 predicts 50 actions per inference. Executing all 50 would put
consecutive inferences 2.5 s apart, far outside the spacing the weights were
measured at, and would give a LIBERO episode only a handful of cycles. We
execute 10 and replan, which is both a legitimate deployment choice and the
cadence Fast-WAM uses, so the two backbones are compared at matched spacing.
With n_action_steps set to that value, select_action runs inference exactly when
(t mod 10) == 0, which is what the cache semantics key off.

Placement mirrors the published one at matched fraction: denoise steps 0-7 of 10
across all 18 expert layers is 144 of 180 cells, the same 80% that steps 0-7 of
10 across 30 blocks is for Fast-WAM.
"""
from __future__ import annotations

import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import provenance, file_digest   # noqa: E402
import numpy as np
import torch

MODEL = os.environ.get("PI_MODEL", "lerobot/pi05_libero_finetuned_v044")
TOKENIZER = os.environ.get("PI_TOKENIZER", "leo009/paligemma-3b-pt-224")
SUITE = os.environ.get("PI_SUITE", "libero_goal")
N_TASKS = int(os.environ.get("PI_TASKS", "10"))
N_EPISODES = int(os.environ.get("PI_EPISODES", "10"))
FIRST_TASK = int(os.environ.get("PI_FIRST_TASK", "0"))
EXEC = int(os.environ.get("PI_EXEC", "10"))
SKIP_STEPS = int(os.environ.get("PI_SKIP_STEPS", "8"))
CONFIGS = os.environ.get("PI_CONFIGS", "full,c3ache:2,c3ache:2:s").split(",")
SHRINK = os.environ.get("PI_SHRINK", "/workspace/shrink_pi05.json")
OUT = os.environ.get("PI_OUT", "/workspace/pi05_libero.json")
SEED = int(os.environ.get("PI_SEED", "42"))


# Something in the policy stack compiles with mode="reduce-overhead", i.e. CUDA
# graph capture. Graph-owned memory is reclaimed on the next run, so a residual
# recorded on one cycle is overwritten before the next cycle reads it, and
# cloning does not help because the clone lands in the same pool. Dropping to
# plain inductor keeps the compilation while letting cached tensors outlive the
# call that produced them. Each cache configuration is also a distinct set of
# dynamo guards, so switching arms recompiles; the default limit of eight is too
# low for five arms.
import torch._inductor.config as _ind
_ind.triton.cudagraphs = False
_SAFE = {"reduce-overhead": None, "max-autotune": "max-autotune-no-cudagraphs"}
_torch_compile = torch.compile
def _compile_without_cudagraphs(*a, **k):
    if k.get("mode") in _SAFE:
        k = {**k, "mode": _SAFE[k["mode"]]}
    return _torch_compile(*a, **k)
torch.compile = _compile_without_cudagraphs
for _lim in ("recompile_limit", "accumulated_recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _lim):
        setattr(torch._dynamo.config, _lim, 256)


def _batch1(x):
    """Add a leading batch dimension to every leaf array.

    The env returns dicts nested two deep (robot_state -> eef -> pos), so a
    single-level comprehension turns the inner dicts into object arrays and
    torch.from_numpy rejects them.
    """
    if isinstance(x, dict):
        return {k: _batch1(v) for k, v in x.items()}
    return np.asarray(x)[None]


class CacheHarness:
    """Records or substitutes the residual of each (denoise step, expert layer)."""

    def __init__(self, layers):
        self.n = len(layers)
        self.step, self.layer, self.on = -1, 0, False
        self.record = self.apply = self.cells = self.scale = None
        self.handles = [l.register_forward_hook(self._mk(i), with_kwargs=True)
                        for i, l in enumerate(layers)]

    def _mk(self, li):
        def hook(module, args, kwargs, output):
            if not self.on: return output
            x_in = kwargs.get("hidden_states", args[0] if args else None)
            if x_in is None: return output
            tup = isinstance(output, (tuple, list))
            x_out = output[0] if tup else output
            if x_out.shape != x_in.shape: return output
            key = (self.step, li)
            if self.record is not None:
                self.record[key] = (x_out - x_in).detach().clone()
            if self.cells is not None and key in self.cells:
                c = self.apply.get(key) if self.apply else None
                if c is not None and c.shape == x_in.shape:
                    s = self.scale.get(key, 1.0) if self.scale else 1.0
                    new = x_in + (c if s == 1.0 else s * c)
                    return (new,) + tuple(output[1:]) if tup else new
            return output
        return hook

    def begin_step(self): self.step += 1; self.layer = 0; self.on = True
    def end_step(self): self.on = False
    def reset(self, record=None, apply=None, cells=None, scale=None):
        self.step, self.layer = -1, 0
        self.record, self.apply, self.cells, self.scale = record, apply, cells, scale
    def close(self):
        for h in self.handles: h.remove()


def main():
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    dev = torch.device("cuda")
    policy = PI05Policy.from_pretrained(MODEL).to(dev).eval()
    policy.config.n_action_steps = EXEC          # replan every EXEC env steps
    pre, post = make_pre_post_processors(
        policy.config, MODEL,
        preprocessor_overrides={"device_processor": {"device": str(dev)},
                                "tokenizer_processor": {"tokenizer_name": TOKENIZER}})
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
    n_steps = int(policy.config.num_inference_steps)
    H = CacheHarness(layers)
    dstep = policy.model.denoise_step
    def traced(*a, **k):
        H.begin_step()
        try: return dstep(*a, **k)
        finally: H.end_step()
    policy.model.denoise_step = traced

    cells = {(k, b) for k in range(SKIP_STEPS) for b in range(len(layers))}

    # PI_SHRINK is either one path (the original form) or a comma-separated list
    # of label=path pairs. The list form lets several weight settings run as arms
    # of ONE file, paired on (task, episode) within the run rather than compared
    # across files. That matters for a confirmation: comparing across files needs
    # the two full-compute arms to agree, which is an extra assumption and an
    # extra failure mode, and it doubles the cost by running full twice.
    def _load(path):
        if not os.path.exists(path): return None
        d = json.load(open(path))
        return {(int(k.split(",")[0]), int(k.split(",")[1])): float(v[0])
                for k, v in d["coef"].items()}

    WEIGHTS, SC = {}, None
    if "=" in SHRINK:
        for part in SHRINK.split(","):
            lbl, _, path = part.partition("=")
            w = _load(path.strip())
            if w is None: raise SystemExit(f"shrink file not found: {path}")
            WEIGHTS[lbl.strip()] = w
    else:
        SC = _load(SHRINK)

    specs = []
    for c in CONFIGS:
        c = c.strip()
        if c == "full": specs.append(("full", None, None))
        else:
            p = c.split(":"); tau = int(p[1])
            tag = p[2] if len(p) > 2 else ""
            lbl = ""
            if tag.startswith("s"):
                # "s" uses the single file; "s=name" picks one from the list
                _, _, lbl = tag.partition("=")
                if lbl:
                    if lbl not in WEIGHTS:
                        raise SystemExit(f"config {c}: no weight labelled {lbl!r} in PI_SHRINK")
                    sc = WEIGHTS[lbl]
                else:
                    sc = SC
                    if sc is None: raise SystemExit(f"config {c} needs {SHRINK}")
            else:
                sc = None
            name = f"c3ache_tau{tau}" + ("_shrink" if sc else "") + (f"_{lbl}" if lbl else "")
            specs.append((name, tau, sc))
    print(f"[pi05] {len(layers)} layers x {n_steps} steps | caching {len(cells)} of "
          f"{len(layers)*n_steps} cells | exec {EXEC} | configs {[s[0] for s in specs]}",
          flush=True)

    suite = _get_suite(SUITE)
    rows, t0 = [], time.time()
    for ti in range(FIRST_TASK, N_TASKS):
        for cname, tau, sc in specs:
            # episode_index 0 with n_envs 1 makes successive resets walk init states
            # 0, 1, 2, ... so episode e sees the same initial state in every arm
            env = LiberoEnv(task_suite=suite, task_id=ti, task_suite_name=SUITE,
                            obs_type="pixels_agent_pos", episode_index=0, n_envs=1)
            max_steps = env._max_episode_steps
            for ep in range(N_EPISODES):
                torch.manual_seed(SEED); np.random.seed(SEED)
                policy.reset()
                obs, _ = env.reset(seed=SEED)
                cache, success, t = {}, False, 0
                while t < max_steps:
                    cycle = t // EXEC
                    if t % EXEC == 0:
                        if tau is None or cycle == 0 or (tau > 0 and cycle % tau == 0):
                            cache = {}; H.reset(record=cache)
                        else:
                            H.reset(apply=cache, cells=cells, scale=sc)
                    batch = env_pre(preprocess_observation(_batch1(obs)))
                    batch["task"] = env.task
                    with torch.inference_mode():
                        act = post(policy.select_action(pre(batch)))
                    a = act.squeeze(0).float().cpu().numpy()
                    obs, _, term, _, info = env.step(a)
                    t += 1
                    if info.get("is_success"): success = True; break
                    if term: break
                rows.append({"suite": SUITE, "task": ti, "config": cname,
                             "episode": ep, "success": bool(success), "steps": t})
            sr = np.mean([r["success"] for r in rows if r["config"] == cname and r["task"] == ti])
            print(f"[pi05] task {ti} {cname:20s} SR={sr:.2f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            env.close()
        # Seed and provenance recording: ensure each run explicitly records
        # seed, commit, machine, and weight file provenance in the meta block.
        json.dump({"meta": {"suite": SUITE, "model": MODEL, "exec": EXEC,
                            "seed": SEED,
                            "n_tasks": N_TASKS, "first_task": FIRST_TASK,
                            "n_episodes": N_EPISODES, "skip_cells": len(cells),
                            "skip_steps": SKIP_STEPS,
                            "total_cells": len(layers) * n_steps,
                            "shrink": os.path.basename(SHRINK.split(",")[0]),
                            "shrink_sha256_16": file_digest(
                                SHRINK.split("=")[-1].split(",")[0]),
                            "configs": [s[0] for s in specs],
                            "provenance": provenance()}, "rows": rows},
                  open(OUT, "w"))
    policy.model.denoise_step = dstep; H.close()
    print("\n=== success rate by config ===")
    for cname in [s[0] for s in specs]:
        v = [r["success"] for r in rows if r["config"] == cname]
        if v: print(f"  {cname:20s} {np.mean(v)*100:5.1f}%  (n={len(v)})")


if __name__ == "__main__":
    main()

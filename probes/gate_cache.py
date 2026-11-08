"""Check loaded weights, replay error, cache coverage, and memory integrity.

Schema 3 keeps the 0.005 relative action-error bound and replaces the stale/fresh
error-ratio requirement with exact CPU snapshots checked before and after later
inference calls. Small stale error is compatible with a useful cache; it cannot
by itself diagnose memory corruption. The ratio remains a reported diagnostic.
The original schema-2 failure is retained unchanged in the research artifact.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
_ld = torch.load
torch.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_libero import CacheHarness, StackHarness, placements, N_BLOCKS, N_STEPS, SKIP
from cache_integrity import CacheIntegrityMonitor
from stack_probe_common import file_sha256, inference_runtime, validate_checkpoint_state

# Thresholds. The identity bound is loose enough to tolerate bf16 reduction
# order differing between a recorded and a substituted path, and tight enough
# that a reclaimed-memory cache (which lands in the 1e-1 range) cannot pass.
MAX_IDENTITY_ERR = 5e-3


def rel_l2(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64))
    n = np.linalg.norm(b.astype(np.float64))
    return float(np.linalg.norm(d) / n) if n > 0 else float("inf")


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pr = OmegaConf.merge(OmegaConf.create(
        {"oracle": "/workspace/oracle_cells.json", "suite": "libero_goal",
         "task": 0, "warmup_cycles": 4, "stale_gap": 4,
         "out": "/workspace/gate_cache.json"}), cfg.get("PROBE", {}))

    sys.path.insert(0, "/w/experiments/libero")
    import eval_libero_single as E
    from libero_utils import get_libero_env, get_libero_dummy_action
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    payload = model.load_checkpoint(str(cfg.ckpt))
    checkpoint_coverage = validate_checkpoint_state(model, payload)
    del payload
    model = model.to(dev).eval()
    runtime = inference_runtime(cfg, model)
    runtime["checkpoint_state_coverage"] = checkpoint_coverage
    runtime["checkpoint_sha256"] = file_sha256(str(cfg.ckpt))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path)))

    ah = int(cfg.data.train.num_frames) - 1
    h, w = (int(v) for v in cfg.data.train.video_size)
    replan = int(cfg.EVALUATION.get("replan_steps", 10))
    wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    steps = int(cfg.EVALUATION.get("num_inference_steps", N_STEPS))

    # The model facts the paper states (parameters per expert, blocks, chunk,
    # replanning, schedule), read from the loaded model and its config so a new
    # host re-measures them. Recorded only; nothing below depends on them.
    def _facts():
        f = {"action_horizon": ah, "replan_steps": replan, "num_inference_steps": steps}
        for name in ("video_expert", "action_expert"):
            mod = getattr(model, name, None)
            if mod is not None:
                f[f"{name}_params"] = int(sum(q.numel() for q in mod.parameters()))
        blocks = getattr(getattr(model, "action_expert", None), "blocks", None)
        if blocks is not None:
            f["action_expert_blocks"] = len(blocks)
        f["model_params"] = int(sum(q.numel() for q in model.parameters()))
        return f
    try:
        model_facts = _facts()
    except Exception as ex:          # a fact we cannot read is not a reason to fail the gate
        model_facts = {"error": f"{type(ex).__name__}: {ex}"}
    cells = placements(str(pr.oracle), steps)["c3ache"]
    stack_steps = {k for k, _ in cells}
    if not cells:
        raise ValueError("This gate requires a schedule with at least one cacheable step.")

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[str(pr.suite)]()
    task = suite.get_task(int(pr.task))
    inits = suite.get_task_init_states(int(pr.task))
    env, desc = get_libero_env(task, E.LIBERO_ENV_RESOLUTION, int(cfg.get("seed", 42)))

    SH = StackHarness(model.mot)
    SH.on = True
    H = CacheHarness(model.mot)
    dn = model._denoise_action_with_video_cache
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    model._denoise_action_with_video_cache = traced

    integrity = CacheIntegrityMonitor()

    def predict(obs, seed):
        integrity.check()
        # the denoiser draws noise; fixing the seed makes two calls on identical
        # inputs comparable, which is the whole basis of this test
        torch.manual_seed(seed)
        chunk, _, _ = E._predict_action_chunk(
            obs=obs, task_description=desc, model=model, processor=processor,
            cfg=cfg, action_horizon=ah, input_w=w, input_h=h, model_device=str(dev))
        integrity.check()
        return np.asarray(chunk, dtype=np.float64)

    env.reset(); obs = env.set_init_state(inits[0])
    for _ in range(wait):
        obs, _, _, _ = env.step(get_libero_dummy_action())

    # advance a few cycles so the model is in a realistic state, keeping the
    # residuals from `stale_gap` cycles back for the control arm
    if not 1 <= int(pr.stale_gap) <= int(pr.warmup_cycles):
        raise ValueError("Require 1 <= stale_gap <= warmup_cycles")
    old_cache, old_stack = None, None
    for c in range(int(pr.warmup_cycles)):
        cache, stack = {}, {}
        H.reset(record=cache)
        SH.reset(record=stack)
        chunk = predict(obs, 1234)
        if c == int(pr.warmup_cycles) - int(pr.stale_gap):
            old_cache = dict(cache)
            old_stack = dict(stack)
            integrity.watch("old_block", old_cache, cells)
            integrity.watch("old_stack", old_stack, stack_steps)
        for a in chunk[:replan]:
            obs, _, done, _ = env.step(a.tolist())
            if done: break

    # 1. full compute, recording this cycle's residuals
    fresh, fresh_stack = {}, {}
    H.reset(record=fresh)
    SH.reset(record=fresh_stack)
    a_full = predict(obs, 4242)
    if not cells.issubset(fresh) or not stack_steps.issubset(fresh_stack):
        raise RuntimeError("The gate did not record every selected cache entry.")

    integrity.watch("fresh_block", fresh, cells)
    integrity.watch("fresh_stack", fresh_stack, stack_steps)

    H.reset(record={}); SH.reset(record={})
    a_repeat = predict(obs, 4242)

    # 2. same cycle, same input, substituting the residuals just recorded.
    #    rx + (out - rx) == out, so this must reproduce a_full.
    H.reset(apply=fresh, cells=cells)
    SH.reset()
    a_ident = predict(obs, 4242)

    H.reset(apply=fresh, cells=cells, skip=True)
    SH.reset()
    a_ident_skip = predict(obs, 4242)

    H.reset()
    SH.reset(apply=fresh_stack, steps=stack_steps); SH.zero_counts()
    a_ident_whole = predict(obs, 4242)
    whole_identity_counts = SH.counts()

    # 3. control: the same substitution with residuals from `stale_gap` cycles ago
    a_stale = None
    if old_cache:
        H.reset(apply=old_cache, cells=cells)
        SH.reset()
        a_stale = predict(obs, 4242)

    a_stale_whole = None
    if old_stack:
        H.reset()
        SH.reset(apply=old_stack, steps=stack_steps); SH.zero_counts()
        a_stale_whole = predict(obs, 4242)

    model._denoise_action_with_video_cache = dn; H.close(); SH.close(); env.close()

    e_ident = rel_l2(a_ident, a_full)
    e_stale = rel_l2(a_stale, a_full) if a_stale is not None else float("nan")
    margin = (e_stale / e_ident) if e_ident > 0 else None
    e_repeat = rel_l2(a_repeat, a_full)
    e_skip = rel_l2(a_ident_skip, a_full)
    e_whole = rel_l2(a_ident_whole, a_full)
    e_stale_whole = rel_l2(a_stale_whole, a_full) if a_stale_whole is not None else None
    whole_coverage = (whole_identity_counts["hits"] == len(stack_steps)
                      and whole_identity_counts["misses"] == 0)

    print("\n=== cache identity gate ===")
    print(f"  selected block cells  {len(cells)} of {steps*N_BLOCKS} (record coverage checked)")
    print(f"  identity error        {e_ident:.3e}   (must be < {MAX_IDENTITY_ERR:.0e})")
    print(f"  stale-cache control   {e_stale:.3e}   ({pr.stale_gap} cycles old)")
    print(f"  stale/fresh ratio     {margin} (diagnostic only)")
    print(f"  cache integrity       {integrity.report()}")
    print(f"  repeat-full error     {e_repeat:.3e}")
    print(f"  skip identity error   {e_skip:.3e}")
    print(f"  whole identity error  {e_whole:.3e}; actual hits/misses {whole_identity_counts}")

    ok = (all(np.isfinite(e) and e < MAX_IDENTITY_ERR
              for e in (e_ident, e_skip, e_whole, e_repeat))
          and integrity.passed() and whole_coverage)
    json.dump({"identity_err": e_ident, "stale_err": e_stale, "margin": margin,
               "schema_version": 3, "runtime": runtime,
               "cache_integrity": integrity.report(),
               "repeat_full_err": e_repeat, "skip_identity_err": e_skip,
               "whole_identity_err": e_whole, "whole_stale_err": e_stale_whole,
               "whole_identity_counts": whole_identity_counts,
               # the size of the cache a whole-stack arm keeps between calls:
               # one stack residual per CACHED step, in the model's dtype. The
               # full-compute pass records every step; only these are kept.
               "whole_cache_bytes": int(sum(fresh_stack[k].numel() * fresh_stack[k].element_size()
                                            for k in stack_steps)),
               "whole_cache_entry_shape": list(next(iter(fresh_stack.values())).shape),
               "model_facts": model_facts,
               "perblock_whole_stale_difference": (
                   rel_l2(a_stale, a_stale_whole) if a_stale is not None and a_stale_whole is not None else None),
               "ckpt": os.path.basename(str(cfg.ckpt)),
               "compiled": bool(cfg.EVALUATION.get("compile_action_infer", False)),
               "cells": len(cells), "pass": bool(ok),
               "gpu": torch.cuda.get_device_name(0)}, open(str(pr.out), "w"))

    if not ok:
        print("\n  FAIL -- at least one replay-error, memory-integrity, or coverage check failed.")
        print("  Inspect the individual errors before collecting task-success results.")
        print("  A failure alone does not identify memory corruption or its cause.")
        sys.exit(1)
    print("\n  PASS -- cached residuals survive the call that produced them.\n")


if __name__ == "__main__":
    main()

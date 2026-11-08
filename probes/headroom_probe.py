"""Phase 0 headroom probe for FastWAM.

Answers the two questions that decide whether a cross-chunk residual cache can
still buy anything on this backbone:

  A) How much of the published cache speedup has torch.compile already taken?
     The official LIBERO eval ships compile_action_infer=true, so the honest
     baseline for any cache is the CUDA-graph-compiled model, not the eager one.

  B) What fraction of a cycle is spent in the denoise steps a cache can skip?
     Everything outside them (VAE encode, video prefill) is untouchable, so this
     is a hard ceiling on any block-caching method.

Stage timing runs in eager mode only. Under mode="reduce-overhead",
fullgraph=True the denoise step is a captured CUDA graph: Python-side event
records inside it would be a graph break, and the internal boundaries stop being
meaningful anyway. So we take the breakdown eagerly and the end-to-end both ways.

All timings use CUDA events recorded during the run and resolved after a single
synchronize at the end. Wrapping perf_counter around an async CUDA op measures
kernel-launch time, not execution time.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import platform
import statistics
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CACHEABLE_STEPS = 8  # C3ache's best LIBERO config caches denoise steps [0,7] of 10


class StageProfiler:
    """Records CUDA event pairs during the run; resolves them only at the end."""

    def __init__(self) -> None:
        self.pending: list[tuple[int, str, torch.cuda.Event, torch.cuda.Event]] = []
        self.enabled = False
        self.cycle = -1
        self.denoise_idx = 0

    @contextlib.contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((self.cycle, name, start, end))

    def resolve(self) -> list[dict]:
        torch.cuda.synchronize()
        return [
            {"cycle": c, "stage": n, "ms": s.elapsed_time(e)}
            for c, n, s, e in self.pending
        ]


def install_probes(model, prof: StageProfiler) -> list[tuple[object, str, object]]:
    """Wrap the three real stage boundaries in FastWAM's inference path.

    Boundaries taken from fastwam.py::infer_action:
      _encode_input_image_latents_tensor -> VAE encode of the current frame
      mot.prefill_video_cache_tensor     -> 5B video expert, builds video K/V
      _denoise_action_with_video_cache   -> 1B action expert, called once per step
    """
    originals: list[tuple[object, str, object]] = []

    def wrap(owner, attr, label, is_denoise=False):
        fn = getattr(owner, attr)
        originals.append((owner, attr, fn))

        def wrapped(*args, **kwargs):
            if is_denoise:
                name = f"denoise_k{prof.denoise_idx}"
                prof.denoise_idx += 1
            else:
                name = label
            with prof.stage(name):
                return fn(*args, **kwargs)

        setattr(owner, attr, wrapped)

    # The eval loop passes `prompt=` on every cycle, so T5 re-encodes a constant
    # instruction once per inference. Measure it: it is inside the 190 ms budget,
    # it is trivially cacheable, and a DiT block cache cannot touch it.
    if getattr(model, "text_encoder", None) is not None:
        wrap(model, "encode_prompt", "text_encode")
    wrap(model, "_encode_input_image_latents_tensor", "vae_encode")
    wrap(model.mot, "prefill_video_cache_tensor", "video_prefill")
    wrap(model, "_denoise_action_with_video_cache", "denoise", is_denoise=True)
    return originals


def remove_probes(originals) -> None:
    for owner, attr, fn in originals:
        setattr(owner, attr, fn)


def run_mode(model, kwargs, device, *, label, warmup, iters, prof=None):
    """Warmup then timed iterations. Returns per-iteration end-to-end ms."""
    k = dict(kwargs)
    for _ in range(warmup):
        if prof is not None:
            prof.denoise_idx = 0
        with torch.no_grad():
            model.infer_action(**k)
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    e2e = []
    for i in range(iters):
        if prof is not None:
            prof.cycle = i
            prof.denoise_idx = 0
            prof.enabled = True
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.infer_action(**k)
        torch.cuda.synchronize(device)
        e2e.append((time.perf_counter() - t0) * 1000.0)
        if prof is not None:
            prof.enabled = False
        logger.info("%s %d/%d: %.2f ms", label, i + 1, iters, e2e[-1])

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return e2e, peak


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def summarize(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "p95_ms": pct(xs, 0.95),
        "p99_ms": pct(xs, 0.99),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    probe = OmegaConf.merge(
        OmegaConf.create({"warmup": 5, "iters": 60, "seed": 42, "out": "/tmp/headroom.json"}),
        cfg.get("PROBE", {}),
    )
    device = torch.device(str(cfg.EVALUATION.device))

    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        str(cfg.mixed_precision)
    ]
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()

    height, width = (int(v) for v in cfg.data.train.video_size)
    action_horizon = int(cfg.data.train.num_frames) - 1
    num_inference_steps = int(cfg.eval_num_inference_steps)

    torch.manual_seed(int(probe.seed))
    use_random_context = bool(probe.get("use_random_context", False)) or (
        getattr(model, "text_encoder", None) is None
    )
    if use_random_context:
        logger.warning("no text encoder: using random context, text_encode unmeasured")
        ctx = torch.randn(1, int(cfg.data.train.context_len), model.text_dim)
        prompt_arg, ctx_arg, mask_arg = None, ctx, torch.ones(ctx.shape[:2], dtype=torch.bool)
    else:
        prompt_arg, ctx_arg, mask_arg = "pick up the object", None, None

    infer_kwargs = {
        "prompt": prompt_arg,
        "context": ctx_arg,
        "context_mask": mask_arg,
        "input_image": torch.rand(1, 3, height, width) * 2 - 1,
        "action_horizon": action_horizon,
        "proprio": torch.randn(1, int(cfg.data.train.processor.proprio_output_dim)),
        "num_inference_steps": num_inference_steps,
        "seed": int(probe.seed),
        "rand_device": str(cfg.EVALUATION.rand_device),
    }
    params = inspect.signature(model.infer_action).parameters
    if "num_video_frames" in params:
        infer_kwargs["num_video_frames"] = (
            action_horizon // int(cfg.data.train.action_video_freq_ratio) + 1
        )
    if "action_infer_mode" in params:
        infer_kwargs["action_infer_mode"] = "idm"

    meta = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "num_inference_steps": num_inference_steps,
        "action_horizon": action_horizon,
        "video_size": [height, width],
        "ckpt": str(cfg.ckpt),
        "warmup": int(probe.warmup),
        "iters": int(probe.iters),
        "host": platform.platform(),
    }
    logger.info("meta=%s", json.dumps(meta))

    # ---- Pass 1: eager, fully instrumented -------------------------------
    prof = StageProfiler()
    originals = install_probes(model, prof)
    eager_kwargs = dict(infer_kwargs, compile_action_infer=False)
    eager_e2e, eager_peak = run_mode(
        model, eager_kwargs, device, label="eager", warmup=int(probe.warmup),
        iters=int(probe.iters), prof=prof,
    )
    stages = prof.resolve()
    remove_probes(originals)

    # ---- Pass 2: compiled, end-to-end only -------------------------------
    compiled_kwargs = dict(infer_kwargs, compile_action_infer=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.infer_action(**compiled_kwargs)
    torch.cuda.synchronize(device)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0
    logger.info("compile cold start: %.1f ms", cold_start_ms)

    compiled_e2e, compiled_peak = run_mode(
        model, compiled_kwargs, device, label="compiled",
        warmup=int(probe.warmup), iters=int(probe.iters), prof=None,
    )

    # ---- Derived answers -------------------------------------------------
    by_stage: dict[str, list[float]] = {}
    per_cycle_denoise: dict[int, float] = {}
    per_cycle_total: dict[int, float] = {}
    for row in stages:
        by_stage.setdefault(row["stage"], []).append(row["ms"])
        per_cycle_total[row["cycle"]] = per_cycle_total.get(row["cycle"], 0.0) + row["ms"]
        if row["stage"].startswith("denoise"):
            per_cycle_denoise[row["cycle"]] = per_cycle_denoise.get(row["cycle"], 0.0) + row["ms"]

    eager_mean = statistics.mean(eager_e2e)
    compiled_mean = statistics.mean(compiled_e2e)
    denoise_mean = statistics.mean(per_cycle_denoise.values()) if per_cycle_denoise else 0.0
    denoise_share = denoise_mean / eager_mean if eager_mean else 0.0
    cacheable_share = denoise_share * (CACHEABLE_STEPS / num_inference_steps)

    stage_sum_mean = statistics.mean(per_cycle_total.values()) if per_cycle_total else 0.0
    answers = {
        "A_compile_already_wins": {
            "eager_mean_ms": eager_mean,
            "compiled_mean_ms": compiled_mean,
            "compile_speedup_x": eager_mean / compiled_mean if compiled_mean else None,
            "compile_cold_start_ms": cold_start_ms,
            "note": "Any cache must beat the compiled number, not the eager one.",
        },
        "B_cacheable_ceiling": {
            "denoise_stack_mean_ms": denoise_mean,
            "denoise_share_of_eager_cycle": denoise_share,
            "max_reduction_if_all_cacheable_steps_free": cacheable_share,
            "implied_best_case_speedup_x": (
                1.0 / (1.0 - cacheable_share) if cacheable_share < 1 else None
            ),
            "note": f"Assumes {CACHEABLE_STEPS} of {num_inference_steps} steps skippable, eager.",
        },
        "C_where_else_time_goes": {
            "text_encode_mean_ms": statistics.mean(by_stage["text_encode"]) if "text_encode" in by_stage else None,
            "vae_encode_mean_ms": statistics.mean(by_stage["vae_encode"]) if "vae_encode" in by_stage else None,
            "video_prefill_mean_ms": statistics.mean(by_stage["video_prefill"]) if "video_prefill" in by_stage else None,
            "instrumented_sum_mean_ms": stage_sum_mean,
            "unaccounted_mean_ms": eager_mean - stage_sum_mean,
            "note": "text_encode re-runs every cycle on a constant instruction; it is free to cache and a DiT cache cannot claim it.",
        },
    }
    logger.info("ANSWERS=%s", json.dumps(answers, indent=2))

    out = {
        "meta": meta,
        "answers": answers,
        "eager": {"e2e": summarize(eager_e2e), "peak_gib": eager_peak, "raw_ms": eager_e2e},
        "compiled": {"e2e": summarize(compiled_e2e), "peak_gib": compiled_peak, "raw_ms": compiled_e2e},
        "stages_mean_ms": {k: statistics.mean(v) for k, v in sorted(by_stage.items())},
        "stages_p99_ms": {k: pct(v, 0.99) for k, v in sorted(by_stage.items())},
        "stages_raw": stages,
    }
    os.makedirs(os.path.dirname(str(probe.out)), exist_ok=True)
    with open(str(probe.out), "w") as f:
        json.dump(out, f, indent=2)
    logger.info("wrote %s", probe.out)

    repo = os.environ.get("TRACE_REPO", "").strip()
    if repo:
        from huggingface_hub import HfApi
        name = os.environ.get("RUN_NAME", "headroom_v1")
        HfApi().upload_file(
            path_or_fileobj=str(probe.out),
            path_in_repo=f"killchecks/{name}.json",
            repo_id=repo,
            repo_type="dataset",
        )
        logger.info("uploaded to %s killchecks/%s.json", repo, name)


if __name__ == "__main__":
    main()

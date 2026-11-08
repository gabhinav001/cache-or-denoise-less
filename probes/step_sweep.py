"""Measure the compiled per-denoise-step cost by sweeping the step count.

CUDA graphs cannot be instrumented from Python, so the compiled stage split
cannot be read directly. But the denoise loop runs `num_inference_steps` times
over identical shapes while everything else runs exactly once, so end-to-end
latency is linear in the step count:

    e2e(n) = fixed + n * per_step

The slope is the true compiled cost of one denoise step; the intercept is
everything a block cache can never touch (text encode, VAE, video prefill).

The eager sweep is a positive control: we already measured eager directly at
n=10 (51.5 ms/step, ~104 ms fixed), so if the eager fit recovers those numbers
the method is sound and the compiled slope can be trusted.
"""

from __future__ import annotations

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

CACHEABLE_STEPS = 8


def ols(xs, ys):
    """Least squares fit y = a + b*x, with R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def timed(model, kwargs, device, n):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.infer_action(**kwargs)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def sweep(model, base_kwargs, device, *, compiled, steps_list, warmup, iters):
    out = {}
    for n in steps_list:
        k = dict(base_kwargs, num_inference_steps=n, compile_action_infer=compiled)
        for _ in range(warmup):
            timed(model, k, device, n)
        ts = [timed(model, k, device, n) for _ in range(iters)]
        out[n] = ts
        logger.info(
            "%s steps=%2d  mean=%7.2f ms  median=%7.2f ms",
            "compiled" if compiled else "eager   ", n,
            statistics.mean(ts), statistics.median(ts),
        )
    return out


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    probe = OmegaConf.merge(
        OmegaConf.create({
            "warmup": 3, "iters": 25, "seed": 42,
            "steps": [2, 4, 6, 8, 10],
            "out": "/tmp/step_sweep.json",
        }),
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
    torch.manual_seed(int(probe.seed))

    base = {
        "prompt": "pick up the object",
        "input_image": torch.rand(1, 3, height, width) * 2 - 1,
        "action_horizon": action_horizon,
        "proprio": torch.randn(1, int(cfg.data.train.processor.proprio_output_dim)),
        "seed": int(probe.seed),
        "rand_device": str(cfg.EVALUATION.rand_device),
    }
    params = inspect.signature(model.infer_action).parameters
    if "num_video_frames" in params:
        base["num_video_frames"] = (
            action_horizon // int(cfg.data.train.action_video_freq_ratio) + 1
        )
    if "action_infer_mode" in params:
        base["action_infer_mode"] = "idm"

    steps_list = [int(x) for x in probe.steps]
    warmup, iters = int(probe.warmup), int(probe.iters)

    eager = sweep(model, base, device, compiled=False,
                  steps_list=steps_list, warmup=warmup, iters=iters)
    comp = sweep(model, base, device, compiled=True,
                 steps_list=steps_list, warmup=warmup, iters=iters)

    results = {}
    for label, data in (("eager", eager), ("compiled", comp)):
        xs = [float(n) for n in steps_list]
        ys = [statistics.mean(data[n]) for n in steps_list]
        fixed, per_step, r2 = ols(xs, ys)
        total10 = fixed + 10 * per_step
        removable = CACHEABLE_STEPS * per_step
        results[label] = {
            "fixed_overhead_ms": fixed,
            "per_denoise_step_ms": per_step,
            "r2": r2,
            "total_at_10_steps_ms": total10,
            "denoise_share_at_10": (10 * per_step) / total10,
            "removable_ms_if_8_cached": removable,
            "best_case_speedup_x": total10 / (total10 - removable),
            "means_by_steps": {str(n): statistics.mean(data[n]) for n in steps_list},
        }
        logger.info(
            "%-9s fixed=%6.1f ms  per_step=%6.2f ms  R2=%.5f  "
            "denoise_share=%.1f%%  best_case=%.2fx",
            label, fixed, per_step, r2,
            results[label]["denoise_share_at_10"] * 100,
            results[label]["best_case_speedup_x"],
        )

    verdict = {
        "honest_baseline_ms": results["compiled"]["total_at_10_steps_ms"],
        "cache_ceiling_vs_compiled_x": results["compiled"]["best_case_speedup_x"],
        "cache_ceiling_vs_eager_x": results["eager"]["best_case_speedup_x"],
        "compile_speedup_x": (
            results["eager"]["total_at_10_steps_ms"]
            / results["compiled"]["total_at_10_steps_ms"]
        ),
        "uncacheable_fixed_ms_compiled": results["compiled"]["fixed_overhead_ms"],
    }
    logger.info("VERDICT=%s", json.dumps(verdict, indent=2))

    out = {
        "meta": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "dtype": str(dtype),
            "action_horizon": action_horizon,
            "video_size": [height, width],
            "ckpt": str(cfg.ckpt),
            "steps_swept": steps_list,
            "warmup": warmup, "iters": iters,
            "host": platform.platform(),
        },
        "verdict": verdict,
        "fits": results,
        "raw_ms": {"eager": {str(k): v for k, v in eager.items()},
                   "compiled": {str(k): v for k, v in comp.items()}},
    }
    with open(str(probe.out), "w") as f:
        json.dump(out, f, indent=2)

    repo = os.environ.get("TRACE_REPO", "").strip()
    if repo:
        from huggingface_hub import HfApi
        name = os.environ.get("RUN_NAME", "step_sweep_v1")
        HfApi().upload_file(
            path_or_fileobj=str(probe.out),
            path_in_repo=f"killchecks/{name}.json",
            repo_id=repo, repo_type="dataset",
        )
        logger.info("uploaded to %s killchecks/%s.json", repo, name)


if __name__ == "__main__":
    main()

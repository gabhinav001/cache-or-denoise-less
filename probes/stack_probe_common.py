"""Checkpoint and measurement checks shared by the whole-stack probes.

Schema 2 excludes the initial collectors, which named and hashed a checkpoint
without loading it. A successful load is required before collecting any data.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path

SCHEMA_VERSION = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requested_sigma_shift(cfg):
    value = cfg.EVALUATION.get("sigma_shift")
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("EVALUATION.sigma_shift must be finite and positive.")
    return value


def validate_checkpoint_state(model, payload):
    """Reject a loader's silent partial mot restore before evaluating a policy.

    Upstream load_checkpoint uses strict=False and also accepts video-only 'dit'
    checkpoints. Neither a successful call nor a file hash proves the action
    expert was restored. Check the complete named state and shape coverage.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError("Evaluation requires a complete mot checkpoint, not video-only weights")
    expected, stored = model.mot.state_dict(), payload["mot"]
    missing, unexpected = sorted(set(expected) - set(stored)), sorted(set(stored) - set(expected))
    wrong_shape = sorted(k for k in set(expected) & set(stored) if expected[k].shape != stored[k].shape)
    if missing or unexpected or wrong_shape:
        raise ValueError(f"Checkpoint state mismatch: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected, {len(wrong_shape)} wrong shapes; "
                         f"examples={(missing + unexpected + wrong_shape)[:5]}")
    action = sum(k.startswith("mixtures.action.") for k in expected)
    video = sum(k.startswith("mixtures.video.") for k in expected)
    if not action or not video:
        raise ValueError("No restored action/video expert state keys")
    proprio = getattr(model, "proprio_encoder", None)
    proprio_keys = 0
    if proprio is not None:
        state = proprio.state_dict()
        weights = payload.get("proprio_encoder", {})
        if set(state) != set(weights) or any(state[k].shape != weights[k].shape for k in state):
            raise ValueError("Incomplete proprioception checkpoint")
        proprio_keys = len(state)
    return {"verified": True, "mot_keys": len(expected), "action_keys": action,
            "video_keys": video, "proprio_keys": proprio_keys}


def inference_runtime(cfg, model):
    """Record settings that distinguish the checkpoint/scheduler evaluations."""
    override = requested_sigma_shift(cfg)
    mode = (str(cfg.EVALUATION.get("action_infer_mode", "idm"))
            if "action_infer_mode" in inspect.signature(model.infer_action).parameters
            else "first_frame")
    stats = cfg.EVALUATION.get("dataset_stats_path")
    bundle_sha = os.environ.get("WAM_SOURCE_BUNDLE_SHA256")
    if not bundle_sha:
        for mpath in ("/repo/inputs/source_manifest.json",
                      "/workspace/source_manifest.json",
                      str(Path(__file__).resolve().parent.parent / "inputs/source_manifest.json")):
            if os.path.isfile(mpath):
                try:
                    with open(mpath) as mf:
                        bundle_sha = json.load(mf).get("source_bundle_sha256")
                    if bundle_sha:
                        break
                except Exception:
                    pass
    return {"sigma_shift_override": override,
            "source_bundle_sha256": bundle_sha,
            "effective_action_sigma_shift": (float(model.infer_action_scheduler.shift)
                                              if override is None else override),
            "action_infer_mode": mode,
            "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "mixed_precision": str(cfg.mixed_precision),
            "dataset_stats": Path(str(stats)).name if stats else None,
            "dataset_stats_sha256": file_sha256(stats) if stats and Path(str(stats)).is_file() else None}


def load_probe_model(cfg, instantiate):
    """Load the configured checkpoint, using the same API as fw_families.py."""
    import torch

    if not cfg.get("ckpt"):
        raise ValueError("A checkpoint is required; random experts are not a valid probe.")
    path = Path(str(cfg.ckpt)).expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is absent or empty: {path}")
    before = path.stat()
    digest = file_sha256(path)
    dtype = {"no": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    device = str(cfg.EVALUATION.device)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(path))  # Do not record loaded=True until this succeeds.
    model = model.to(device).eval()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while hashing/loading; discard this run.")
    audit = {"loaded": True, "loader": "model.load_checkpoint",
             "name": path.name, "sha256": digest,
             "bytes": after.st_size, "model_dtype": str(dtype),
             "model_class": f"{type(model).__module__}.{type(model).__name__}"}
    return model, audit


def validate_artifact(data, kind):
    """Reject legacy results and incomplete schema-2 load records."""
    meta = data.get("meta", {})
    audit = meta.get("checkpoint_load", {})
    sha = audit.get("sha256", "")
    if (meta.get("schema_version") != SCHEMA_VERSION
            or meta.get("collector") != kind
            or audit.get("loaded") is not True
            or audit.get("loader") != "model.load_checkpoint"
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError(
            "EXCLUDED: this artifact does not establish that the configured checkpoint "
            "was loaded. The legacy stackdiff/stackbeta collectors omitted loading. "
            "Preserve the file and rerun with the corrected collector; do not use its "
            "numbers as evidence about the trained policy.")


def tensor_errors(a, b):
    """Direct maximum and RMS difference of two tensors, accumulated in float64."""
    if a is None or b is None or a.shape != b.shape:
        raise ValueError("A direct comparison requires matching, recorded tensors.")
    d = a.detach().double().cpu() - b.detach().double().cpu()
    return float(d.abs().max()), float(d.square().mean().sqrt())


def load_stack_weights(path, checkpoint_path, n_steps):
    """Do not let an old unloaded-model fit become a closed-loop treatment."""
    with open(path) as fh:
        data = json.load(fh)
    validate_artifact(data, "stackbeta")
    expected_steps = list(range(max(0, n_steps - 2)))
    if (data.get("fit_on") != "train" or not expected_steps
            or data["meta"].get("cached_steps") != expected_steps
            or data["meta"].get("steps") != n_steps):
        raise ValueError("Stack weights were not fit on this schedule's cacheable steps.")
    expected_held = list(range(0, data["episodes_total"], data["holdout_every"]))
    if data.get("episode_ids", {}).get("held") != expected_held:
        raise ValueError("Stack weights do not use the declared per-cell calibration split.")
    if file_sha256(checkpoint_path) != data["meta"]["checkpoint_load"]["sha256"]:
        raise ValueError("Stack weights were fitted with a different checkpoint.")
    weights = {int(k): float(v) for k, v in data.get("beta_by_age", {}).items()}
    if not weights or any(k < 1 or not math.isfinite(v) for k, v in weights.items()):
        raise ValueError("Stack coefficient ages/values are invalid.")
    return weights


def new_stats():
    return {"target_cached_dot": 0.0, "cached_squared_norm": 0.0,
            "target_squared_norm": 0.0, "n_pairs": 0, "n_elements": 0}


def add_stats(stats, target, cached):
    if target.shape != cached.shape:
        raise ValueError("Calibration pairs must have matching shapes.")
    t, c = target.detach().double().cpu().reshape(-1), cached.detach().double().cpu().reshape(-1)
    stats["target_cached_dot"] += float(t.dot(c))
    stats["cached_squared_norm"] += float(c.dot(c))
    stats["target_squared_norm"] += float(t.dot(t))
    stats["n_pairs"] += 1
    stats["n_elements"] += t.numel()


def add_step_stats(stats, target, cached, steps):
    """Pool only the steps the deployed cache will reuse."""
    for step in steps:
        add_stats(stats, target[step], cached[step])


def fitted_beta(stats):
    den = stats["cached_squared_norm"]
    return stats["target_cached_dot"] / den if den > 0 else None


def score_beta(stats, beta):
    """Evaluate a supplied coefficient without refitting it on these samples."""
    if beta is None or not stats["n_elements"]:
        return None
    if not math.isfinite(beta):
        raise ValueError("Coefficient must be finite.")
    tt, tc, cc = (stats[k] for k in
                  ("target_squared_norm", "target_cached_dot", "cached_squared_norm"))
    sse = max(0.0, tt - 2 * beta * tc + beta * beta * cc)
    plain_sse = max(0.0, tt - 2 * tc + cc)
    return {"beta": beta, "mse": sse / stats["n_elements"],
            "plain_mse": plain_sse / stats["n_elements"],
            "relative_squared_error": sse / tt if tt > 0 else None,
            "relative_mse_reduction": 1 - sse / plain_sse if plain_sse > 0 else None}


def episode_split(index, holdout_every=4):
    if holdout_every < 2:
        raise ValueError("holdout_every must leave both training and held-out episodes.")
    return "held" if index % holdout_every == 0 else "train"

"""CPU regressions for checkpoint loading and whole-stack measurement mistakes."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import report_stackbeta
import report_stackdiff
from stack_probe_common import (SCHEMA_VERSION, add_stats, add_step_stats,
                                episode_split, file_sha256, fitted_beta,
                                inference_runtime, load_probe_model, load_stack_weights, new_stats,
                                requested_sigma_shift,
                                score_beta, tensor_errors, validate_artifact)

HERE = Path(__file__).resolve().parent


class Config(dict):
    __getattr__ = dict.__getitem__


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.0))

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, x):
        return self.gain * x


class StackProbeAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / "tiny.pt"
        torch.save({"gain": torch.tensor(7.0)}, self.ckpt)
        self.cfg = Config(ckpt=str(self.ckpt), model={}, mixed_precision="no",
                          EVALUATION=Config(device="cpu"))

    def loaded_meta(self, kind):
        model, audit = load_probe_model(self.cfg, lambda *_a, **_kw: TinyModel())
        self.assertEqual(float(model(torch.tensor(2.0)).detach()), 14.0)
        self.assertFalse(model.training)
        self.assertEqual(audit["sha256"], file_sha256(self.ckpt))
        return {"schema_version": SCHEMA_VERSION, "collector": kind,
                "checkpoint_load": audit, "steps": 10, "cached_steps": list(range(8))}

    def test_actual_checkpoint_is_loaded_before_use(self):
        calls = []
        def instantiate(*a, **kw):
            calls.append(kw)
            return TinyModel()
        model, audit = load_probe_model(self.cfg, instantiate)
        self.assertEqual(float(model(torch.tensor(3.0)).detach()), 21.0)
        self.assertEqual(calls, [{"model_dtype": torch.float32, "device": "cpu"}])
        self.assertTrue(audit["loaded"])

    def test_runtime_preserves_explicit_scheduler_and_source_identity(self):
        model = TinyModel()
        model.infer_action = lambda prompt: None
        model.infer_action_scheduler = Config(shift=1.0)
        self.cfg.EVALUATION["sigma_shift"] = 5.0
        bundle = "a" * 64
        with mock.patch.dict("os.environ", {"WAM_SOURCE_BUNDLE_SHA256": bundle}):
            runtime = inference_runtime(self.cfg, model)
        self.assertEqual(runtime["effective_action_sigma_shift"], 5.0)
        self.assertEqual(runtime["action_infer_mode"], "first_frame")
        self.assertEqual(runtime["source_bundle_sha256"], bundle)
        self.cfg.EVALUATION["sigma_shift"] = None
        self.assertEqual(inference_runtime(self.cfg, model)["effective_action_sigma_shift"], 1.0)

    def test_optional_model_mode_is_recorded_and_invalid_shift_rejected(self):
        model = TinyModel()
        model.infer_action = lambda prompt, action_infer_mode="idm": None
        model.infer_action_scheduler = Config(shift=1.0)
        self.cfg.EVALUATION["action_infer_mode"] = "first_frame"
        self.assertEqual(inference_runtime(self.cfg, model)["action_infer_mode"], "first_frame")
        for value in (0, -1, float("nan"), float("inf")):
            self.cfg.EVALUATION["sigma_shift"] = value
            with self.assertRaises(ValueError):
                requested_sigma_shift(self.cfg)

    def test_missing_and_empty_checkpoint_fail_before_instantiation(self):
        for name in ("missing.pt", "empty.pt"):
            path = self.root / name
            if name.startswith("empty"):
                path.touch()
            self.cfg["ckpt"] = str(path)
            with self.assertRaises(FileNotFoundError):
                load_probe_model(self.cfg, lambda *_a, **_kw: self.fail("instantiated too early"))

    def test_load_failure_does_not_become_a_successful_audit(self):
        self.ckpt.write_bytes(b"not a checkpoint")
        with self.assertRaises(Exception):
            load_probe_model(self.cfg, lambda *_a, **_kw: TinyModel())

    def test_collectors_use_the_checked_loader(self):
        for name in ("fw_stackdiff.py", "fw_stackbeta.py"):
            tree = ast.parse((HERE / name).read_text())
            calls = [n.func.id for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn("load_probe_model", calls)
            self.assertNotIn("instantiate", calls)

    def test_equal_errors_to_full_do_not_imply_pairwise_equality(self):
        full, pb, ws = torch.zeros(2), torch.tensor([1., 0.]), torch.tensor([0., 1.])
        self.assertEqual(tensor_errors(pb, full), tensor_errors(ws, full))
        self.assertEqual(tensor_errors(pb, ws), (1.0, 1.0))
        data = {"meta": self.loaded_meta("stackdiff"),
                "rows": [{"arm": "full_repeat", "compiled": False, "age": 1, "action_rms": 0.}],
                "comparisons": [{"compiled": False, "age": 1, "left": "perblock",
                                 "right": "wholestack", "action_rms": 1.,
                                 "action_max": 1., "hidden_rms": 1.}]}
        path = self.root / "difference.json"
        path.write_text(json.dumps(data))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(report_stackdiff.main(str(path)), 0)
        self.assertIn("action RMS=1", output.getvalue())

    def test_training_split_matches_original_per_cell_split(self):
        self.assertEqual([i for i in range(24) if episode_split(i) == "held"],
                         [0, 4, 8, 12, 16, 20])
        self.assertEqual(sum(episode_split(i) == "train" for i in range(24)), 18)

    def test_uncached_final_steps_do_not_change_the_fit(self):
        target = {i: torch.tensor([0.5 if i < 8 else 10000.]) for i in range(10)}
        cached = {i: torch.ones(1) for i in range(10)}
        stats = new_stats()
        add_step_stats(stats, target, cached, range(8))
        self.assertEqual(fitted_beta(stats), 0.5)
        self.assertEqual(stats["n_pairs"], 8)

    def test_heldout_scores_the_training_weight_without_refitting(self):
        train, held = new_stats(), new_stats()
        c = torch.tensor([1., 2.])
        add_stats(train, 0.5 * c, c)
        add_stats(held, 2.0 * c, c)
        beta = fitted_beta(train)
        self.assertEqual(beta, 0.5)
        self.assertEqual(fitted_beta(held), 2.0)
        scores = score_beta(held, beta)
        self.assertEqual(scores["mse"], 5.625)
        self.assertEqual(scores["plain_mse"], 2.5)
        data = {"meta": self.loaded_meta("stackbeta"),
                "sufficient_statistics": {"train": {"1": train}, "held": {"1": held}}}
        path = self.root / "fit.json"
        path.write_text(json.dumps(data))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(report_stackbeta.main(str(path)), 0)
        self.assertIn("beta=0.5, held-out MSE=5.625", output.getvalue())

    def test_old_artifacts_are_excluded_and_cannot_supply_weights(self):
        for kind, name, reporter in (
                ("stackdiff", "fw_stackdiff.json", report_stackdiff),
                ("stackbeta", "stack_shrink_idm.json", report_stackbeta)):
            # A legacy result can name a real checkpoint and contain plausible
            # numbers without having loaded it. Keep this fixture independent
            # of future corrected GPU collections replacing a result path.
            data = {"meta": {"ckpt": self.ckpt.name}, "beta_by_age": {"1": 0.995}}
            path = self.root / name
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "EXCLUDED"):
                validate_artifact(data, kind)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(reporter.main(str(path)), 2)
        with self.assertRaisesRegex(ValueError, "EXCLUDED"):
            load_stack_weights(path, self.ckpt, 10)

    def test_weight_consumer_checks_checkpoint_and_cache_scope(self):
        data = {"meta": self.loaded_meta("stackbeta"), "fit_on": "train",
                "episodes_total": 24, "holdout_every": 4,
                "episode_ids": {"held": [0, 4, 8, 12, 16, 20]}, "beta_by_age": {"1": 0.5}}
        path = self.root / "weights.json"
        path.write_text(json.dumps(data))
        self.assertEqual(load_stack_weights(path, self.ckpt, 10), {1: 0.5})
        with self.assertRaisesRegex(ValueError, "cacheable steps"):
            load_stack_weights(path, self.ckpt, 6)
        self.ckpt.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            load_stack_weights(path, self.ckpt, 10)


if __name__ == "__main__":
    unittest.main()

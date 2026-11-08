"""Unit tests for report_tau8_fresh validation and statistical checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import report_tau8_fresh as report

ROOT = Path(__file__).resolve().parent.parent


def make_valid_fixtures():
    """Create valid fixtures for seed42 and seed7 matching experiment parameters."""
    fixtures = {}
    for seed in (42, 7):
        plain_rows = []
        atten_rows = []
        for t in range(10):
            for e in range(10):
                state_hash = f"state_hash_{t}_{e:02d}"
                # Seed 42: 66 vs 81; Seed 7: 70 vs 86
                p_succ = (t * 10 + e) < (66 if seed == 42 else 70)
                a_succ = (t * 10 + e) < (81 if seed == 42 else 86)
                plain_rows.append({"task": t, "episode": e, "success": p_succ, "init_state": state_hash})
                atten_rows.append({"task": t, "episode": e, "success": a_succ, "init_state": state_hash})

            meta_base = {
                "seed": seed,
                "initial_latent": "uniform_noise_standard",
                "provenance": {"commit": "rev-test", "host_tag": "test-host"},
                "runtime": {
                    "checkpoint_sha256": report.EXPECTED_CHECKPOINT_SHA256,
                    "dataset_stats_sha256": "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638",
                    "model_class": "fastwam.models.wan22.fastwam.FastWAM",
                    "effective_action_sigma_shift": 1.0,
                    "action_infer_mode": "first_frame",
                    "mixed_precision": "bf16",
                }
            }
            p_meta = copy.deepcopy(meta_base)
            p_meta["configs"] = ["c3ache_tau8"]
            a_meta = copy.deepcopy(meta_base)
            a_meta["configs"] = ["c3ache_tau8_shrink"]

            fixtures[f"seed{seed}_plain.json"] = {"meta": p_meta, "rows": plain_rows}
            fixtures[f"seed{seed}_attenuated.json"] = {"meta": a_meta, "rows": atten_rows}
    return fixtures


class TestReportTau8Fresh(unittest.TestCase):
    def test_actual_data_passes(self):
        actual_dir = ROOT / "results" / "tau8_fresh"
        if not actual_dir.exists():
            self.skipTest(f"{actual_dir} does not exist")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = report.analyze_dir(str(actual_dir))
        self.assertIn("seed42", res["seeds"])
        self.assertIn("seed7", res["seeds"])
        self.assertEqual(res["seeds"]["seed42"]["plain_succ"], 66)
        self.assertEqual(res["seeds"]["seed42"]["atten_succ"], 81)
        self.assertAlmostEqual(res["seeds"]["seed42"]["diff_pts"], 15.0)
        self.assertEqual(res["seeds"]["seed7"]["plain_succ"], 70)
        self.assertEqual(res["seeds"]["seed7"]["atten_succ"], 86)
        self.assertAlmostEqual(res["seeds"]["seed7"]["diff_pts"], 16.0)
        self.assertAlmostEqual(res["mean_diff_pts"], 15.5)
        self.assertEqual(res["state_clustered_bootstrap_95ci"], [9.0, 22.0])

    def test_valid_fixtures_pass(self):
        fixtures = make_valid_fixtures()
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                res = report.analyze_dir(td)
            self.assertEqual(res["mean_diff_pts"], 15.5)

    def test_incomplete_coverage_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["rows"].pop()  # 99 rows instead of 100
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("expected 100 rows", str(ctx.exception))

    def test_duplicate_episode_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["rows"][1] = copy.deepcopy(fixtures["seed42_plain.json"]["rows"][0])
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("duplicate episode", str(ctx.exception))

    def test_missing_init_state_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["rows"][0]["init_state"] = ""
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("missing init_state", str(ctx.exception))

    def test_state_hash_mismatch_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["rows"][5]["init_state"] = "mismatched_hash"
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("state hash mismatch between arms", str(ctx.exception))

    def test_process_isolation_multiple_configs_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["meta"]["configs"] = ["c3ache_tau8", "c3ache_tau8_shrink"]
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("singleton configs list", str(ctx.exception))

    def test_wrong_config_rejected(self):
        fixtures = make_valid_fixtures()
        fixtures["seed42_plain.json"]["meta"]["configs"] = ["c3ache_tau4"]
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("expected config c3ache_tau8", str(ctx.exception))

    def test_missing_partner_file_rejected(self):
        fixtures = make_valid_fixtures()
        del fixtures["seed42_attenuated.json"]
        with tempfile.TemporaryDirectory() as td:
            for fname, content in fixtures.items():
                (Path(td) / fname).write_text(json.dumps(content))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError) as ctx:
                    report.analyze_dir(td)
            self.assertIn("Mismatched arm files", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

"""Fail-closed checks for the staged baseline audit; synthetic CPU fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from report_baseline_audit import ARMS, read_collection, validate_arm


def fixture(tag):
    runtime = dict(source_bundle_sha256="a" * 64, checkpoint_sha256="b" * 64,
                   dataset_stats_sha256="c" * 64, dataset_stats="libero_uncond_2cam224_dataset_stats.json",
                   sigma_shift_override=5.0, effective_action_sigma_shift=5.0,
                   action_infer_mode="first_frame", compile_action_infer=False,
                   model_class="test.FastWAM", mixed_precision="bf16")
    runtime["checkpoint_state_coverage"] = dict(verified=True, action_keys=1, video_keys=1, mot_keys=2, proprio_keys=0)
    meta = dict(suite="libero_goal", perturb="none", seed=42, configs=[ARMS[tag]],
                n_tasks=1, n_episodes=2, num_inference_steps=10, replan=10, no_reseed=0,
                timing=1, warmup_episodes=1, skip_cells=240, initial_latent="latent",
                runtime=runtime, provenance=dict(commit="revision", host_tag="host", gpu="GPU",
                python="3.11", torch="2.7", mujoco="3.3.2", ckpt="libero_uncond_2cam224.pt"))
    stock = ARMS[tag] == "stock"
    rows = []
    for ep in range(2):
        rows.append(dict(task=0, episode=ep, config=ARMS[tag], suite="libero_goal", success=True,
                         init_state=f"state{ep}", cycles=2, steps=20, refresh_cycles=2 if stock else 1,
                         reuse_cycles=0 if stock else 1, denoise_calls=2, warmup=ep == 0,
                         ms_per_cycle=100.0, cycle_ms=[100.0, 100.0], cycle_refresh=[True, stock],
                         whole_cache_hits=8, whole_cache_misses=0))
    return dict(meta=meta, rows=rows)


class BaselineAuditTest(unittest.TestCase):
    def collection(self, mutation=None, gate_mutation=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "seed42"
            root.mkdir()
            runs = {tag: fixture(tag) for tag in ARMS}
            if mutation:
                mutation(runs)
            for tag, data in runs.items():
                (root / (tag + ".json")).write_text(json.dumps(data))
            gate = dict(schema_version=3, **{"pass": True}, runtime=fixture("00_stock_a")["meta"]["runtime"],
                        gpu="GPU", ckpt="libero_uncond_2cam224.pt", identity_err=0.001,
                        repeat_full_err=0.0, skip_identity_err=0.001, whole_identity_err=0.001,
                        margin=2, cells=240, whole_identity_counts=dict(hits=8, misses=0))
            gate['cache_integrity'] = {n: dict(entries=c, checks=2, unchanged=True, finite=True)
                for n,c in (("old_block",240),("fresh_block",240),("old_stack",8),("fresh_stack",8))}
            if gate_mutation:
                gate_mutation(gate)
            (root / "gate.json").write_text(json.dumps(gate))
            return read_collection(root, "smoke", "original", samples=100)

    def test_smoke_cannot_be_claimed_as_full_evaluation(self):
        report = self.collection()
        self.assertTrue(report["valid"])
        self.assertFalse(report["complete_500_state_comparison"])
        self.assertEqual(report["n_paired_states"], 2)

    def test_corrupted_cache_is_rejected_even_with_small_replay_error(self):
        with self.assertRaisesRegex(ValueError, "cache integrity"):
            self.collection(gate_mutation=lambda g: g['cache_integrity']['old_stack'].__setitem__('unchanged', False))

    def test_legacy_gate_cannot_be_relabelled_as_validated(self):
        with self.assertRaisesRegex(ValueError, "schema-3"):
            self.collection(gate_mutation=lambda g: g.__setitem__('schema_version', 2))

    def test_repeat_disagreement_blocks_scaling(self):
        report = self.collection(lambda d: d["03_stock_b"]["rows"][1].__setitem__("success", False))
        self.assertFalse(report["valid"])
        self.assertFalse(report["stock_repeat"]["pass"])

    def test_changed_checkpoint_is_not_a_paired_comparison(self):
        with self.assertRaisesRegex(ValueError, "pairing differs"):
            self.collection(lambda d: d["01_whole_tau8"]["meta"]["runtime"].__setitem__("checkpoint_sha256", "d" * 64))

    def test_duplicate_state_is_not_coverage(self):
        with self.assertRaisesRegex(ValueError, "Duplicate task-state"):
            self.collection(lambda d: d["00_stock_a"]["rows"].__setitem__(1, copy.deepcopy(d["00_stock_a"]["rows"][0])))

    def test_scheduled_reuse_without_observed_hits_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "observed cache hits"):
            self.collection(lambda d: d["01_whole_tau8"]["rows"][1].__setitem__("whole_cache_hits", 0))

    def test_wrong_scheduler_is_rejected(self):
        data = fixture("00_stock_a")
        data["meta"]["runtime"]["effective_action_sigma_shift"] = 1.0
        with self.assertRaisesRegex(ValueError, "scheduler shift"):
            validate_arm(data, "00_stock_a", "smoke", "original", 42)


if __name__ == "__main__":
    unittest.main()

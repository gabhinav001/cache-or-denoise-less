"""CPU fixtures for coverage, pairing, and collection isolation in the report."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import report_published as report


def fixture(tasks=10, episodes=50, seed=42, revision="rev-a", wins=None):
    n = tasks * episodes
    wins = wins or {arm: n for arm in report.ARMS}
    rows = [
        {"suite": "libero_goal", "task": t, "episode": e, "config": arm,
         "success": t * episodes + e < wins[arm], "init_state": f"state-{t}-{e}"}
        for arm in report.ARMS for t in range(tasks) for e in range(episodes)
    ]
    return {"meta": {"suite": "libero_goal", "n_tasks": tasks,
                     "n_episodes": episodes, "seed": seed,
                     "provenance": {"commit": revision, "host_tag": "fixture-host"}},
            "rows": rows}


class PublishedReportTest(unittest.TestCase):
    def run_report(self, artifacts):
        with tempfile.TemporaryDirectory() as directory:
            for name, data in artifacts.items():
                (Path(directory) / name).write_text(json.dumps(data))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = report.main(directory)
        return code, output.getvalue()

    def assert_no_equivalence_verdict(self, output):
        self.assertNotIn("The harness reproduces", output)
        self.assertNotIn("checkpoint is not implicated", output)
        self.assertNotIn("their method", output)
        self.assertIn("reference equivalence remains unverified", output)

    def test_two_episode_smoke_is_not_full_reproduction(self):
        code, output = self.run_report({"goal42.json": fixture(tasks=1, episodes=2)})
        self.assertEqual(code, 0)
        self.assertIn("SMOKE / SUBSET / INCOMPLETE", output)
        self.assertNotIn("COMPLETE LOCAL COLLECTION:", output)
        self.assertIn("actual rows 6", output)
        self.assertIn("paired on 2 verified initial states", output)
        self.assert_no_equivalence_verdict(output)

    def test_complete_matching_rates_do_not_establish_reference_equivalence(self):
        wins = {"full": 482, "whole_tau8": 482, "c3ache_tau8": 338}
        code, output = self.run_report({"goal42.json": fixture(wins=wins)})
        self.assertEqual(code, 0)
        self.assertIn("COMPLETE LOCAL COLLECTION:", output)
        self.assertIn("96.4%", output)
        self.assertIn("local whole-stack implementation", output)
        self.assertIn("paired on 500 verified initial states: +28.8 points", output)
        self.assert_no_equivalence_verdict(output)

    def test_declared_full_run_with_partial_rows_stays_incomplete(self):
        data = fixture()
        data["rows"] = [r for r in data["rows"] if r["task"] == 0]
        code, output = self.run_report({"partial.json": data})
        self.assertEqual(code, 0)
        self.assertIn("actual rows 150", output)
        self.assertNotIn("COMPLETE LOCAL COLLECTION:", output)

    def test_gate_and_offline_diagnostic_are_skipped(self):
        code, output = self.run_report({
            "goal42.json": fixture(tasks=1, episodes=2),
            "gate.json": {"pass": True},
            "fw_stackdiff.json": {"meta": {"steps": 10}, "rows": [{"arm": "full"}]},
        })
        self.assertEqual(code, 0)
        self.assertIn("Skipped unrelated gates/diagnostics", output)
        self.assertEqual(output.count("Collection:"), 1)

    def test_seeds_and_revisions_are_reported_separately(self):
        code, output = self.run_report({
            "goal42.json": fixture(tasks=1, episodes=2),
            "goal7.json": fixture(tasks=1, episodes=2, seed=7, revision="rev-b"),
        })
        self.assertEqual(code, 0)
        self.assertIn("MULTIPLE COLLECTIONS", output)
        self.assertEqual(output.count("Collection:"), 2)
        self.assertEqual(output.count("actual rows 6"), 2)
        self.assertEqual(output.count("paired on 2 verified initial states"), 2)
        self.assertIn("Seed 42", output)
        self.assertIn("Seed 7", output)
        self.assertIn("revision rev-a", output)
        self.assertIn("revision rev-b", output)

    def test_duplicate_episode_fails_instead_of_overwriting(self):
        data = fixture(tasks=1, episodes=2)
        data["rows"].append(dict(data["rows"][0]))
        code, output = self.run_report({"duplicate.json": data})
        self.assertEqual(code, 1)
        self.assertIn("duplicate episode", output)
        self.assertNotIn("Collection:", output)

    def test_mismatched_state_prevents_pairing_and_full_coverage_claim(self):
        data = fixture()
        row = next(r for r in data["rows"] if r["config"] == "whole_tau8")
        row["init_state"] = "another-initial-state"
        code, output = self.run_report({"different-state.json": data})
        self.assertEqual(code, 0)
        self.assertIn("Paired cache contrast omitted", output)
        self.assertNotIn("COMPLETE LOCAL COLLECTION:", output)


if __name__ == "__main__":
    unittest.main()

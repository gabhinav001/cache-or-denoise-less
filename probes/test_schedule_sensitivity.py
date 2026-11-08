"""Checks for silent corruption of the paper's pairing and sensitivity analysis."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from report_schedule_sensitivity import bootstrap, contrast, read_run, analyze

ROOT = Path(__file__).resolve().parents[1] / "results"


class ScheduleSensitivityTest(unittest.TestCase):
    def validate_changed(self, mutate):
        data = json.loads((ROOT / "statescale/s2.json").read_text())
        mutate(data)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s2.json"
            path.write_text(json.dumps(data))
            return read_run(path, 42, 2, 10, 50, ["stock"])

    def test_duplicate_rows_cannot_be_hidden_by_a_dictionary(self):
        with self.assertRaisesRegex(ValueError, "Duplicate/missing"):
            self.validate_changed(lambda d: d["rows"].__setitem__(-1, copy.deepcopy(d["rows"][0])))

    def test_non_boolean_success_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-boolean"):
            self.validate_changed(lambda d: d["rows"][0].__setitem__("success", "false"))

    def test_missing_timing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid timing"):
            self.validate_changed(lambda d: d["rows"][1].__setitem__("ms_per_cycle", float("nan")))

    def test_equal_task_weighting_after_unequal_exclusion(self):
        # One task wins 1/1, another loses 3/3: equal-task delta is zero,
        # whereas a pooled-state mean would be -50 pp.
        a = {(0, 0): {"success": True}, **{(1, i): {"success": False} for i in range(3)}}
        b = {k: {"success": not v["success"]} for k, v in a.items()}
        p = contrast({"a": a, "b": b}, "a", "b", sorted(a), "toy", 100)
        self.assertEqual(p["difference_pp_equal_task"], 0)
        self.assertEqual(p["ci95_pp"], [0, 0])
        self.assertEqual(p["leave_one_task_out_range_pp"], [-100, 100])

    def test_intervals_do_not_depend_on_other_comparisons(self):
        first = bootstrap([[1, 0, -1], [0, 0, 1]], "same", 100)
        bootstrap([[1, -1]], "unrelated", 100)
        self.assertEqual(first, bootstrap([[1, 0, -1], [0, 0, 1]], "same", 100))

    def test_raw_evidence_and_exclusion_coverage(self):
        report = analyze(ROOT, samples=100)
        self.assertEqual(len(report["sources"]), 24)
        for c, expected in zip(report["collections"], (3.3, 6.2)):
            p = c["contrasts"]["excluding_pilot_470"][0]
            self.assertEqual(p["states"], 470)
            self.assertAlmostEqual(p["difference_pp_equal_task"], expected)
            self.assertGreater(p["leave_one_task_out_range_pp"][0], 0)


if __name__ == "__main__":
    unittest.main()

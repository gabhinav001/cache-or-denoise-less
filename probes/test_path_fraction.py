"""CPU checks for the cached-prefix controls added to fw_libero.py.

Covers the two things the shift experiments depend on: that an explicit
cache_prefix caches exactly the leading steps it names, and that the recorded
share of the flow path matches FastWAM's scheduler (sigma = s*u/(1+(s-1)*u)).
"""
import contextlib
import importlib.util
import os
import sys
import types
import unittest

for _m in ("hydra", "hydra.utils", "omegaconf"):
    if _m not in sys.modules:
        _stub = types.ModuleType(_m)
        _stub.main = lambda *a, **k: (lambda f: f)
        _stub.instantiate = lambda *a, **k: None
        _stub.open_dict = lambda cfg: contextlib.nullcontext()
        _stub.DictConfig = dict
        _stub.OmegaConf = types.SimpleNamespace(merge=lambda *a: {}, create=lambda *a: {})
        sys.modules[_m] = _stub
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("fw_libero", os.path.join(_here, "fw_libero.py"))
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)


def sigma(k, n, s):
    u = 1.0 - k / n
    return s * u / (1.0 + (s - 1.0) * u)


class PathFraction(unittest.TestCase):
    def test_published_rule_at_both_shifts(self):
        # 8 of 10 steps: 80% of the path at s=1, 44.4% at s=5
        self.assertAlmostEqual(fw.cached_path_fraction(10, 8, 1.0), 0.8, places=9)
        self.assertAlmostEqual(fw.cached_path_fraction(10, 8, 5.0), 1 - 1 / 1.8, places=9)

    def test_matches_scheduler_formula(self):
        for n in (1, 2, 4, 10):
            for s in (1.0, 3.0, 5.0):
                for m in range(n + 1):
                    self.assertAlmostEqual(fw.cached_path_fraction(n, m, s),
                                           1 - sigma(m, n, s), places=12)

    def test_endpoints(self):
        self.assertEqual(fw.cached_path_fraction(10, 0, 5.0), 0.0)
        self.assertAlmostEqual(fw.cached_path_fraction(10, 10, 5.0), 1.0, places=12)
        with self.assertRaises(ValueError):
            fw.cached_path_fraction(10, 11, 1.0)


class Prefix(unittest.TestCase):
    def test_default_is_all_but_last_two(self):
        for n, keep in ((10, 8), (4, 2), (2, 0), (1, 0)):
            steps = {k for k, _ in fw.placements(None, n_steps=n)["c3ache"]}
            self.assertEqual(steps, set(range(keep)))

    def test_explicit_prefix_caches_leading_steps_in_every_block(self):
        for m in (0, 3, 4, 5, 9):
            cells = fw.placements(None, n_steps=10, prefix=m)["c3ache"]
            self.assertEqual({k for k, _ in cells}, set(range(m)))
            self.assertEqual(len(cells), m * fw.N_BLOCKS)

    def test_negative_prefix_means_default(self):
        self.assertEqual(fw.placements(None, n_steps=10, prefix=-1)["c3ache"],
                         fw.placements(None, n_steps=10)["c3ache"])

    def test_prefix_beyond_schedule_is_refused(self):
        with self.assertRaises(ValueError):
            fw.placements(None, n_steps=4, prefix=5)


class CudaGraphGuard(unittest.TestCase):
    def test_flag_defaults_off(self):
        # the shim must stay active unless explicitly disabled for stock timing
        self.assertFalse(fw.CUDAGRAPHS)


if __name__ == "__main__":
    unittest.main()

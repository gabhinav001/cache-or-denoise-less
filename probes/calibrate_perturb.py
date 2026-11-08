"""Does a perturbation actually change what the policy sees, and by how much?

test_perturb.py proves each perturbation modifies the thing it claims to modify.
It cannot prove that modification reaches the rendered image -- moving a light in
the simulator changes nothing if the scene is ambient-dominated, or if the
renderer never re-reads the model. That gap is dangerous: an inert perturbation
produces a confident "the correction is robust" from a run with no shift.

This renders the same initial state with and without each perturbation and
reports the pixel change, so magnitudes are chosen from measurement instead of
guesswork. Needs the simulator but not the model, so it is cheap.

usage: calibrate_perturb.py [suite] [spec ...]
"""
from __future__ import annotations

import os, sys
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: F401  (fw_libero patches torch.load on import)
from fw_libero import make_perturb

SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_goal"
SPECS = sys.argv[2:] or ["light:0.10", "light:0.50", "light:1.00",
                         "cam:0.005", "cam:0.02", "cam:0.05",
                         "bright:0.15", "noise:0.05", "blur:2"]


def images(obs):
    return {k: v for k, v in obs.items()
            if k.endswith("_image") and hasattr(v, "shape") and getattr(v, "ndim", 0) == 3}


def main():
    sys.path.insert(0, "/w/experiments/libero")
    import eval_libero_single as E
    from libero_utils import get_libero_env, get_libero_dummy_action
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(0)
    inits = suite.get_task_init_states(0)

    def rollout(spec, steps=12):
        env, _ = get_libero_env(task, E.LIBERO_ENV_RESOLUTION, 42)
        env.reset(); obs = env.set_init_state(inits[0])
        ofn, efn = make_perturb(spec, 10007)
        if efn is not None:
            efn(env); obs, _, _, _ = env.step(get_libero_dummy_action())
        for _ in range(steps):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        if ofn is not None: obs = ofn(obs)
        out = {k: v.astype(np.float32) for k, v in images(obs).items()}
        env.close()
        return out

    base = rollout("none")
    print(f"\n  suite {SUITE}, task 0, {len(base)} camera(s)\n")
    print(f"  {'perturbation':16s} {'mean |dpix|':>12s} {'max |dpix|':>11s}  verdict")
    print("  " + "-" * 62)
    for spec in SPECS:
        try:
            cur = rollout(spec)
        except Exception as e:
            print(f"  {spec:16s} {'ERROR':>12s}  {type(e).__name__}: {e}"); continue
        ds = [float(np.abs(cur[k] - base[k]).mean()) for k in base if k in cur]
        mx = [float(np.abs(cur[k] - base[k]).max()) for k in base if k in cur]
        d, m = (max(ds) if ds else 0.0), (max(mx) if mx else 0.0)
        verdict = ("INERT -- will not test anything" if d < 0.5 else
                   "subtle" if d < 3 else
                   "clear" if d < 25 else
                   "severe -- may floor the policy")
        print(f"  {spec:16s} {d:12.2f} {m:11.1f}  {verdict}")
    print("\n  Pixels are 0-255. A perturbation the policy cannot see is not a test of\n"
          "  robustness; one that floors full compute is not either.\n")


if __name__ == "__main__":
    main()

"""Verify every perturbation actually perturbs, and does so reproducibly.

A perturbation that silently does nothing is the worst possible failure here: it
would produce "the correction survives distribution shift" from a run where there
was no shift. So each kind is checked to (a) change the image measurably, (b) be
deterministic given its seed, and (c) differ across seeds -- and the magnitudes
are checked to be ordered, so a larger setting really is a larger shift.

No GPU, no simulator. The sim-space kinds (light, cam) are checked against a stub
that records what they touch, since exercising them for real needs mujoco.
"""
from __future__ import annotations
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _m in ("hydra", "hydra.utils", "omegaconf", "torch"):
    if _m not in sys.modules:
        stub = types.ModuleType(_m)
        stub.__path__ = []                      # make it a package, so submodules resolve
        if _m == "hydra": stub.main = lambda *a, **k: (lambda f: f)
        if _m == "hydra.utils": stub.instantiate = lambda *a, **k: None
        if _m == "omegaconf":
            stub.DictConfig = dict
            stub.open_dict = lambda cfg: __import__("contextlib").nullcontext()
            stub.OmegaConf = types.SimpleNamespace(merge=lambda *a: {}, create=lambda *a: {})
        if _m == "torch":
            stub.load = lambda *a, **k: None
            stub.compile = lambda *a, **k: (lambda f: f)
            stub._dynamo = types.SimpleNamespace(config=types.SimpleNamespace())
        sys.modules[_m] = stub
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fw_libero", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fw_libero.py"))
fw = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(fw)
except Exception as e:
    print(f"could not import fw_libero ({e}); perturbation check skipped"); raise SystemExit(1)

fails = 0
def check(name, cond, detail=""):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond: fails += 1

rng = np.random.RandomState(0)
base = {"agentview_image": rng.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7)}

print("none is a no-op")
o, e = fw.make_perturb("none", 0)
check("returns (None, None)", o is None and e is None)

print("\nimage-space perturbations change the image")
for spec in ("bright:0.15", "contrast:0.30", "noise:0.05", "blur:2"):
    o, e = fw.make_perturb(spec, 1)
    out = o(base)
    d = float(np.abs(out["agentview_image"].astype(float) - base["agentview_image"].astype(float)).mean())
    check(f"{spec:16s} changes pixels", d > 0.5, f"mean |Δ| = {d:.2f}")
    check(f"{spec:16s} leaves non-image keys alone",
          np.array_equal(out["robot0_joint_pos"], base["robot0_joint_pos"]))
    check(f"{spec:16s} preserves dtype/shape",
          out["agentview_image"].dtype == np.uint8 and out["agentview_image"].shape == (128,128,3))
    check(f"{spec:16s} perturbs the wrist camera too",
          not np.array_equal(out["robot0_eye_in_hand_image"], base["robot0_eye_in_hand_image"]))

print("\nmagnitude is monotone (a bigger setting is a bigger shift)")
for kind in ("bright", "contrast", "noise"):
    ds = []
    for m in (0.05, 0.15, 0.30):
        o, _ = fw.make_perturb(f"{kind}:{m}", 7)
        ds.append(float(np.abs(o(base)["agentview_image"].astype(float)
                               - base["agentview_image"].astype(float)).mean()))
    check(f"{kind:10s} monotone", ds[0] < ds[1] < ds[2],
          " < ".join(f"{d:.1f}" for d in ds))

print("\nreproducible given a seed, and different across seeds")
o1, _ = fw.make_perturb("noise:0.05", 42); a = o1(base)["agentview_image"]
o2, _ = fw.make_perturb("noise:0.05", 42); b = o2(base)["agentview_image"]
o3, _ = fw.make_perturb("noise:0.05", 43); c = o3(base)["agentview_image"]
check("same seed -> identical", np.array_equal(a, b))
check("different seed -> different", not np.array_equal(a, c))

print("\nsimulator-space perturbations touch the simulator")
class _Model:
    def __init__(self):
        self.light_pos = np.ones((2, 3)); self.light_diffuse = np.full((2, 3), 0.5)
        self.cam_pos = np.ones((3, 3))
class _Sim:
    def __init__(self): self.model = _Model(); self.forwarded = False
    def forward(self): self.forwarded = True
class _Env:
    def __init__(self): self.sim = _Sim()

for kind, attr in (("light", "light_pos"), ("cam", "cam_pos")):
    env = _Env(); before = getattr(env.sim.model, attr).copy()
    _, e = fw.make_perturb(f"{kind}:0.05", 3)
    e(env)
    check(f"{kind:10s} modifies {attr}", not np.allclose(getattr(env.sim.model, attr), before))
    check(f"{kind:10s} calls sim.forward()", env.sim.forwarded)
env = _Env(); before = env.sim.model.cam_pos.copy()
_, e = fw.make_perturb("light:0.05", 3); e(env)
check("light leaves the camera alone", np.allclose(env.sim.model.cam_pos, before))

print("\nunknown kinds are rejected, not silently ignored")
try:
    fw.make_perturb("wobble:1", 0); check("rejects unknown kind", False)
except SystemExit:
    check("rejects unknown kind", True)

print()
if fails: sys.exit(f"{fails} CHECK(S) FAILED")
print("ALL PERTURBATION CHECKS PASSED")

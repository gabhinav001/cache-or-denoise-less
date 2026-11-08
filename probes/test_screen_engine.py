"""Check the cache screen (probes/screen_engine.py) on a stub model, without a GPU.

The stub has the call structure of FastWAM's action inference: a per-step
denoiser that prepares tokens, runs one "stack" whose output depends on the
observation, and maps the result to a velocity; an Euler loop over a shifted
grid. The real StackHarness and CacheHarness from fw_libero.py wrap it, as on a
GPU host. The test checks that

  1. every replay equals an independent computation of the cached Euler loop
     (residuals of the configuration's last refresh call at steps 0..m-1);
  2. replays land on exactly the reuse calls of each refresh interval;
  3. with an unchanging observation and fixed noise every deviation is zero;
  4. a replay that substitutes nothing is refused rather than reported.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types

import numpy as np
import torch

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
sys.path.insert(0, _here)
_spec = importlib.util.spec_from_file_location("fw_libero", os.path.join(_here, "fw_libero.py"))
fw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fw)
import screen_engine as SE   # noqa: E402

fails = 0
def check(cond, msg):
    global fails
    if not cond:
        print(f"  FAIL: {msg}"); fails += 1

K, T, D, H_ = 10, 6, 4, 8
g = torch.Generator().manual_seed(0)
W_in = torch.randn(D, H_, generator=g) / 2
W_out = torch.randn(H_, D, generator=g) / 2
W_obs = torch.randn(3, H_, generator=g)


class Mot:
    """The stack: a residual that depends on the tokens and the observation."""
    def _apply_expert_post_block_tensor(self, *a, **k):   # wrapped by CacheHarness
        raise AssertionError("not used by the whole-stack path")

    def forward_action_with_video_cache_tensor(self, action_tokens, obs):
        return action_tokens + torch.tanh(action_tokens @ torch.eye(H_) * 0.7 + obs @ W_obs)


class Sched:
    def build_inference_schedule(self, num_inference_steps, device, dtype, shift_override=None):
        s = 1.0 if shift_override is None else float(shift_override)
        u = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        sig = s * u / (1 + (s - 1) * u)
        return (sig[:-1] * 1000).to(dtype), (sig[1:] - sig[:-1]).to(dtype)

    @staticmethod
    def step(model_output, delta, sample):
        return sample + model_output * delta


class Model:
    def __init__(self):
        self.mot = Mot()
        self.infer_action_scheduler = Sched()


model = Model()
SH = fw.StackHarness(model.mot)
SH.on = True
HB = fw.CacheHarness(model.mot)


def denoise(latents_action, timestep_action, obs):
    """The per-step denoiser: prepare, the stack, post (as `traced` wraps it)."""
    HB.begin()
    try:
        tok = latents_action @ W_in + timestep_action / 1000.0
        out = model.mot.forward_action_with_video_cache_tensor(action_tokens=tok, obs=obs)
        return out @ W_out
    finally:
        HB.end()


cap = SE.StepCapture(denoise)
SHIFT = 1.0


def infer(obs, x0):
    """Upstream's Euler loop, calling the per-step denoiser through the capture."""
    ts, ds = model.infer_action_scheduler.build_inference_schedule(K, "cpu", torch.float32, SHIFT)
    x = x0
    for t_, d_ in zip(ts, ds):
        x = Sched.step(cap(latents_action=x, timestep_action=t_.unsqueeze(0), obs=obs), d_, x)
    return {"action": x[0]}


def post(action):
    a = action.numpy().copy()
    a[..., -1] = np.sign(a[..., -1])
    return a


def reference(obs, x0, R_stale, m):
    """The cached loop computed directly: stale residuals at steps 0..m-1."""
    ts, ds = model.infer_action_scheduler.build_inference_schedule(K, "cpu", torch.float32, SHIFT)
    x = x0
    for k, (t_, d_) in enumerate(zip(ts, ds)):
        tok = x @ W_in + t_.unsqueeze(0) / 1000.0
        if k < m:
            out = tok + R_stale[k]
        else:
            out = model.mot.forward_action_with_video_cache_tensor(action_tokens=tok, obs=obs)
        x = x + (out @ W_out) * d_
    return x[0]


def residuals(obs, x0):
    """Whole-stack residuals of an uncached call, computed directly."""
    ts, ds = model.infer_action_scheduler.build_inference_schedule(K, "cpu", torch.float32, SHIFT)
    x, R = x0, {}
    for k, (t_, d_) in enumerate(zip(ts, ds)):
        tok = x @ W_in + t_.unsqueeze(0) / 1000.0
        out = model.mot.forward_action_with_video_cache_tensor(action_tokens=tok, obs=obs)
        R[k] = out - tok
        x = x + (out @ W_out) * d_
    return R


def run_episode(obs_seq, noise_seq, configs, replan=4):
    eng = SE.ScreenEngine(model, cap, SH, HB, post, replan, K, SHIFT, compiled=False,
                          configs=configs)
    eng.reset_episode()
    for c, (obs, x0) in enumerate(zip(obs_seq, noise_seq)):
        eng.begin_call(c)
        out = infer(obs, x0)
        eng.end_call(out["action"])
    return eng


# ---- 1 and 2: replays against the direct computation, on changing observations
n_cycles = 17
gg = torch.Generator().manual_seed(1)
obs_seq = [torch.randn(1, 1, 3, generator=gg) * 0.3 for _ in range(n_cycles)]
fixed = torch.randn(1, T, D, generator=gg)
fresh = [torch.randn(1, T, D, generator=gg) for _ in range(n_cycles)]
configs = ((3, 8), (8, 8), (9, 8), (8, 2), (8, 0))
for name, noise in (("fixed noise", [fixed] * n_cycles), ("fresh noise", fresh)):
    eng = run_episode(obs_seq, noise, configs)
    log = eng.episode_summary()
    worst = 0.0
    for m, tau in configs:
        lb = SE.label(m, tau)
        want_cycles = [c for c in range(n_cycles) if not SE.is_refresh(c, tau)]
        check(log["cycle"][lb] == want_cycles, f"{name} {lb}: replayed on {log['cycle'][lb]}")
        for c, d in zip(log["cycle"][lb], log["d"][lb]):
            r = max(cc for cc in range(c + 1) if SE.is_refresh(cc, tau))
            R = residuals(obs_seq[r], noise[r])
            a_c = reference(obs_seq[c], noise[c], R, m)[:4]
            a_f = reference(obs_seq[c], noise[c], {}, 0)[:4]
            d_ref = float(torch.sqrt(((a_c - a_f) ** 2).mean()))
            worst = max(worst, abs(d - d_ref))
    print(f"  {name}: largest |d - direct computation| = {worst:.2e}")
    check(worst < 1e-5, f"{name}: replay disagrees with the direct computation ({worst:.2e})")
    check(len(log["drift8"]) == sum(not SE.is_refresh(c, 8) for c in range(n_cycles)),
          f"{name}: drift8 on {len(log['drift8'])} calls")
    check(all(x > 0 for row in log["drift8"] for x in row), f"{name}: zero drift on a changing observation")
    check(log["hits_ok"], f"{name}: hit counts")
    if name == "fresh noise":
        dfix = np.mean(run_episode(obs_seq, [fixed] * n_cycles, configs).log["d"]["p8_tau8"])
        dfr = np.mean(log["d"]["p8_tau8"])
        print(f"  mean d at 8 cached steps, tau = 8: fixed noise {dfix:.4f}, fresh noise {dfr:.4f}")

# ---- 3: nothing changes between calls -> no deviation
eng = run_episode([obs_seq[0]] * 9, [fixed] * 9, configs)
log = eng.episode_summary()
mx = max(max(v) for v in log["d"].values() if v)
check(mx < 1e-6, f"unchanging observation and noise: largest d {mx}")
check(max(max(v) for v in log["e"].values() if v) < 1e-6, "unchanging inputs: latent deviation")
check(max(max(r) for r in log["drift8"]) < 1e-6, "unchanging inputs: residual drift")
print(f"  unchanging observation and fixed noise: largest d = {mx:.1e}")

# ---- 4: a replay whose stored residuals cannot be substituted is refused
eng = SE.ScreenEngine(model, cap, SH, HB, post, 4, K, SHIFT, compiled=False, configs=((8, 8),))
eng.reset_episode()
eng.begin_call(0); eng.end_call(infer(obs_seq[0], fixed)["action"])
eng.store[8] = {k: torch.zeros(1, 1, 1) for k in range(K)}   # wrong shape: the harness cannot substitute
eng.begin_call(1)
try:
    eng.end_call(infer(obs_seq[1], fixed)["action"])
    check(False, "a replay that substituted nothing was accepted")
except RuntimeError as e:
    print(f"  replay that substituted nothing refused: {e}")

check(SE.parse_screen("") == SE.DEFAULT_CONFIGS, "default configurations")
check(SE.parse_screen("8/8;3/2") == ((8, 8), (3, 2)), "explicit configurations")
for bad in ("8-8", "8/8;8/8"):
    try:
        SE.parse_screen(bad); check(False, f"accepted {bad!r}")
    except SystemExit:
        pass

if fails:
    print(f"screen engine: {fails} FAILED"); sys.exit(1)
print("screen engine: replays match the direct computation; schedules and refusals as specified")

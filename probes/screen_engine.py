"""Screen cache configurations on an uncached rollout (the `screen` arm of fw_libero.py).

A closed-loop cached arm answers one question per run: how often does this cache
configuration succeed. This module asks, at every replanning call of an UNCACHED
rollout, what each of several cache configurations would have returned for the
same observation, and how far that is from what the uncached policy returned.
The executed action is always the uncached one, so one rollout screens every
configuration at once, and the rollout itself is the stock policy's.

How a call is screened. FastWAM's `infer_action` encodes the observation, builds
the video key-value cache and then runs the Euler loop, calling the per-step
denoiser once per step. `StepCapture` wraps that per-step denoiser (the compiled
one, exactly as upstream builds it) and keeps references to each step's inputs:
the initial latent, the timestep, the text context and the video cache. Upstream
code runs unmodified and produces the executed action. Afterwards each shadow
configuration replays the same Euler loop from the same initial latent, with the
whole-stack residual at steps 0..m-1 replaced by the one recorded on that
configuration's last refresh call, through the same StackHarness substitution the
closed-loop `whole:TAU` arms use. A shadow therefore returns exactly what a
closed-loop cache would return at this observation, given the residuals of the
uncached rollout's refresh calls. It does not reproduce the trajectory a
closed-loop cache would follow: from its first reuse call on, a cache executes
its own actions and reaches other states, while the replays stay on the
uncached rollout. The screen measures local error along uncached states.

Per shadow configuration (m cached steps, refresh interval tau) and reuse call:
  d     RMS difference between the cached and the uncached action chunk over the
        actions the robot executes (the first `replan`), in the model's
        normalised action space
  grip  executed actions whose binarised gripper command differs
  e     relative difference of the latent where fresh computation resumes
        (the input to step m)
On reuse calls of the tau = 8 store, `drift8` adds, per denoising step k, how far
the stored residual is from this call's: |R_k(t) - R_k(t_r)| / |R_k(t)|.

Nothing here draws random numbers: replays reuse the call's own initial latent,
so the global streams the rollout reads are untouched.
"""
from __future__ import annotations

import time

import numpy as np
import torch

# The configurations the screen replays by default: every cached-step count of
# the share sweep at the default refresh interval, and the refresh intervals of
# Table 1 at the default eight cached steps (tau = 0: refreshed only when an
# episode starts).
DEFAULT_CONFIGS = ((3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
                   (8, 2), (8, 4), (8, 0))


def parse_screen(spec):
    """'' -> DEFAULT_CONFIGS; 'M/TAU;M/TAU;...' -> tuple of (m, tau)."""
    spec = str(spec or "").strip()
    if not spec:
        return DEFAULT_CONFIGS
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        m, _, tau = part.partition("/")
        if not (m.isdigit() and tau.isdigit()):
            raise SystemExit(f"screen config {part!r}: expected M/TAU")
        out.append((int(m), int(tau)))
    if not out or len(set(out)) != len(out):
        raise SystemExit(f"screen spec {spec!r}: empty or repeated configurations")
    return tuple(out)


def label(m, tau):
    return f"p{m}_tau{tau}"


def is_refresh(cycle, tau):
    """Whether a cache refreshed every tau calls (tau = 0: only at the start) recomputes on this call."""
    return cycle == 0 or (tau > 0 and cycle % tau == 0)


class StepCapture:
    """Wraps the per-step denoiser and keeps references to each step's inputs.

    Pure Python around the compiled callable: it changes no argument and no
    result, so the call it wraps computes exactly what it computed before.
    """

    def __init__(self, fn):
        self.fn = fn
        self.active = False
        self.calls = []

    def __call__(self, *a, **kw):
        if a:
            raise TypeError("the denoiser is called with keyword arguments only")
        if self.active:
            self.calls.append(kw)
        return self.fn(**kw)


def _digest(x):
    from provenance import tensor_digest
    return tensor_digest(x.detach().float().cpu())


def _rel(a, b):
    """|a - b| / |b| in float32, as a Python float."""
    a32, b32 = a.float(), b.float()
    return float((a32 - b32).norm() / b32.norm().clamp_min(1e-30))


class ScreenEngine:
    """Replays cache configurations after every uncached call of an episode.

    model        the FastWAM model (its scheduler builds the Euler deltas)
    capture      the StepCapture installed around the per-step denoiser
    stack        fw_libero.StackHarness (records and substitutes whole-stack residuals)
    block        fw_libero.CacheHarness (reset so its step counter stays in range)
    post         normalised action [T, D] tensor -> executed actions (numpy), as
                 eval_libero_single._predict_action_chunk post-processes them
    dump         also keep every replay's executed actions (smoke tests only: it
                 is what lets a replay be compared with a closed-loop cached arm)
    """

    def __init__(self, model, capture, stack, block, post, replan, n_steps, sigma_shift,
                 compiled, configs=DEFAULT_CONFIGS, timing=False, dump=False):
        self.model, self.capture, self.stack, self.block = model, capture, stack, block
        self.post, self.replan, self.n_steps = post, int(replan), int(n_steps)
        self.sigma_shift, self.compiled = sigma_shift, bool(compiled)
        self.configs = tuple(configs)
        for m, tau in self.configs:
            if not 1 <= m < self.n_steps:
                raise SystemExit(f"screen: {m} cached steps leaves no fresh step of {self.n_steps}")
            if tau < 0:
                raise SystemExit("screen: tau must be >= 0")
        self.taus = sorted({tau for _, tau in self.configs})
        self.timing, self.dump = bool(timing), bool(dump)
        self._deltas = None
        self.reset_episode()

    # ------------------------------------------------------------------ episode
    def reset_episode(self):
        self.cycle = 0
        self.store = {}           # tau -> {step: residual} from that tau's last refresh
        self.fresh_res = {}       # this call's residuals, filled by the stack harness
        keys = ("d", "grip", "e", "cycle") + (("exec",) if self.dump else ())
        self.log = {k: {} for k in keys}
        self.log.update(drift8=[], screen_ms=[], latent=[])
        for m, tau in self.configs:
            for k in keys:
                self.log[k][label(m, tau)] = []
        self.hits_ok = True

    def begin_call(self, cycle):
        """Called by the rollout loop before every uncached call."""
        self.cycle = int(cycle)
        self.fresh_res = {}
        self.capture.calls = []
        self.capture.active = True
        # record the whole-stack residual of every step of this call
        self.stack.reset(record=self.fresh_res, steps=None)
        self.block.reset()
        return self.fresh_res

    # ------------------------------------------------------------------ replay
    def _deltas_for(self, dtype, device):
        if self._deltas is None:
            _, deltas = self.model.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=self.n_steps, device=device, dtype=dtype,
                shift_override=self.sigma_shift)
            self._deltas = deltas
        return self._deltas

    def _replay(self, calls, apply, m):
        """The Euler loop of this call with steps 0..m-1 served from `apply`."""
        x = calls[0]["latents_action"]
        deltas = self._deltas_for(x.dtype, x.device)
        self.stack.reset(apply=apply, steps=set(range(m)), beta=1.0)
        self.stack.zero_counts()
        self.block.reset()
        e = None
        for k, kw in enumerate(calls):
            if k == m:
                e = _rel(x, kw["latents_action"])
            if self.compiled:
                torch.compiler.cudagraph_mark_step_begin()
            pred = self.capture.fn(**{**kw, "latents_action": x})
            x = self.model.infer_action_scheduler.step(pred, deltas[k], x)
        c = self.stack.counts()
        if c["hits"] != m or c["misses"]:
            self.hits_ok = False
            raise RuntimeError(f"screen replay p{m}: {c} (expected {m} hits)")
        return x, e

    def end_call(self, action_fresh):
        """Called with the uncached call's normalised action [T, D] (CPU float32)."""
        self.capture.active = False
        calls = self.capture.calls
        self.capture.calls = []
        if len(calls) != self.n_steps:
            raise RuntimeError(f"screen: captured {len(calls)} denoiser calls, expected {self.n_steps}")
        if set(self.fresh_res) != set(range(self.n_steps)):
            raise RuntimeError(f"screen: recorded residuals for steps {sorted(self.fresh_res)}")
        c, R = self.cycle, self.fresh_res
        # which initial latent this call started from (compared with the run's
        # noise policy by the analysis: one latent per run, per episode or per call)
        self.log["latent"].append(_digest(calls[0]["latents_action"]))
        if self.timing:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        a_f = action_fresh[: self.replan].float()
        exe_f = None
        # residual drift against the tau = 8 store, before this call can replace it
        if 8 in self.store and not is_refresh(c, 8):
            S8 = self.store[8]
            self.log["drift8"].append([round(_rel(S8[k], R[k]), 6) for k in range(self.n_steps)])
        for tau in self.taus:
            if is_refresh(c, tau):
                continue
            for m, t_ in self.configs:
                if t_ != tau:
                    continue
                x, e = self._replay(calls, self.store[tau], m)
                a_c = x[0].detach().to(device="cpu", dtype=torch.float32)[: self.replan]
                d = float(torch.sqrt(((a_c - a_f) ** 2).mean()))
                if exe_f is None:
                    exe_f = self.post(action_fresh)[: self.replan]
                exe_c = self.post(x[0].detach().to(device="cpu", dtype=torch.float32))[: self.replan]
                flips = int((np.sign(exe_c[:, -1]) != np.sign(exe_f[:, -1])).sum())
                lb = label(m, tau)
                self.log["d"][lb].append(round(d, 6))
                self.log["grip"][lb].append(flips)
                self.log["e"][lb].append(round(e, 6))
                self.log["cycle"][lb].append(c)
                if self.dump:
                    self.log["exec"][lb].append([list(map(float, a)) for a in exe_c])
        # refresh calls store this call's residuals for later reuse calls
        for tau in self.taus:
            if is_refresh(c, tau):
                self.store[tau] = R
        self.stack.reset()
        self.block.reset()
        if self.timing:
            torch.cuda.synchronize()
            self.log["screen_ms"].append(round((time.perf_counter() - t0) * 1000.0, 3))

    def episode_summary(self):
        out = {"configs": [label(m, t) for m, t in self.configs], "hits_ok": self.hits_ok}
        out.update(self.log)
        return out

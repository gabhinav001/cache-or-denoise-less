"""Closed-loop LIBERO success rates for cache placements.

Everything measured so far is open-loop action deviation -- a proxy whose
relationship to task outcome is unestablished. This closes that gap.

The first cycle fills the cache; a positive refresh interval recomputes it
on cycles 0, tau, 2*tau, ... (tau=0 refreshes only once). The ten-step default
caches all 30 blocks at steps 0-7, supporting both whole-stack residual bypass
(the reference schedule) and per-block residual injection.

The observation pipeline, denormalisation and gripper handling are reused from
the released evaluation script, so only the cache differs from a stock run.
"""
from __future__ import annotations

import hashlib, json, os, random, sys, time
from provenance import provenance, file_digest, tensor_digest
from stack_probe_common import file_sha256, inference_runtime, load_stack_weights, validate_checkpoint_state
import numpy as np

# must precede any mujoco/torch import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
_ld = torch.load                       # LIBERO init states are pickled numpy
torch.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

N_BLOCKS, N_STEPS, SKIP = 30, 10, 240

# The eval path compiles with mode="reduce-overhead", i.e. CUDA graph capture.
# Graph-owned memory is reclaimed on the next run, so a residual recorded on one
# cycle is overwritten before the next cycle reads it -- cloning does not help,
# because the clone lands in the same pool. Dropping to plain inductor keeps most
# of the compile speedup (which the caching claim is measured against) while
# letting cached tensors outlive the call that produced them.
#
# WAM_CUDAGRAPHS=1 leaves upstream's reduce-overhead compilation in place. It
# exists to measure what an uncached deployment would actually pay, so main()
# refuses every arm except `stock` when it is set: a cache under CUDA graphs
# reads overwritten memory and returns plausible wrong actions.
CUDAGRAPHS = os.environ.get("WAM_CUDAGRAPHS", "0") == "1"
_torch_compile = torch.compile
def _compile_without_cudagraphs(*a, **k):
    if k.get("mode") == "reduce-overhead":
        k = {**k, "mode": None}
    return _torch_compile(*a, **k)
if not CUDAGRAPHS:
    torch.compile = _compile_without_cudagraphs

# Each cache configuration is a distinct set of dynamo guards (which cells are
# substituted, whether a weight is applied), so switching between the arms
# recompiles. That is a fixed one-off cost per arm, not per cycle, but it
# exceeds the default limit of eight.
for _lim in ("recompile_limit", "accumulated_recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _lim):
        setattr(torch._dynamo.config, _lim, 256)


class CacheHarness:
    def __init__(self, mot):
        self.mot = mot; self.orig = mot._apply_expert_post_block_tensor
        self.on = False; self.step = -1; self.layer = 0
        self.record = None; self.apply = None; self.cells = None; self.scale = None
        self.staleness = 1; self.oracle_beta = False; self.skip = False
        mot._apply_expert_post_block_tensor = self._wrap(self.orig)

    def _wrap(self, fn):
        def w(*a, **k):
            if not self.on: return fn(*a, **k)
            key = (self.step, self.layer); self.layer += 1
            rx = k.get("residual_x")
            if rx is None: return fn(*a, **k)
            # Explicitly skip the post-block function on a cache hit. QKV and
            # attention have already been computed by the caller; eliminating
            # those unused operations depends on compilation. This branch and
            # the substitution branch agree for eager stub arithmetic, but their
            # compiled rollout equivalence has not been established.
            if self.skip and self.record is None and not self.oracle_beta \
                    and self.cells is not None and key in self.cells:
                c = self.apply.get(key) if self.apply else None
                if c is not None and c.shape == rx.shape:
                    return rx + (c if self._w(key) == 1.0 else self._w(key) * c)
            # MEASURED: the compiler deletes this call when its result is unused.
            # On an A6000, c3ache_tau4 -- which takes this path and discards `out`
            # for every cached cell -- runs at 371 ms/cycle against full compute's
            # 503, non-overlapping intervals. If every block really ran it would be
            # 503. So "identical compute, differing in one scalar multiply" was
            # never true of this harness.
            #
            # It is deleted in exactly the places a deployment would skip, which is
            # why the effect is partial rather than total:
            #   refresh cycles   `record` is set, `out` is used below -> kept
            #   uncached cells   `out` is returned at the end          -> kept
            #   cached cells on a reuse cycle, `out` discarded         -> DELETED
            # Explicit skipping still buys more (reuse cost 0.579 of a full cycle
            # against 0.652), so the elimination is not complete.
            #
            # The plain-versus-attenuated comparison is unaffected: both arms come
            # through here and lose the same work. What the deletion invalidates is
            # the DESCRIPTION of those runs as full-compute-schedule measurements.
            out = fn(*a, **k)
            if self.record is not None: self.record[key] = (out - rx).detach().clone()
            if self.cells is not None and key in self.cells:
                c = self.apply.get(key) if self.apply else None
                if c is not None and c.shape == rx.shape:
                    # the published method substitutes the cached residual at unit
                    # weight; `scale` lets it be attenuated per cell instead, at the
                    # same compute -- the multiply rides on a copy that already happens
                    if self.oracle_beta:
                        # CEILING ARM, NOT DEPLOYABLE. The block has already been
                        # computed, so the true residual is in hand; pick the
                        # weight that is optimal for THIS cycle. It saves nothing
                        # -- every block still runs -- and exists only to bound
                        # how much task success any per-cell scalar weight could
                        # recover, separating "the weight is badly estimated"
                        # from "a scalar weight cannot do better".
                        # No Python branch and no .item() here: this runs inside
                        # a compiled region, where branching on a tensor value is
                        # "Data-dependent branching" that dynamo cannot trace, and
                        # .item() forces a host sync and a graph break. clamp_min
                        # handles a zero-norm cached residual, which then yields a
                        # weight of zero -- correct, since such a residual carries
                        # no information to scale.
                        r = (out - rx).float(); cf = c.float()
                        den = (cf * cf).sum().clamp_min(1e-30)
                        b = ((r * cf).sum() / den).clamp(0.0, 1.5)
                        return rx + b.to(c.dtype) * c
                    s = self._w(key)
                    return rx + (c if s == 1.0 else s * c)
            return out
        return w

    def _w(self, key):
        """The weight for one cell, shared by the skip and substitute paths.

        A plain float is one weight for every staleness (what the original runs
        used); a dict is indexed by how many cycles old the cached residual is.
        """
        if not self.scale: return 1.0
        sc = self.scale.get(key, 1.0)
        if isinstance(sc, dict):
            return sc.get(self.staleness, sc.get(max(sc), 1.0))
        return sc

    def begin(self): self.step += 1; self.layer = 0; self.on = True
    def end(self): self.on = False
    def reset(self, record=None, apply=None, cells=None, scale=None, staleness=1,
              oracle_beta=False, skip=False):
        self.step = -1; self.layer = 0
        self.record, self.apply, self.cells, self.scale = record, apply, cells, scale
        self.staleness = int(staleness); self.oracle_beta = bool(oracle_beta)
        self.skip = bool(skip)
    def close(self): self.mot._apply_expert_post_block_tensor = self.orig



def rng_fingerprint():
    """A short hash of every global random stream a rollout could draw from.

    Cheap enough to take once per episode. Called before any per-episode reseed
    has been skipped or applied, so two runs of the same configuration should
    produce identical fingerprints for the same (task, episode) whenever the
    seeding is position-independent.
    """
    out = {}
    def h(b): return hashlib.sha256(bytes(b)).hexdigest()[:12]
    out["torch_cpu"] = h(torch.random.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        out["torch_cuda"] = h(torch.cuda.get_rng_state().numpy().tobytes())
    st = np.random.get_state()
    out["numpy"] = h(np.asarray(st[1]).tobytes())
    out["python"] = h(repr(random.getstate()).encode())
    return out


# ---------------------------------------------------------------- perturbation
# Everything measured so far runs the policy on exactly the distribution its
# weights were fit on. The attenuation coefficients are fit offline on clean
# replay episodes, so the obvious question is whether they survive a shift in the
# observations -- and whether the cached-residual geometry they encode still
# holds. These are the cheapest honest perturbations: two act on the rendered
# image, two act on the simulator itself and so change shading and parallax
# rather than just pixel statistics.
#
# The perturbation is seeded per (task, episode) and applied identically in every
# arm, so a pair stays a pair: both arms see the SAME perturbed observation and
# the comparison remains matched.
def make_perturb(spec, seed):
    """spec = "none" | "<kind>:<magnitude>". Returns (obs_fn, env_fn)."""
    if not spec or spec == "none":
        return None, None
    kind, _, mag = str(spec).partition(":")
    mag = float(mag) if mag else 0.0
    rng = np.random.RandomState(seed)

    if kind in ("bright", "contrast", "noise", "blur"):
        def obs_fn(obs):
            out = dict(obs)
            for k, v in obs.items():
                if not (k.endswith("_image") and hasattr(v, "shape") and v.ndim == 3):
                    continue
                x = v.astype(np.float32)
                if kind == "bright":
                    x = x + mag * 255.0
                elif kind == "contrast":
                    m = x.mean(); x = (x - m) * (1.0 + mag) + m
                elif kind == "noise":
                    x = x + rng.normal(0.0, mag * 255.0, x.shape)
                elif kind == "blur":
                    k2 = max(3, int(2 * round(mag) + 1))
                    try:
                        import cv2
                        x = cv2.GaussianBlur(x, (k2, k2), 0)
                    except ImportError:
                        # separable box blur, repeated: close enough to a gaussian
                        # for a robustness probe and keeps this runnable without cv2
                        pad = k2 // 2
                        for ax in (0, 1):
                            xp = np.pad(x, [(pad, pad) if a == ax else (0, 0)
                                            for a in range(x.ndim)], mode="edge")
                            acc = np.zeros_like(x)
                            for o in range(k2):
                                sl = [slice(None)] * x.ndim
                                sl[ax] = slice(o, o + x.shape[ax])
                                acc += xp[tuple(sl)]
                            x = acc / k2
                out[k] = np.clip(x, 0, 255).astype(v.dtype)
            return out
        return obs_fn, None

    if kind in ("light", "cam"):
        def env_fn(env):
            sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
            if sim is None:
                raise RuntimeError("cannot reach env.sim to apply a simulator perturbation")
            m = sim.model
            if kind == "light":
                # move the lights and vary their intensity: this changes shading
                # and shadow direction, not just overall brightness
                if m.light_pos.size:
                    m.light_pos[:] = m.light_pos + rng.normal(0, mag, m.light_pos.shape)
                if m.light_diffuse.size:
                    m.light_diffuse[:] = np.clip(
                        m.light_diffuse * (1.0 + rng.normal(0, mag, m.light_diffuse.shape)), 0, 1)
            else:
                # jitter camera position: parallax shift, as from an imperfectly
                # remounted camera
                if m.cam_pos.size:
                    m.cam_pos[:] = m.cam_pos + rng.normal(0, mag, m.cam_pos.shape)
            sim.forward()
        return None, env_fn

    raise SystemExit(f"unknown perturbation kind: {kind}")



class StackHarness:
    """One residual for the WHOLE 30-block stack per denoise step.

    This is what C$^3$ache describes: record $h_L-h_0$ once and add it once.
    CacheHarness records 30 separate residuals and adds them 30 times. The two
    telescope in exact arithmetic -- `residual_x` really is the block input, so
    $\sum_b (x_{b+1}-x_b) = x_{30}-x_0$ -- but the model runs in BF16, where the
    per-block form rounds thirty times and the whole-stack form rounds once.
    That difference is unmeasured, and it is the reason this class exists: the
    telescoping argument alone cannot establish that our plain cache is their
    plain cache, so a gap in task success cannot be attributed to the method.

    Substitution here inherently skips all thirty blocks -- there is no cached
    value left to compute -- so there is no compute-and-discard variant to keep
    in step with, and `hits`/`misses` are counted rather than assumed.
    """

    def __init__(self, mot):
        self.mot = mot
        self.orig = mot.forward_action_with_video_cache_tensor
        self.on = False; self.step = 0
        self.record = None; self.apply = None; self.steps = None
        self.beta = 1.0; self.fp32 = False
        # tap stores the value this step RETURNED, substituted or computed, so
        # every arm can be compared at the same point in the stack rather than
        # only at the action the denoiser finally emits.
        self.tap = None
        self.hits = 0; self.misses = 0
        mot.forward_action_with_video_cache_tensor = self._wrap(self.orig)

    def _wrap(self, fn):
        def w(*a, **k):
            if not self.on: return fn(*a, **k)
            step = self.step; self.step += 1
            xin = k.get("action_tokens")
            if xin is None: return fn(*a, **k)
            if self.steps is not None and step in self.steps:
                c = self.apply.get(step) if self.apply else None
                if c is not None and c.shape == xin.shape:
                    self.hits += 1
                    if self.fp32:
                        # Accumulate in fp32 and round once on the way out, so a
                        # BF16 accumulation effect can be separated from the
                        # substitution itself rather than confounded with it.
                        sub = (xin.float() + self.beta * c.float()).to(xin.dtype)
                    else:
                        sub = xin + (c if self.beta == 1.0 else self.beta * c)
                    if self.tap is not None: self.tap[step] = sub.detach().clone()
                    return sub
                # A shape mismatch or an absent entry silently fell back to
                # computing the block in the per-cell harness, which is how a
                # cache that was never populated could still report "240 cells".
                self.misses += 1
            out = fn(*a, **k)
            if self.record is not None:
                self.record[step] = (out - xin).detach().clone()
            if self.tap is not None: self.tap[step] = out.detach().clone()
            return out
        return w

    def reset(self, record=None, apply=None, steps=None, beta=1.0, fp32=False,
              tap=None):
        self.step = 0
        self.record, self.apply, self.steps = record, apply, steps
        self.beta = float(beta); self.fp32 = bool(fp32); self.tap = tap

    def counts(self):
        return {"hits": self.hits, "misses": self.misses}

    def zero_counts(self):
        self.hits = self.misses = 0

    def close(self):
        self.mot.forward_action_with_video_cache_tensor = self.orig



def parse_configs(configs, SC=None, WB=None, shrink="shrink.json",
                  stack_shrink="stack_shrink.json"):
    """Turn a config string into (label, placement, tau, weight, oracle, skip).

    Extracted from main() so it can be exercised without a GPU. The `whole`
    branch was added late and a mislabelled arm here would not raise -- it would
    quietly file whole-stack results under a per-block name, which is precisely
    the confusion this whole diagnostic exists to remove.
    """
    specs = []
    for c in str(configs).split(","):
        c = c.strip()
        if not c: continue
        if c == "full": specs.append(("full", None, None, None, False, False))
        # STOCK: full compute with residual recording switched OFF. `full` records
        # a residual for every block on every cycle -- 300 subtract-and-clone per
        # cycle -- which a deployment that is not building a cache never does. Every
        # speedup in this repository was quoted against `full`, so the baseline
        # carried instrumentation the arms it was compared against carried less of
        # (a cached arm records only on refresh cycles). This arm is the honest
        # denominator. It is quality-identical to `full` by construction: recording
        # only reads the block's output, so removing it cannot change any action.
        elif c == "stock": specs.append(("stock", None, None, None, False, "norec"))
        # SCREEN: the stock policy, uncached, with every call also replayed under
        # the cache configurations of probes/screen_engine.py. Nothing it
        # replays is executed, so its rollouts are the uncached policy's.
        elif c == "screen": specs.append(("screen", "screen", None, None, False, False))
        else:
            parts = c.split(":")
            if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
                raise SystemExit(f"config {c}: expected NAME:TAU[:FLAGS]")
            name, tau = parts[0], int(parts[1])
            suffix = parts[2] if len(parts) > 2 else ""
            bad = set(suffix) - set("soxa")
            if bad: raise SystemExit(f"config {c}: unknown suffix flag(s) {sorted(bad)}")
            orc, skip, agg = "o" in suffix, "x" in suffix, "a" in suffix
            sc = SC if ("s" in suffix or "a" in suffix) else None
            if ("s" in suffix or "a" in suffix) and SC is None and name != "whole":
                raise SystemExit(f"config {c} needs {shrink}")
            if orc and skip:
                raise SystemExit(f"config {c}: the oracle weight needs the true "
                                 "residual, so it cannot skip the block")
            # WHOLE-STACK: one residual for the 30-block stack, added once --
            # what C3ache describes. Substituting it inherently skips every
            # block, so `x` is implied and `o` is meaningless here.
            if name == "whole":
                if orc:
                    raise SystemExit(f"config {c}: no per-cell oracle weight "
                                     "exists for a whole-stack residual")
                if agg:
                    if SC is None:
                        raise SystemExit(f"config {c} needs {shrink}")
                    label = f"whole_agg_tau{tau}"
                    specs.append((label, "whole_agg", tau, SC, False, False))
                    continue
                wb = WB if "s" in suffix else None
                if "s" in suffix and WB is None:
                    raise SystemExit(f"config {c} needs {stack_shrink}")
                label = f"whole_tau{tau}" + ("_shrink" if wb is not None else "")
                specs.append((label, name, tau, wb, False, False))
                continue
            label = (f"{name}_tau{tau}" + ("_shrink" if sc else "")
                     + ("_oracle" if orc else "") + ("_skip" if skip else ""))
            specs.append((label, name, tau, sc, orc, skip))
    return specs


def parse_delay(spec, replan, horizon):
    """'0' | 'D' | 'R/U' -> (delay after a refresh call, delay after a reuse call), in control steps."""
    parts = str(spec).strip().split("/")
    if len(parts) not in (1, 2) or not all(x.strip().isdigit() for x in parts):
        raise SystemExit(f"delay {spec!r}: expected D or R/U (control steps)")
    d = tuple(int(x) for x in parts) * (2 if len(parts) == 1 else 1)
    for x in d:
        if not 0 <= x < replan or replan + x > horizon:
            raise SystemExit(f"delay {x}: need 0 <= delay < {replan} and {replan} + delay <= {horizon}")
    return d


def delayed_actions(chunk, prev, delay, replan, dummy):
    """The actions executed after a call whose result arrives `delay` control steps late.

    Naive asynchronous execution: the chunk is computed from the observation at
    time t, but the robot keeps executing the previous chunk until it arrives,
    so steps t..t+delay-1 take the previous chunk's actions at indices
    replan..replan+delay-1 (or `dummy`, holding still, before the first chunk)
    and steps t+delay..t+replan-1 take this chunk's actions at delay..replan-1.
    delay = 0 is synchronous execution.
    """
    if delay == 0:
        return [list(a) for a in chunk[:replan]]
    head = ([list(a) for a in prev[replan:replan + delay]] if prev is not None
            else [list(dummy) for _ in range(delay)])
    return head + [list(a) for a in chunk[delay:replan]]


def cached_path_fraction(n_steps, prefix, shift):
    """Share of the flow path (in sigma) integrated with cached residuals.

    Steps 0..prefix-1 are cached. FastWAM's continuous scheduler places step k
    at sigma_k = s*u/(1+(s-1)*u) with u = 1 - k/K, and step k integrates from
    sigma_k to sigma_{k+1}, so the cached steps cover sigma in [sigma_prefix, 1].
    The same step index therefore caches very different amounts of the path
    under different shifts: 8 of 10 steps is 80% at s=1 and 44% at s=5.
    """
    if n_steps < 1 or not 0 <= prefix <= n_steps:
        raise ValueError("need 0 <= prefix <= n_steps")
    u = 1.0 - prefix / n_steps
    s = float(shift)
    return 1.0 - s * u / (1.0 + (s - 1.0) * u)


def placements(oracle_path, n_steps=N_STEPS, prefix=None):
    """Which (denoise step, block) cells a placement caches.

    THE PUBLISHED RULE IS "EVERY BLOCK, ALL BUT THE LAST TWO DENOISE STEPS":
    steps [0,7] of 10. The last two steps are left exact because that is where
    the residual stops being reusable.

    At fewer than ten steps that rule has to be restated, and the restatement is
    a design decision rather than a config change -- which is why this used to be
    hardcoded to range(8) with an assert that every placement had exactly 240
    cells. Generalised by PRESERVING THE STRUCTURE, not the count: cache every
    block for all but the last two steps, whatever the step count. At 10 steps
    that is the published [0,7]; at 6 it is [0,3]; at 4 it is [0,1].

    The consequence is that a shorter schedule caches a smaller FRACTION of its
    work (80% of steps at 10, 67% at 6, 50% at 4), so a 4-step cache is less
    aggressive than a 10-step one by construction. That is the honest
    generalisation -- the alternative, holding the fraction fixed, would cache
    into the final refinement steps that the published method deliberately
    leaves alone.

    `prefix` overrides that rule with an explicit number of leading steps to
    cache (PROBE.cache_prefix). It exists to separate the effect of the shift
    from the effect of how much of the flow path the cache covers.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    rng = random.Random(0)
    if prefix is None or int(prefix) < 0:
        keep = max(0, n_steps - 2)              # all but the last two steps
    else:
        keep = int(prefix)
        if keep > n_steps:
            raise ValueError(f"cache_prefix {keep} exceeds {n_steps} denoising steps")
    skip = keep * N_BLOCKS
    allc = [(k, b) for k in range(n_steps) for b in range(N_BLOCKS)]
    P = {"c3ache": {(k, b) for k in range(keep) for b in range(N_BLOCKS)},
         "random": set(rng.sample(allc, skip))}
    if oracle_path and os.path.exists(oracle_path):
        cells = {tuple(x) for x in json.load(open(oracle_path))["oracle"]}
        if len(cells) == skip: P["oracle"] = cells
    for n, s in P.items(): assert len(s) == skip, (n, len(s))
    return P


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pr = OmegaConf.merge(OmegaConf.create(
        {"oracle": "/workspace/oracle_cells.json", "shrink": "/workspace/shrink.json",
         "stack_shrink": "/workspace/stack_shrink.json",
         "out": "/workspace/fw_libero.json",
         "suite": "libero_goal", "n_tasks": 10, "n_episodes": 8, "first_task": 0,
         "perturb": "none", "no_reseed": 0, "rng_probe": 0, "timing": 0,
         "warmup_episodes": 1, "cache_prefix": -1, "resample_noise": 0,
         "first_episode": 0, "screen": "", "dump_chunks": 0,
         "episode_noise": 0, "delay": "0",
         "configs": "full,c3ache:0,oracle:0,random:0,c3ache:2,random:2"}),
        cfg.get("PROBE", {}))

    # FastWAM builds its initial action latent from a generator seeded with
    # cfg.seed on EVERY call, so every replanning cycle starts from the same
    # noise. That is the upstream default, and it favours cross-cycle reuse: at
    # the high-noise steps the input is identical from one cycle to the next.
    # resample_noise=1 passes seed=None to the model instead, so each call draws
    # fresh noise from the global generator. The per-episode reseed below still
    # fixes that stream per (task, episode), which keeps arms paired: the i-th
    # call of an episode gets the same draw in every arm.
    run_seed = int(cfg.get("seed", 42))
    resample = bool(int(pr.resample_noise))
    # episode_noise=1 keeps each episode's latent fixed for the whole episode:
    # the model is given the per-episode seed below, so every call of an
    # episode starts from one latent, the same one the first call of the
    # resample_noise arm draws from the reseeded global stream (a CPU generator
    # seeded with s draws exactly what torch.manual_seed(s) does).
    episode_noise = bool(int(pr.episode_noise))
    if episode_noise and (resample or int(pr.no_reseed)):
        raise SystemExit("episode_noise needs the per-episode reseed and excludes resample_noise")
    if resample:
        if int(pr.no_reseed):
            raise SystemExit("resample_noise needs the per-episode reseed; "
                             "without it the noise stream depends on arm history")
        with open_dict(cfg):
            cfg.seed = None

    sys.path.insert(0, "/w/experiments/libero")
    import eval_libero_single as E
    from libero_utils import get_libero_env, get_libero_dummy_action
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    dev = torch.device(str(cfg.EVALUATION.device))
    dt = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dt, device=str(dev))
    payload = model.load_checkpoint(str(cfg.ckpt))
    checkpoint_coverage = validate_checkpoint_state(model, payload)
    del payload
    model = model.to(dev).eval()
    runtime = inference_runtime(cfg, model)
    runtime["checkpoint_state_coverage"] = checkpoint_coverage
    runtime["checkpoint_sha256"] = file_sha256(str(cfg.ckpt))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path)))

    ah = int(cfg.data.train.num_frames) - 1
    h, w = (int(v) for v in cfg.data.train.video_size)
    replan = int(cfg.EVALUATION.get("replan_steps", 10))
    DELAY = parse_delay(str(pr.delay), replan, ah)
    wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    max_steps = E._get_max_steps(str(pr.suite))
    n_steps = int(cfg.EVALUATION.get("num_inference_steps", N_STEPS))
    PL = placements(str(pr.oracle), n_steps=n_steps, prefix=int(pr.cache_prefix))
    cache_prefix = len({k for k, _ in PL["c3ache"]})
    path_fraction = cached_path_fraction(
        n_steps, cache_prefix, runtime["effective_action_sigma_shift"])

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[str(pr.suite)]()
    if not 0 <= int(pr.first_task) < int(pr.n_tasks) <= suite.n_tasks:
        raise ValueError("Require 0 <= first_task < n_tasks <= suite.n_tasks")
    if CUDAGRAPHS:
        # Upstream's own compiled path, untouched: no hooks at all, so nothing
        # here can add latency or interact with graph capture.
        bad = [c for c in str(pr.configs).split(",") if c.strip() not in ("", "stock")]
        if bad:
            raise SystemExit(f"WAM_CUDAGRAPHS=1 supports only the stock arm, got {bad}")
        class _NoCache:
            def begin(self): pass
            def end(self): pass
            def reset(self, *a, **k): pass
            def close(self): pass
        H = _NoCache()
    else:
        H = CacheHarness(model.mot)
    dn = model._denoise_action_with_video_cache
    # NOTHING UNTRACEABLE MAY LIVE IN HERE. This wrapper sits at the boundary of
    # a compiled region, so dynamo traces it, and the first version of the timing
    # code put torch.cuda.Event().record() inside it. Dynamo cannot trace a CUDA
    # event, and the graph-break message it tried to raise hit a formatting bug
    # in torch itself, so the run died with
    #     InternalTorchDynamoError: 'tuple' object has no attribute 'splitlines'
    # which names neither the cause nor the line. Timing now happens in the
    # rollout loop below, which is plain Python that nothing compiles.
    TIMER = {"ms": 0.0, "n": 0}
    def traced(*a, **k):
        H.begin()
        try: return dn(*a, **k)
        finally: H.end()
    if not CUDAGRAPHS:
        model._denoise_action_with_video_cache = traced

    # per-cell attenuation coefficients, fit offline on held-out replay episodes
    SC = None
    if os.path.exists(str(pr.shrink)):
        d = json.load(open(str(pr.shrink)))
        def _key(k): return (int(k.split(",")[0]), int(k.split(",")[1]))
        if d.get("staleness_aware") and "coef_k" in d:
            # one weight per (cell, staleness): the cache is reused at staleness
            # 1..tau-1, and a weight fit at staleness 1 under-attenuates the rest
            SC = {}
            for ks, per in d["coef_k"].items():
                for k, v in per.items():
                    SC.setdefault(_key(k), {})[int(ks)] = float(v[0])
            med = {ks: sorted(SC[c][int(ks)] for c in SC)[len(SC)//2]
                   for ks in d["coef_k"]}
            print(f"[libero] loaded staleness-aware coefficients for {len(SC)} cells; "
                  f"medians by staleness " +
                  ", ".join(f"{k}:{v:.3f}" for k, v in sorted(med.items())), flush=True)
        else:
            SC = {_key(k): float(v[0]) for k, v in d["coef"].items()}
            print(f"[libero] loaded {len(SC)} coefficients, median "
                  f"{sorted(SC.values())[len(SC)//2]:.3f}", flush=True)

    # The whole-stack weight is a SINGLE scalar (or one per cache age): a stack
    # residual has no cells to index. Fit on the TRAINING replay split only --
    # probes/fw_stackbeta.py writes this file and refuses to read evaluation
    # rollouts -- so it is comparable with the per-cell coefficients rather than
    # advantaged by seeing more episodes.
    WB = None
    needs_stack_weights = any(
        c.strip().split(":")[0] == "whole" and len(c.strip().split(":")) > 2
        and "s" in c.strip().split(":")[2] for c in str(pr.configs).split(","))
    if needs_stack_weights:
        WB = load_stack_weights(str(pr.stack_shrink), str(cfg.ckpt),
                                int(cfg.EVALUATION.get("num_inference_steps", N_STEPS)))
        print(f"[libero] validated whole-stack weight: {WB}", flush=True)

    # The suffix is a SET of flags, not one choice, so "sx" is the deployment
    # arm: attenuated and actually skipping. Single letters keep every existing
    # config string ("s", "o") meaning what it always did.
    #   s  apply the fitted per-cell weight
    #   o  oracle per-cycle weight (ceiling arm; never skips, by definition)
    #   x  actually skip the block instead of computing and discarding it
    specs = parse_configs(str(pr.configs), SC, WB, str(pr.shrink),
                          str(pr.stack_shrink))
    print(f"[libero] suite={pr.suite} tasks={pr.n_tasks} eps={pr.n_episodes} "
          f"perturb={pr.perturb} configs={[s[0] for s in specs]}", flush=True)

    # ONLY install the whole-stack hook when an arm actually uses it. `stock` is
    # the honest latency denominator; giving every arm an extra Python frame on
    # the per-denoise-step path would put instrumentation back into exactly the
    # baseline that deployfrontier.sh exists to keep clean.
    class _NoStack:
        on = False
        def reset(self, *a, **k): pass
        def zero_counts(self): pass
        def counts(self): return {"hits": 0, "misses": 0}
        def close(self): pass
    SH = StackHarness(model.mot) if any(sp[1] in ("whole", "whole_agg", "screen") for sp in specs) else _NoStack()
    caching = any(sp[1] not in (None, "screen") for sp in specs)
    SH.on = True

    # Hash the initial action latent exactly as FastWAM builds it, so the
    # artefact records the starting point the denoiser was actually given.
    # Under resample_noise there is no single latent to record, and drawing one
    # here would consume the global stream the rollouts use.
    _latent_digest = None
    if not resample and not episode_noise:
        try:
            _rd = str(cfg.EVALUATION.get("rand_device", "cpu"))
            _g = torch.Generator(device=_rd).manual_seed(run_seed)
            _latent_digest = tensor_digest(
                torch.randn((1, ah, model.action_expert.action_dim),
                            generator=_g, device=_rd, dtype=torch.float32))
        except Exception:
            pass

    TIME = bool(int(pr.timing)) and torch.cuda.is_available()
    DUMP = bool(int(pr.dump_chunks))
    SCREEN = None
    if any(sp[1] == "screen" for sp in specs):
        # The screen replays the per-step denoiser of each call, so it wraps the
        # callable upstream's infer_action uses (built here exactly as upstream
        # builds it lazily: compiled, around `traced`) and the infer_action
        # method itself, which hands it the uncached action. One arm per process.
        if len(specs) != 1:
            raise SystemExit("the screen arm runs alone in its process")
        import functools
        from screen_engine import ScreenEngine, StepCapture, parse_screen
        from libero_utils import invert_gripper_action
        _compiled = bool(cfg.EVALUATION.get("compile_action_infer", False))
        if _compiled:
            _cap = StepCapture(torch.compile(model._denoise_action_with_video_cache,
                                             mode="reduce-overhead", fullgraph=True))
            model._denoise_action_with_video_cache_compiled = _cap
        else:
            _cap = StepCapture(model._denoise_action_with_video_cache)
            model._denoise_action_with_video_cache = _cap
        _binarize = bool(cfg.EVALUATION.get("binarize_gripper", False))
        def _executed(action):
            # eval_libero_single._predict_action_chunk's post-processing
            a = E._denormalize_action(action.clone(), processor)[0]
            a[..., -1] = a[..., -1] * 2 - 1
            a = invert_gripper_action(a)
            if _binarize:
                a[..., -1] = np.sign(a[..., -1])
            return a
        _shift = cfg.EVALUATION.get("sigma_shift")
        SCREEN = ScreenEngine(model, _cap, SH, H, _executed, replan, n_steps,
                              None if _shift is None else float(_shift), _compiled,
                              configs=parse_screen(str(pr.screen)), timing=TIME, dump=DUMP)
        _infer = model.infer_action
        @functools.wraps(_infer)
        def _infer_screened(*a, **k):
            out = _infer(*a, **k)
            SCREEN.end_call(out["action"])
            return out
        model.infer_action = _infer_screened
        print(f"[libero] screen: {len(SCREEN.configs)} configurations "
              f"{[f'{m}/{t}' for m, t in SCREEN.configs]}", flush=True)
    rows, t_start = [], time.time()
    # first_task lets a run that died partway resume at the next task rather than
    # repeat the ones already written out
    for ti in range(int(pr.first_task), int(pr.n_tasks)):
        task = suite.get_task(ti)
        inits = suite.get_task_init_states(ti)
        first_ep = int(pr.first_episode)
        if not (0 < int(pr.n_episodes) and 0 <= first_ep
                and first_ep + int(pr.n_episodes) <= len(inits)):
            raise ValueError(f"Task {ti}: requested episodes {first_ep}..{first_ep + int(pr.n_episodes) - 1} "
                             f"but only {len(inits)} distinct initial states exist; refusing repeats")
        env, desc = get_libero_env(task, E.LIBERO_ENV_RESOLUTION, run_seed)
        for cname, pname, tau, sc, orc, skp in specs:
            norec = (skp == "norec")
            skp = False if norec else skp
            whole = (pname == "whole")
            whole_agg = (pname == "whole_agg")
            screen = (pname == "screen")
            cells = None if (whole or screen) else (PL["c3ache"] if whole_agg else (PL[pname] if pname else None))
            # A whole-stack arm caches the same denoise steps the per-block
            # placement covers, so the two differ in granularity and in nothing
            # else. Deriving it from the placement keeps that true if the step
            # count changes.
            wsteps = {k for k, _ in PL["c3ache"]} if (whole or whole_agg) else None
            for ep in range(first_ep, first_ep + int(pr.n_episodes)):
                # Seed global streams per (task, episode), independently of arm.
                # This is hygiene, not a demonstrated repair of arm-list effects:
                # results/rngdrift shows identical recorded rollouts with this
                # reseed on/off for a fixed arm list, but changes across arm lists.
                # FastWAM's explicit cfg.seed constructs a local generator on
                # every infer call, giving one fixed latent across this run.
                if not int(pr.no_reseed):
                    # The cause of the observed arm-list dependence remains
                    # unresolved; compiled graphs/kernel selection are candidates.
                    _s = run_seed * 1000003 + ti * 1009 + ep
                    torch.manual_seed(_s)
                    np.random.seed(_s % (2 ** 32))
                    random.seed(_s)
                    if episode_noise:
                        with open_dict(cfg):
                            cfg.seed = _s
                # Fingerprints document global RNG state; a mismatch alone
                # does not show that a rollout consumes the differing stream.
                fp = rng_fingerprint() if int(pr.rng_probe) else None
                init = inits[ep]
                env.reset(); obs = env.set_init_state(init)
                # seeded on (task, episode) only -- NOT on the arm -- so both
                # arms of a pair see exactly the same perturbed world
                ofn, efn = make_perturb(str(pr.perturb), 10007 * ti + ep)
                if efn is not None:
                    efn(env); obs, _, _, _ = env.step(get_libero_dummy_action())
                if ofn is not None: obs = ofn(obs)
                TIMER["ms"] = 0.0; TIMER["n"] = 0
                # PER-CYCLE TIMING, not just the mean. A periodic cache has two
                # populations -- the expensive refresh and the cheap reuse -- and
                # an episode mean hides both. Deployment cares about the tail: a
                # control loop misses its deadline on the slow cycle, not on the
                # average one. Each entry is (ms, was_this_a_refresh).
                TIMER["cyc"] = []
                cache, cycle, pending, t, done = {}, 0, [], 0, False
                refresh_cycles, reuse_cycles = 0, 0
                chunks = [] if DUMP else None
                chunk_prev = None
                if screen: SCREEN.reset_episode()
                whole_hits, whole_misses = 0, 0
                while t < max_steps + wait:
                    if t < wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action()); t += 1; continue
                    if not pending:
                        _active = wsteps if (whole or whole_agg) else cells
                        _is_refresh = (_active is None or cycle == 0
                                       or (tau > 0 and cycle % tau == 0))
                        if _is_refresh:
                            # norec: full compute, nothing recorded -- the stock arm
                            cache = {}
                            if screen:
                                SCREEN.begin_call(cycle)
                            elif whole:
                                SH.reset(record=cache, steps=None); H.reset()
                            elif whole_agg:
                                H.reset(record=cache); SH.reset()
                            else:
                                H.reset(record=None if norec else cache); SH.reset()
                        else:
                            # how many cycles since the last refresh: with
                            # interval tau the refresh is at floor(c/tau)*tau,
                            # so the cache in hand is (cycle % tau) cycles old
                            st = max(1, (cycle % tau) if tau > 0 else cycle)
                            if whole:
                                b = (sc.get(st, sc[max(sc)]) if isinstance(sc, dict)
                                     else (1.0 if sc is None else sc))
                                SH.reset(apply=cache, steps=wsteps, beta=b)
                                H.reset()
                            elif whole_agg:
                                SH.reset(apply=cache, steps=wsteps, beta=1.0)
                                H.reset()
                            else:
                                H.reset(apply=cache, cells=cells, scale=sc,
                                        staleness=st, oracle_beta=orc,
                                        skip=skp)                            # reuse
                                SH.reset()
                        # Count schedules outside the compiled model. A schedule
                        # count is not an observed block-cache hit count.
                        refresh_cycles += int(_is_refresh)
                        reuse_cycles += int(not _is_refresh)
                        SH.zero_counts()
                        # One replanning cycle, timed end to end: the encode and
                        # the denoise, which together are what a deployment pays
                        # per cycle. Deliberately NOT inside the cache wrapper --
                        # see the note on `traced` above. The synchronise either
                        # side makes the number a real elapsed time rather than a
                        # queue-submission time; it costs the same in every arm,
                        # so the comparison between arms is unaffected.
                        if TIME:
                            torch.cuda.synchronize(); _t0 = time.perf_counter()
                        chunk, _imgs, _ = E._predict_action_chunk(
                            obs=obs, task_description=desc, model=model, processor=processor,
                            cfg=cfg, action_horizon=ah, input_w=w, input_h=h,
                            model_device=str(dev))
                        if whole_agg and _is_refresh:
                            stack_cache = {}
                            for step in wsteps:
                                acc = sum((sc.get((step, b), 1.0) if sc else 1.0) * cache[(step, b)].float()
                                          for b in range(N_BLOCKS))
                                stack_cache[step] = acc.to(cache[(step, 0)].dtype)
                            cache = stack_cache
                        if TIME:
                            torch.cuda.synchronize()
                            _dt = (time.perf_counter() - _t0) * 1000.0
                            TIMER["ms"] += _dt
                            TIMER["n"] += 1
                            TIMER["cyc"].append((round(_dt, 3), bool(_is_refresh)))
                        if _active is not None and _is_refresh and not whole_agg and not _active.issubset(cache):
                            raise RuntimeError("Refresh did not record every selected cache entry.")
                        if whole_agg and _is_refresh and not wsteps.issubset(cache):
                            raise RuntimeError("Refresh did not aggregate every selected cache entry.")
                        if (whole or whole_agg) and not _is_refresh:
                            observed = SH.counts()
                            whole_hits += observed["hits"]
                            whole_misses += observed["misses"]
                            if observed["hits"] != len(wsteps) or observed["misses"]:
                                raise RuntimeError(f"Incomplete whole-stack reuse: {observed}")
                        _d = DELAY[0] if _is_refresh else DELAY[1]
                        pending = delayed_actions(chunk, chunk_prev, _d, replan, get_libero_dummy_action())
                        chunk_prev = chunk; cycle += 1
                        if DUMP: chunks.append([list(map(float, a)) for a in pending])
                    obs, _, done, _ = env.step(pending.pop(0))
                    if ofn is not None: obs = ofn(obs)
                    if done: break
                    t += 1
                row = {"suite": str(pr.suite), "task": ti, "task_name": desc,
                       "config": cname, "episode": ep, "success": bool(done),
                       "steps": t, "cycles": cycle,
                       "refresh_cycles": refresh_cycles, "reuse_cycles": reuse_cycles,
                       # The episode's actual starting point. Two runs that
                       # disagree cannot otherwise be told apart from two runs
                       # that were never given the same starting point -- which
                       # is exactly the ambiguity that made the arm-order defect
                       # take three attempts to explain.
                       "init_state": tensor_digest(init)}
                if whole or whole_agg:
                    row.update(whole_cache_hits=whole_hits, whole_cache_misses=whole_misses)
                if episode_noise:
                    # the latent every call of this episode started from
                    _rd = str(cfg.EVALUATION.get("rand_device", "cpu"))
                    row["episode_latent"] = tensor_digest(torch.randn(
                        (1, ah, model.action_expert.action_dim), dtype=torch.float32,
                        generator=torch.Generator(device=_rd).manual_seed(int(cfg.seed))
                    ).to(model.torch_dtype).float())
                if screen:
                    row["screen"] = SCREEN.episode_summary()
                if DUMP:
                    row["chunks"] = chunks
                if int(pr.timing) and TIMER["n"]:
                    # Per-cycle, so episodes of different length stay comparable.
                    # The first episodes of an arm pay compilation, which is a
                    # one-off per arm and not what a deployment sees; the
                    # analysis discards them via warmup_episodes.
                    row["ms_per_cycle"] = round(TIMER["ms"] / TIMER["n"], 3)
                    # The distribution, not just its mean. Kept only when asked
                    # for (PROBE.cycle_times=1): it is one float per replanning
                    # cycle and would otherwise inflate every artefact in the
                    # repository for the sake of two experiments that need it.
                    if int(pr.get("cycle_times", 0)):
                        row["cycle_ms"] = [c[0] for c in TIMER["cyc"]]
                        row["cycle_refresh"] = [c[1] for c in TIMER["cyc"]]
                    row["denoise_calls"] = TIMER["n"]
                    row["warmup"] = ep < int(pr.warmup_episodes)
                if fp is not None: row["rng"] = fp
                rows.append(row)
            sr = np.mean([r["success"] for r in rows if r["config"] == cname and r["task"] == ti])
            print(f"[libero] task {ti} {cname:14} SR={sr:.2f}  "
                  f"({(time.time()-t_start)/60:.1f} min)", flush=True)
        env.close()
        json.dump({"meta": {"suite": str(pr.suite), "perturb": str(pr.perturb),
                            "runtime": runtime,
                            "n_tasks": int(pr.n_tasks),
                            "n_episodes": int(pr.n_episodes), "replan": replan,
                            "max_steps": max_steps, "skip_cells": len(PL["c3ache"]),
                            "seed": run_seed,
                            # what differs from the stock protocol, if anything
                            "resample_noise": int(resample),
                            # recorded only when used, so ordinary runs keep
                            # the metadata of the runs shipped before them
                            **({"first_episode": int(pr.first_episode)}
                               if int(pr.first_episode) else {}),
                            **({"screen": [f"p{m}_tau{t}" for m, t in SCREEN.configs]}
                               if SCREEN is not None else {}),
                            **({"episode_noise": 1} if episode_noise else {}),
                            **({"delay": list(DELAY)} if any(DELAY) else {}),
                            "cudagraphs": int(CUDAGRAPHS),
                            # zero for arms that cache nothing, whatever the placement
                            "cache_prefix": cache_prefix if caching else 0,
                            "cached_path_fraction": round(path_fraction, 4) if caching else 0.0,
                            # The denoiser's initial latent. With seed= set this
                            # is one constant tensor for the whole run (see
                            # probes/check_seed_path.py), so it is recorded once
                            # rather than per episode -- and recording it is what
                            # lets a reviewer confirm that rather than take it on
                            # trust.
                            "initial_latent": _latent_digest,
                            "no_reseed": int(pr.no_reseed),
                            "rng_probe": int(pr.rng_probe),
                            "timing": int(pr.timing),
                            "warmup_episodes": int(pr.warmup_episodes),
                            # the baselines experiment varies this, and an
                            # artefact that does not record it cannot be told
                            # apart from one collected at the default
                            "num_inference_steps": int(
                                cfg.EVALUATION.get("num_inference_steps", 0)) or None,
                            "configs": [s[0] for s in specs],
                            # which code and which machine -- a repeat that
                            # disagrees is uninterpretable without both
                            "provenance": provenance(ckpt=str(cfg.get("ckpt", "")))},
                   "rows": rows},
                  open(str(pr.out), "w"))

    model._denoise_action_with_video_cache = dn; H.close(); SH.close()
    print("\n=== success rate by config ===")
    for cname in [sp[0] for sp in specs]:
        v = [r["success"] for r in rows if r["config"] == cname]
        if v: print(f"  {cname:16} {np.mean(v)*100:5.1f}%  (n={len(v)})")
    print(f"[libero] DONE {len(rows)} episodes in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

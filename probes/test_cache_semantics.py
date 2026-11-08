"""Verify the cache scheduling and substitution logic without loading a model.

Three things can silently be wrong in a residual cache and none of them raise:
  * which cycles refresh versus reuse (the tau schedule)
  * which cells are substituted at a given budget
  * whether the weight is actually applied to the substituted value

This exercises all three against the same code paths fw_libero.py uses, with a
stub standing in for the block so the expected output is known exactly.
"""
from __future__ import annotations
import sys
import torch

# import the harness without pulling in hydra/fastwam, which need a GPU box
import os
import importlib.util, types
for _m in ("hydra", "hydra.utils", "omegaconf"):
    if _m not in sys.modules:
        _stub = types.ModuleType(_m)
        _stub.main = lambda *a, **k: (lambda f: f)
        _stub.instantiate = lambda *a, **k: None
        _stub.open_dict = lambda cfg: __import__("contextlib").nullcontext()
        _stub.DictConfig = dict; _stub.OmegaConf = types.SimpleNamespace(
            merge=lambda *a: {}, create=lambda *a: {})
        sys.modules[_m] = _stub
# resolve against this file, not the cwd, so the test runs from anywhere
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("fw_libero", os.path.join(_here, "fw_libero.py"))
fw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fw)
CacheHarness, placements = fw.CacheHarness, fw.placements
N_BLOCKS, N_STEPS, SKIP = fw.N_BLOCKS, fw.N_STEPS, fw.SKIP

fails = 0
def check(cond, msg):
    global fails
    if not cond:
        print(f"  FAIL: {msg}"); fails += 1

# ---- 1. the tau schedule -------------------------------------------------
def schedule(tau, n_cycles, cells_is_none=False):
    """Mirror of the branch in fw_libero.main; returns 'R' refresh / 'C' cached."""
    out = []
    for cycle in range(n_cycles):
        if cells_is_none or cycle == 0 or (tau > 0 and cycle % tau == 0):
            out.append("R")
        else:
            out.append("C")
    return "".join(out)

print("tau schedules over 16 cycles:")
for tau, want in [(1, "R"*16), (2, "RC"*8), (4, "RCCC"*4), (8, "RCCCCCCC"*2),
                  (0, "R" + "C"*15)]:
    got = schedule(tau, 16)
    print(f"  tau={tau}: {got}")
    check(got == want, f"tau={tau} expected {want}")
check(schedule(8, 16, cells_is_none=True) == "R"*16, "full compute must never cache")

# compute saved = (fraction of cycles cached) x (fraction of cells skipped)
print("\ncompute saved, 240 of 300 cells skipped on a cached cycle:")
for tau, want in [(2, 40.0), (4, 60.0), (8, 70.0), (0, 80.0)]:
    s = schedule(tau, 1000)
    saved = 100 * (s.count("C") / len(s)) * (SKIP / (N_BLOCKS * N_STEPS))
    print(f"  tau={tau}: {saved:.1f}%  (expected {want}%)")
    check(abs(saved - want) < 1.0, f"tau={tau} compute saved {saved:.1f} != {want}")

# ---- 2. placements are disjoint-correct and matched-budget ---------------
P = placements(None)
print("\nplacements:")
for n, s in P.items():
    print(f"  {n:8s} {len(s)} cells")
    check(len(s) == SKIP, f"{n} skips {len(s)}, not {SKIP}")
    check(all(0 <= k < N_STEPS and 0 <= b < N_BLOCKS for k, b in s), f"{n} out of range")
check(P["c3ache"] == {(k, b) for k in range(8) for b in range(N_BLOCKS)},
      "c3ache placement should be steps 0-7, all blocks")

# A one- or two-step schedule has no steps before the final two.
# This exercises the production placement helper, rather than restating it.
for steps, cached_steps in ((1, 0), (2, 0), (3, 1), (4, 2), (6, 4), (10, 8)):
    short = placements(None, n_steps=steps)
    expected = {(k, b) for k in range(cached_steps) for b in range(N_BLOCKS)}
    check(short["c3ache"] == expected,
          f"{steps}-step schedule must leave the final two steps exact")
    check(len(short["random"]) == len(expected),
          f"{steps}-step random placement must have the same budget")
for invalid in (0, -1):
    try:
        placements(None, n_steps=invalid)
    except ValueError:
        pass
    else:
        check(False, f"non-positive step count {invalid} must be rejected")

# ---- 3. substitution actually applies the weight -------------------------
class Stub:
    """Stands in for the block: output = input * 3, so residual = input * 2."""
    def _apply_expert_post_block_tensor(self, *a, **k):
        return k["residual_x"] * 3.0

stub = Stub()
H = CacheHarness(stub)
x = torch.ones(1, 4, 8)

rec = {}
H.reset(record=rec); H.begin()
out = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(out, x * 3), "record pass must not alter the output")
check(len(rec) == 1, f"expected 1 cell recorded, got {len(rec)}")
resid = rec[(0, 0)]
check(torch.allclose(resid, x * 2), "recorded residual should be out - in")

cells = {(0, 0)}
for beta, want in [(1.0, 3.0), (0.5, 2.0), (0.0, 1.0), (0.8, 2.6)]:
    scale = None if beta == 1.0 else {(0, 0): beta}
    H.reset(apply=rec, cells=cells, scale=scale); H.begin()
    o = stub._apply_expert_post_block_tensor(residual_x=x)
    H.end()
    got = float(o.flatten()[0])
    print(f"  beta={beta}: output {got:.2f} (expect {want:.2f} = input + beta*residual)")
    check(abs(got - want) < 1e-6, f"beta={beta} gave {got}, expected {want}")

# a cell outside the placement must be recomputed, whatever the weight says
H.reset(apply=rec, cells=set(), scale={(0, 0): 0.0}); H.begin()
o = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(o, x * 3), "cells outside the placement must not be substituted")
# ---------------------------------------------------------------------------
# Staleness-aware weights. A cache at interval tau is reused at staleness
# 1..tau-1, and the weight applied must match the staleness actually in hand.
# Getting this wrong is silent: it yields plausible numbers from the wrong
# coefficient, which is exactly how the earlier runs under-attenuated on two of
# every three cached cycles at tau=4.
print("\nstaleness-aware substitution")
SC_K = {(0, 0): {1: 0.9, 2: 0.5, 3: 0.2}}
rec_k = {(0, 0): torch.full_like(x, 2.0)}
for st, b in ((1, 0.9), (2, 0.5), (3, 0.2)):
    H.reset(apply=rec_k, cells={(0, 0)}, scale=SC_K, staleness=st); H.begin()
    o = stub._apply_expert_post_block_tensor(residual_x=x)
    H.end()
    want = x + b * 2.0
    check(torch.allclose(o, want, atol=1e-6),
          f"staleness {st} must use beta={b} (got {o.flatten()[0]:.3f}, "
          f"want {want.flatten()[0]:.3f})")

# a plain float still means "this weight at every staleness"
H.reset(apply=rec_k, cells={(0, 0)}, scale={(0, 0): 0.7}, staleness=3); H.begin()
o = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(o, x + 0.7 * 2.0, atol=1e-6),
      "a uniform float weight must ignore staleness")

# past the fitted range, clamp to the stalest fitted weight rather than falling
# back to 1.0, which would UNDER-attenuate precisely the stalest cycles
H.reset(apply=rec_k, cells={(0, 0)}, scale=SC_K, staleness=9); H.begin()
o = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(o, x + 0.2 * 2.0, atol=1e-6),
      "staleness past the fitted range must clamp to the stalest weight, not 1.0")

# ---------------------------------------------------------------------------
# Oracle-weight arm. The block is computed, so the true residual is available and
# the weight chosen must be the exact least-squares optimum for that cycle. Two
# cases have known answers: a cached residual equal to the true one must get
# beta=1, and one that is twice as long must get beta=0.5.
print("\noracle substitution weight")
# stub emits output = input*3, so the true residual is input*2
for factor, want_beta in ((1.0, 1.0), (2.0, 0.5), (0.5, 1.0)):
    rec_o = {(0, 0): torch.full_like(x, 2.0) * factor}
    H.reset(apply=rec_o, cells={(0, 0)}, oracle_beta=True); H.begin()
    o = stub._apply_expert_post_block_tensor(residual_x=x)
    H.end()
    # beta* = <R_true, c>/<c,c> = (2*2f)/(2f)^2 = 1/f, clamped to [0, 1.5]
    b = min(1.5, 1.0 / factor)
    want = x + b * (2.0 * factor)
    check(torch.allclose(o, want, atol=1e-5),
          f"oracle beta for a cache {factor}x the true residual "
          f"(got {o.flatten()[0]:.3f}, want {want.flatten()[0]:.3f})")

# with the exactly-correct cached residual the oracle arm must reproduce full
# compute: that is the same algebraic identity the cache gate relies on
rec_id = {(0, 0): torch.full_like(x, 2.0)}
H.reset(apply=rec_id, cells={(0, 0)}, oracle_beta=True); H.begin()
o = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(o, x * 3, atol=1e-5),
      "oracle arm with the true residual must equal full compute")

# a zero-norm cached residual carries nothing to scale: the weight must come out
# zero rather than raising or defaulting to 1.0
rec_z = {(0, 0): torch.zeros_like(x)}
H.reset(apply=rec_z, cells={(0, 0)}, oracle_beta=True); H.begin()
o = stub._apply_expert_post_block_tensor(residual_x=x)
H.end()
check(torch.allclose(o, x, atol=1e-6),
      "oracle arm with a zero cached residual must emit the input unchanged")

# ---- 4. skipping is EQUIVALENT to substituting, and actually skips -------
# The deployment arm must return the same tensor as the substitute arm, and must
# not call the block. A stub that counts calls settles both at once: any drift
# between the two paths would make every quality number in this study describe a
# schedule no deployment runs.
class CountingStub:
    def __init__(self): self.calls = 0
    def _apply_expert_post_block_tensor(self, *a, **k):
        self.calls += 1
        return k["residual_x"] * 3.0

for beta, scale in [(1.0, None), (0.5, {(0, 0): 0.5}), (0.0, {(0, 0): 0.0}),
                    (0.717, {(0, 0): 0.717})]:
    subs, skips = CountingStub(), CountingStub()
    got = []
    for st, tgt in ((subs, False), (skips, True)):
        h = CacheHarness(st)
        h.reset(apply=rec, cells=cells, scale=scale, skip=tgt); h.begin()
        got.append(st._apply_expert_post_block_tensor(residual_x=x))
        h.end(); h.close()
    check(torch.equal(got[0], got[1]),
          f"skip must equal substitute at beta={beta}: {got[0].flatten()[0]} "
          f"vs {got[1].flatten()[0]}")
    check(subs.calls == 1, f"substitute arm should call the block once, got {subs.calls}")
    check(skips.calls == 0, f"skip arm must NOT call the block, got {skips.calls}")

# a cell that is NOT cached must still run, even in a skipping arm
st = CountingStub(); h = CacheHarness(st)
h.reset(apply=rec, cells={(9, 9)}, skip=True); h.begin()
o = st._apply_expert_post_block_tensor(residual_x=x)
h.end(); h.close()
check(st.calls == 1, "an uncached cell must still be computed in a skipping arm")
check(torch.allclose(o, x * 3), "an uncached cell must return the block's output")

# recording and the oracle weight both need the true residual, so neither skips
for kw in ({"record": {}}, {"apply": rec, "cells": cells, "oracle_beta": True}):
    st = CountingStub(); h = CacheHarness(st)
    h.reset(skip=True, **kw); h.begin()
    st._apply_expert_post_block_tensor(residual_x=x)
    h.end(); h.close()
    check(st.calls == 1, f"skip must be ignored when {list(kw)[0]} is set")

# Whole-stack versus per-block caching is an exact-arithmetic identity only.
# This uses the actual harness and BF16 representable values: a perfect
# same-input replay does not guarantee agreement on the next chunk's input.
class SmallIncrementStub:
    def _apply_expert_post_block_tensor(self, *a, **k):
        return k["residual_x"] + 1.0 / 128.0

print("\nwhole-stack reference counterexample (synthetic tensors)")
for dtype in (torch.float64, torch.bfloat16):
    st = SmallIncrementStub(); h = CacheHarness(st)
    initial = torch.ones(1, 1, 1, dtype=dtype)
    recorded = {}
    h.reset(record=recorded); h.begin()
    fresh = initial
    for _ in range(N_BLOCKS):
        fresh = st._apply_expert_post_block_tensor(residual_x=fresh)
    h.end()
    selected = {(0, b) for b in range(N_BLOCKS)}
    for incoming in (initial, initial * 2):
        h.reset(apply=recorded, cells=selected, skip=True); h.begin()
        per_block = incoming
        for _ in range(N_BLOCKS):
            per_block = st._apply_expert_post_block_tensor(residual_x=per_block)
        h.end()
        whole_stack = incoming + (fresh - initial)
        if torch.equal(incoming, initial):
            check(torch.equal(per_block, fresh), "synthetic self-replay must pass")
        if dtype == torch.float64:
            check(torch.equal(per_block, whole_stack), "exactly representable FP64 sums must telescope")
        elif not torch.equal(incoming, initial):
            check(per_block.item() == 2.0, "BF16 repeated small additions round away")
            check(whole_stack.item() == 2.234375, "BF16 whole-stack residual retains the aggregate")
            print(f"  BF16 self-replay passes; stale input: per-block={per_block.item()}, "
                  f"whole-stack={whole_stack.item()}")
    h.close()

# ---- 5. nothing untraceable inside the compiled-region wrapper -----------
# `traced` wraps the denoise call, so dynamo traces it. A CUDA event or a host
# timer in there is not a graph break you can shrug at: torch's own graph-break
# formatter raised "'tuple' object has no attribute 'splitlines'", which names
# neither the cause nor the line, and the run died a minute in having already
# paid for setup. Structural, because it is not reproducible without a GPU.
import ast as _ast
_src = open(os.path.join(_here, "fw_libero.py"), encoding="utf-8").read()
_traced = None
for _n in _ast.walk(_ast.parse(_src)):
    if isinstance(_n, _ast.FunctionDef) and _n.name == "traced":
        _traced = _n; break
check(_traced is not None, "fw_libero.py should define the traced wrapper")
if _traced is not None:
    _banned = []
    for _n in _ast.walk(_traced):
        if isinstance(_n, _ast.Attribute):
            _parts = []
            _c = _n
            while isinstance(_c, _ast.Attribute): _parts.append(_c.attr); _c = _c.value
            if isinstance(_c, _ast.Name): _parts.append(_c.id)
            _dotted = ".".join(reversed(_parts))
            for _bad in ("torch.cuda", "time.perf_counter", "time.time", "torch.profiler"):
                if _dotted.startswith(_bad): _banned.append(_dotted)
    check(not _banned,
          "traced() must contain nothing dynamo cannot trace; found: "
          + ", ".join(sorted(set(_banned))))

H.close()
check(stub._apply_expert_post_block_tensor(residual_x=x).allclose(x * 3), "close() must restore")

# ---- 6. the WHOLE-STACK cache, and where it parts company with per-block ----
# C3ache records one residual across the stack and adds it once; CacheHarness
# records thirty and adds thirty. The identity that makes them equal holds in
# exact arithmetic, and the model runs in BF16. These checks pin the harness's
# behaviour and then measure the gap directly rather than assuming it away.
StackHarness = fw.StackHarness


class StackStub:
    """Stands in for the 30-block stack: a counted, exactly-representable map."""
    def __init__(self, gain=3.0):
        self.calls = 0; self.gain = gain
    def forward_action_with_video_cache_tensor(self, *a, **k):
        self.calls += 1
        return k["action_tokens"] * self.gain


xs = torch.full((1, 4, 8), 1.0)

# record: one residual per denoise step, equal to out - in
st = StackStub(); sh = StackHarness(st); sh.on = True
rec = {}
sh.reset(record=rec, steps=None)
for _ in range(3): st.forward_action_with_video_cache_tensor(action_tokens=xs)
check(sorted(rec) == [0, 1, 2], f"whole-stack record must key by denoise step, got {sorted(rec)}")
check(torch.allclose(rec[0], xs * 2.0), "whole-stack residual must be out - in")
check(st.calls == 3, f"recording must still run the stack, got {st.calls}")

# substitute: returns in + beta*residual and does NOT run the stack
for beta in (1.0, 0.5, 0.0, 0.717):
    st2 = StackStub(); sh2 = StackHarness(st2); sh2.on = True
    sh2.reset(apply={0: xs * 2.0}, steps={0}, beta=beta)
    got = st2.forward_action_with_video_cache_tensor(action_tokens=xs)
    check(torch.allclose(got, xs + beta * xs * 2.0),
          f"whole-stack substitution wrong at beta={beta}: {got.flatten()[0]}")
    check(st2.calls == 0, f"whole-stack hit must NOT run the stack, got {st2.calls}")
    check(sh2.counts() == {"hits": 1, "misses": 0}, f"hit not counted: {sh2.counts()}")
    sh2.close()

# a MISS must be counted, not silently absorbed. gate_cache reported a requested
# cell count as though it were an observed hit count; an absent or wrong-shaped
# entry falls back to computing, and nothing said so.
for bad in ({}, {0: torch.zeros((1, 4, 99))}):
    st3 = StackStub(); sh3 = StackHarness(st3); sh3.on = True
    sh3.reset(apply=bad, steps={0})
    st3.forward_action_with_video_cache_tensor(action_tokens=xs)
    check(sh3.counts() == {"hits": 0, "misses": 1},
          f"a fallback to compute must count as a miss: {sh3.counts()}")
    check(st3.calls == 1, "a miss must actually run the stack")
    sh3.close()

# an uncached step must still run
st4 = StackStub(); sh4 = StackHarness(st4); sh4.on = True
sh4.reset(apply={0: xs * 2.0}, steps={7})
st4.forward_action_with_video_cache_tensor(action_tokens=xs)
check(st4.calls == 1, "an uncached denoise step must still run the stack")
sh4.close()

# THE POINT OF THE CLASS. Per-block and whole-stack substitution are equal in
# exact arithmetic and need not be in BF16. This does not claim the difference
# explains any task-success gap; it establishes that self-replay agreement
# cannot rule the difference out, which is what the equivalence argument needed.
def per_block(x0, residuals, dt):
    """Thirty adds, rounded to `dt` after each one -- what CacheHarness does."""
    x = x0.to(dt)
    for r in residuals: x = (x + r.to(dt)).to(dt)
    return x.float()


def whole_stack(x0, residuals, dt):
    """One add of one recorded residual -- what StackHarness does. The residual
    is h_L - h_0 as the model produced it, so it is a single value in `dt`."""
    tot = (per_block(x0, residuals, dt) - x0.float()).to(dt)
    return (x0.to(dt) + tot).to(dt).float()


torch.manual_seed(0)
_x0 = torch.randn(64)
_rs = [torch.randn(64) * 0.05 for _ in range(N_BLOCKS)]
_gap = (per_block(_x0, _rs, torch.bfloat16)
        - whole_stack(_x0, _rs, torch.bfloat16)).abs().max().item()
_exact = (per_block(_x0, _rs, torch.float32)
          - whole_stack(_x0, _rs, torch.float32)).abs().max().item()
print(f"\nper-block vs whole-stack over {N_BLOCKS} blocks:")
print(f"  bf16 max |difference|  {_gap:.3e}")
print(f"  fp32 max |difference|  {_exact:.3e}   (the identity itself)")
check(_exact < 1e-5, "the telescoping identity must hold in fp32")
check(_gap >= _exact,
      "BF16 per-block accumulation must not be MORE exact than one whole-stack add")

sh.close()
check(st.forward_action_with_video_cache_tensor(action_tokens=xs).allclose(xs * 3),
      "StackHarness.close() must restore")

# ---- 7. config strings map to the arms they name ------------------------
# A mislabelled arm does not raise: it files whole-stack results under a
# per-block name, which is the exact confusion this diagnostic exists to remove.
parse_configs = fw.parse_configs
SCx = {(0, 0): 0.7}
WBx = {1: 0.9, 4: 0.6}

got = {sp[0]: sp for sp in parse_configs(
    "full,stock,c3ache:8,c3ache:8:s,c3ache:8:sx,c3ache:4:o,whole:8,whole:8:s,whole:8:a",
    SC=SCx, WB=WBx)}
want = {
    "full":                    (None,      None, None, False, False),
    "stock":                   (None,      None, None, False, "norec"),
    "c3ache_tau8":             ("c3ache",  8,    None, False, False),
    "c3ache_tau8_shrink":      ("c3ache",  8,    SCx,  False, False),
    "c3ache_tau8_shrink_skip": ("c3ache",  8,    SCx,  False, True),
    "c3ache_tau4_oracle":      ("c3ache",  4,    None, True,  False),
    "whole_tau8":              ("whole",   8,    None, False, False),
    "whole_tau8_shrink":       ("whole",   8,    WBx,  False, False),
    "whole_agg_tau8":          ("whole_agg", 8,  SCx,  False, False),
}
check(set(got) == set(want), f"labels: got {sorted(got)}, want {sorted(want)}")
for lab, exp in want.items():
    if lab not in got: continue
    check(got[lab][1:] == exp, f"{lab}: got {got[lab][1:]}, want {exp}")

# a whole-stack arm must never be filed under a per-block label, and vice versa
check(all(sp[1] in ("whole", "whole_agg") for lab, sp in got.items() if lab.startswith("whole")),
      "a 'whole_' label must carry the whole placement")
check(not any(lab.startswith("whole") for lab, sp in got.items()
              if sp[1] == "c3ache"), "a per-block arm must not get a whole_ label")

# the flags that cannot combine must be refused, not silently dropped
for bad, why in (("whole:8:o", "oracle weight on a stack residual"),
                 ("c3ache:8:ox", "oracle cannot skip"),
                 ("c3ache:8:q", "unknown flag"),
                 ("c3ache", "missing tau"),
                 ("c3ache:s", "non-numeric tau")):
    try:
        parse_configs(bad, SC=SCx, WB=WBx); check(False, f"{bad!r} must be refused ({why})")
    except SystemExit:
        pass

# asking for a weight with no weights file must fail loudly
for bad in ("c3ache:8:s", "whole:8:s"):
    try:
        parse_configs(bad, SC=None, WB=None)
        check(False, f"{bad!r} must be refused when no weights are loaded")
    except SystemExit:
        pass

print("\nALL CACHE-SEMANTICS CHECKS PASSED" if fails == 0 else f"\n{fails} FAILURES")
sys.exit(1 if fails else 0)

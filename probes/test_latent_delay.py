"""Check the delayed-execution schedule and the per-episode latent of fw_libero.py, without a GPU.

1. delayed_actions: with a delay of d control steps, the action executed at
   every time T is the one planned for T by the newest chunk whose observation
   is at least d steps old; with d = 0 it is synchronous execution; before the
   first chunk arrives the robot holds still.
2. parse_delay accepts D and R/U and refuses delays the chunk cannot cover.
3. episode_noise relies on a CPU generator seeded with s drawing exactly what
   the global generator draws after torch.manual_seed(s), which is how the
   fresh-noise arms draw their first latent of an episode.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types

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

fails = 0
def check(cond, msg):
    global fails
    if not cond:
        print(f"  FAIL: {msg}"); fails += 1

REPLAN, H = 10, 32
DUMMY = [0.0] * 6 + [-1.0]


def simulate(delays, n_calls=6):
    """Run the schedule; each chunk's action i is tagged (call time, planned time = call time + i)."""
    executed, prev, t = [], None, 0
    for c in range(n_calls):
        chunk = [[t, t + i] + [0.0] * 5 for i in range(H)]
        d = delays[c % len(delays)]
        acts = fw.delayed_actions(chunk, prev, d, REPLAN, DUMMY)
        check(len(acts) == REPLAN, f"call {c}: {len(acts)} actions")
        for k, a in enumerate(acts):
            executed.append((t + k, a, d))
        prev, t = chunk, t + REPLAN
    return executed


for d in (0, 1, 2, 5, 9):
    ex = simulate([d])
    bad = 0
    for T, a, d_ in ex:
        if a == DUMMY:
            bad += T >= d_          # holding still only before the first chunk arrives
            continue
        call_t, planned = a[0], a[1]
        bad += planned != T                         # the action was planned for this time
        bad += T - call_t < d_ and T >= d_          # computed from an observation at least d old
        newest = max(ct for ct in range(0, T + 1, REPLAN) if ct <= T - d_) if T >= d_ else None
        bad += newest is not None and call_t != newest   # by the newest chunk that has arrived
    check(bad == 0, f"delay {d}: {bad} misplaced actions")
    print(f"  delay {d}: every executed action is the newest available plan for its time step")
check(simulate([0])[:REPLAN] == [(T, [0, T] + [0.0] * 5, 0) for T in range(REPLAN)], "delay 0 is synchronous")
ex = simulate([5, 2])               # refresh and reuse calls with different delays
check(all((a == DUMMY) == (T < 5) for T, a, _ in ex[:REPLAN]), "first chunk: hold still for its delay")

check(fw.parse_delay("0", REPLAN, H) == (0, 0), "0")
check(fw.parse_delay("5", REPLAN, H) == (5, 5), "D")
check(fw.parse_delay("5/2", REPLAN, H) == (5, 2), "R/U")
for bad in ("10", "-1", "a", "5/2/1", "23"):
    try:
        fw.parse_delay(bad, REPLAN, 32 if bad != "23" else 30); check(False, f"accepted {bad!r}")
    except SystemExit:
        pass

for s in (42 * 1000003 + 3 * 1009 + 4, 7 * 1000003 + 9 * 1009 + 2):
    torch.manual_seed(s); a = torch.randn((1, H, 7), dtype=torch.float32)
    b = torch.randn((1, H, 7), dtype=torch.float32, generator=torch.Generator(device="cpu").manual_seed(s))
    check(torch.equal(a, b), f"seed {s}: per-episode latent differs from the reseeded global draw")
print("  a CPU generator seeded per episode draws the fresh-noise arm's first latent")

if fails:
    print(f"latent/delay: {fails} FAILED"); sys.exit(1)
print("latent/delay: schedules, refusals and the per-episode latent as specified")

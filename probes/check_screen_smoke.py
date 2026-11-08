"""Check that a screen replay returns what the closed-loop cache returns.

    python3 probes/check_screen_smoke.py DIR      # DIR holds smoke_screen.json and smoke_whole8.json

Both smoke runs (repro/experiments/screen.sh @smoke) play the first episode of
task 0 with Optional-IDM at s = 1 and record their executed actions. Their first
calls are identical uncached calls (the cache's refresh, the screen's uncached
call), so the robot is in the same state at the second call, which is the
cache's first reuse call. There the screen's replay of the same configuration
(8 cached steps, tau = 8) must return the cache's action. Exit status 1 if
either comparison differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv):
    d = Path(argv[1] if len(argv) > 1 else ".")
    sc = json.loads((d / "smoke_screen.json").read_text())["rows"][0]
    wh = json.loads((d / "smoke_whole8.json").read_text())["rows"][0]
    if (sc["task"], sc["episode"], sc["init_state"]) != (wh["task"], wh["episode"], wh["init_state"]):
        print("smoke runs did not start from the same state"); return 1
    first = float(np.abs(np.array(sc["chunks"][0]) - np.array(wh["chunks"][0])).max())
    rep = sc["screen"]["exec"]["p8_tau8"]
    cyc = sc["screen"]["cycle"]["p8_tau8"]
    if not rep or cyc[0] != 1:
        print(f"no replay at the first reuse call (cycles {cyc[:3]})"); return 1
    reuse = float(np.abs(np.array(rep[0]) - np.array(wh["chunks"][1])).max())
    moved = float(np.abs(np.array(sc["chunks"][1]) - np.array(wh["chunks"][1])).max())
    print(f"first call, uncached vs refresh:        max |difference| = {first:.3g}")
    print(f"first reuse call, replay vs closed loop: max |difference| = {reuse:.3g}")
    print(f"(the uncached action there differs from the cache's by up to {moved:.3g};"
          f" d = {sc['screen']['d']['p8_tau8'][0]:.4g})")
    ok = first == 0.0 and reuse == 0.0
    print("screen smoke check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Score training coefficients on held-out whole-stack sufficient statistics."""
from __future__ import annotations

import json
import os
import sys

from stack_probe_common import fitted_beta, score_beta, validate_artifact


def main(path):
    if not os.path.exists(path):
        print(f"SKIP: no artifact at {path}")
        return 0
    with open(path) as fh:
        data = json.load(fh)
    try:
        validate_artifact(data, "stackbeta")
    except ValueError as exc:
        print(str(exc))
        return 2
    stats = data.get("sufficient_statistics")
    if not stats or not stats.get("train") or not stats.get("held"):
        print("EXCLUDED: train/held-out sufficient statistics are absent.")
        return 2
    print(f"{os.path.relpath(path)}: cached steps {data['meta']['cached_steps']}")
    print("Coefficients fitted on training episodes; MSE evaluated on held-out episodes:")
    for age in sorted(stats["train"], key=int):
        beta = fitted_beta(stats["train"][age])
        held = score_beta(stats["held"][age], beta)
        if held is None:
            print(f"EXCLUDED: age {age} lacks sufficient training/held-out observations.")
            return 2
        print(f"  age={age}: beta={beta:.6g}, held-out MSE={held['mse']:.6g}, "
              f"plain MSE={held['plain_mse']:.6g}, "
              f"held-out tensor pairs={stats['held'][age]['n_pairs']}")
    print("This is replay reconstruction evidence; task-success benefits require closed-loop evaluation.")
    return 0


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "stackbeta", "stack_shrink_idm.json")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))

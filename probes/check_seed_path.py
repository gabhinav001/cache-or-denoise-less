"""Which random stream does the denoiser's initial latent actually come from?

This exists because an earlier account of the A8 pairing defect named the wrong
one. It said the denoiser's noise stream advanced continuously through a run, so
an arm inherited whatever state the arms before it left behind, and the two arms
of a pair were therefore not seeing the same draw. The measured symptom was real
-- see check_arm_position.py -- but that mechanism is not what produced it.

The action latent is drawn in FastWAM like this:

    eval_libero_single._predict_action_chunk   passes  seed = int(cfg.seed)
    fastwam.FastWAM.infer_action               builds  torch.Generator().manual_seed(seed)
                                               draws   torch.randn(..., generator=...)

A generator constructed inside the call, from a seed that does not change between
calls, yields the SAME tensor every time and is untouched by torch.manual_seed().
So in any run that passes seed= on the command line -- which is every closed-loop
experiment behind a headline number here -- the initial action latent is a
constant: identical across episodes, and identical across arms. It was already
perfectly paired, and the per-episode reseed cannot have changed it.

The per-episode reseed controls global streams, not this local generator.
The later rngdrift artifacts show that switching global reseeding on/off leaves
recorded rollouts unchanged within a fixed arm list. It therefore did not fix
the measured arm-list dependence, whose cause remains unresolved.

This script checks both halves so the claim cannot rot: the source structure, by
locating the two call sites, and the semantics, by demonstrating the invariance
on CPU. No GPU, no checkpoint.

usage: check_seed_path.py [path-to-FastWAM]        (default $WAM_ROOT, else /w)
"""
from __future__ import annotations

import os, re, sys

FAIL = 0


def bad(msg):
    global FAIL
    FAIL = 1
    print(f"  FAIL  {msg}")


def find(root, rel, pattern, what):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        print(f"  SKIP  {rel} not present under {root}")
        return None
    for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
        if re.search(pattern, line):
            print(f"  ok    {what}\n          {rel}:{i}  {line.strip()[:90]}")
            return i
    bad(f"{what}: no line matching /{pattern}/ in {rel}")
    return None


def main():
    root = (sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("WAM_ROOT")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "FastWAM"))
    root = os.path.abspath(root)
    print(f"\n  SOURCE STRUCTURE   ({root})\n")
    find(root, "experiments/libero/eval_libero_single.py",
         r'"seed":\s*None if cfg\.get\("seed"\) is None else int\(cfg\.seed\)',
         "the evaluator forwards cfg.seed into inference")
    find(root, "src/fastwam/models/wan22/fastwam.py",
         r"generator\s*=\s*None if seed is None else torch\.Generator\(",
         "inference builds a fresh local generator from that seed")
    find(root, "src/fastwam/models/wan22/fastwam.py",
         r"generator=generator",
         "the action latent is drawn from it")

    print("\n  SEMANTICS   (CPU, no model)\n")
    try:
        import torch
    except ImportError:
        print("  SKIP  torch not importable"); return FAIL

    def local_draw(seed=7):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(4, generator=g, device="cpu")

    torch.manual_seed(1); a = local_draw()
    torch.manual_seed(999); b = local_draw()
    _ = torch.randn(100)                      # advance the global stream, as a rollout does
    c = local_draw()
    if torch.equal(a, b) and torch.equal(a, c):
        print("  ok    a local generator's draw is invariant to torch.manual_seed()")
        print("          and to global stream position -- three draws identical")
    else:
        bad("local-generator draw changed under global reseeding")

    torch.manual_seed(1); d = torch.randn(4)
    torch.manual_seed(999); e = torch.randn(4)
    if not torch.equal(d, e):
        print("  ok    a generator=None draw DOES follow the global stream")
    else:
        bad("global draw did not change under reseeding")

    print("\n  CONCLUSION\n")
    print("    With seed= set, the initial action latent is a constant within a run:")
    print("    the same tensor for every episode and every arm. The per-episode")
    print("    reseed governs global streams without changing this local draw.")
    print("    These checks do not establish arm-order independence; the shipped")
    print("    rngdrift controls show that global reseeding did not remove it.")
    print("    An infer call without an explicit seed instead uses the global stream.")
    print(f"\n  {'PASS' if not FAIL else 'FAILED'}\n")
    return FAIL


if __name__ == "__main__":
    sys.exit(main())

"""Would repro/regenerate_paper.sh rewrite every raw run behind the paper, as it ran?

    python3 probes/check_regeneration.py [--verbose]

The paper's numbers come from results/budget_shift/summary.json, which
probes/report_budget_shift.py derives from raw run files in several results
directories. repro/regeneration_plan.txt names, for each directory, the
experiment and arm groups that regenerate it. This asks every plan line, by dry
run, which probe invocations it would issue, and checks each file the analysis
reads against them:

  * some plan line writes a file of that name into that directory, and
  * the invocation's settings equal those recorded inside the shipped file
    (checkpoint and model class, suite, tasks, episodes, seed, denoising steps,
    time shift, cache configuration and prefix, fresh noise, camera jitter,
    CUDA graphs, reseeding and timing flags).

A file the analysis reads that no line would rewrite, or would rewrite with
different settings, is reported and the exit status is 1. CPU only; nothing runs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLAN = ROOT / "repro" / "regeneration_plan.txt"

# What the analysis reads (probes/report_budget_shift.py): every run it loads,
# every gate it reports, and the process-sharing pilot. Traced rather than listed,
# so a new comparison in the analysis cannot be missed here.
def analysis_inputs(results=RESULTS):
    sys.path.insert(0, str(ROOT / "probes"))
    import report_budget_shift as rb
    used = set()
    orig = rb.load

    def traced(path):
        r = orig(path)
        if r is not None:
            used.add(Path(path).resolve())
        return r
    rb.load = traced
    argv, sys.argv = sys.argv, ["report_budget_shift.py", "--results", str(results),
                                "--out", os.devnull]
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rb.main()
    finally:
        sys.argv, rb.load = argv, orig
    used |= {p.resolve() for p in (results / "budget_shift").rglob("gate*.json")
             if "repeats_a6000" not in p.parts}
    used |= {p.resolve() for p in (results / "rngdrift").glob("arms*_*.json")}
    mc_gate = results / "matched_composed" / "gate.json"
    if mc_gate.is_file():
        used.add(mc_gate.resolve())
    return sorted(str(p.relative_to(results.resolve())) for p in used)


def read_plan(path=PLAN):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tier, d, exp, seeds, env, hours = (c.strip() for c in line.split("|"))
        rows.append({"tier": tier, "dir": d, "experiment": exp,
                     "seeds": "" if seeds == "-" else seeds,
                     "env": {} if env == "-" else dict(kv.split("=", 1) for kv in env.split()),
                     "hours": float(hours)})
    return rows


def dry_run(row, out):
    cmd = ["bash", str(ROOT / "repro" / "run_experiment.sh"), row["experiment"], "--out", str(out)]
    if row["seeds"]:
        cmd[3:3] = ["--seeds", row["seeds"]]
    env = dict(os.environ, DRY_RUN="1", **row["env"])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"dry run of {row['experiment']} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    calls, cur = {}, None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith(("python probes/", "python3 probes/")):
            args = {}
            parts = s.split()
            for a in parts[2:]:
                if "=" in a:
                    k, v = a.split("=", 1)
                    args[k] = v.strip("'\"")
            cur = {"probe": parts[1], "args": args, "env": {}}
            out_ = args.get("+PROBE.out")
            if out_:
                calls[os.path.basename(out_)] = cur
        elif s.startswith("env ") and cur is not None:
            k, v = s[4:].split("=", 1)
            cur["env"][k] = v
    return calls


def labels(configs):
    """fw_libero.parse_configs' names for a config string, without importing torch."""
    out = []
    for c in configs.split(","):
        c = c.strip()
        if not c:
            continue
        if c in ("full", "stock", "screen"):
            out.append(c); continue
        parts = c.split(":")
        name, tau = parts[0], int(parts[1])
        suf = parts[2] if len(parts) > 2 else ""
        if name == "whole":
            out.append(f"whole_agg_tau{tau}" if "a" in suf else f"whole_tau{tau}" + ("_shrink" if "s" in suf else ""))
        else:
            out.append(f"{name}_tau{tau}" + ("_shrink" if ("s" in suf or "a" in suf) else "")
                       + ("_oracle" if "o" in suf else "") + ("_skip" if "x" in suf else ""))
    return out


MODEL_CLASS = {"libero_optional_idm_2cam224_1e-4": "fastwam.models.wan22.fastwam_optional_idm.FastWAMOptionalIDM",
               "libero_uncond_2cam224_1e-4": "fastwam.models.wan22.fastwam.FastWAM",
               None: "fastwam.models.wan22.fastwam.FastWAM"}


def compare_run(call, meta):
    """Differences between a dry-run invocation and a shipped run's recorded settings."""
    a, diffs = call["args"], []
    rt = meta.get("runtime") or {}
    pv = meta.get("provenance") or {}

    def want(field, got, exp):
        if got is not None and exp is not None and str(got) != str(exp):
            diffs.append(f"{field}: shipped {got!r}, script {exp!r}")
    want("suite", meta.get("suite"), a.get("+PROBE.suite", "libero_goal"))
    want("n_tasks", meta.get("n_tasks"), a.get("+PROBE.n_tasks"))
    want("n_episodes", meta.get("n_episodes"), a.get("+PROBE.n_episodes"))
    want("seed", meta.get("seed"), a.get("seed", "42"))
    want("num_inference_steps", meta.get("num_inference_steps"), a.get("EVALUATION.num_inference_steps", "10"))
    want("configs", meta.get("configs"), labels(a.get("+PROBE.configs", "")))
    want("resample_noise", meta.get("resample_noise"), a.get("+PROBE.resample_noise", "0"))
    want("perturb", meta.get("perturb"), a.get("+PROBE.perturb", "none"))
    want("no_reseed", meta.get("no_reseed"), a.get("+PROBE.no_reseed", "0"))
    want("rng_probe", meta.get("rng_probe"), a.get("+PROBE.rng_probe", "0"))
    want("timing", meta.get("timing"), a.get("+PROBE.timing", "0"))
    want("warmup_episodes", meta.get("warmup_episodes"), a.get("+PROBE.warmup_episodes", "1"))
    want("first_episode", meta.get("first_episode", 0), a.get("+PROBE.first_episode", "0"))
    want("episode_noise", meta.get("episode_noise", 0), a.get("+PROBE.episode_noise", "0"))
    _dl = [int(x) for x in str(a.get("+PROBE.delay", "0")).split("/")]
    want("delay", meta.get("delay", [0, 0]), _dl * (2 if len(_dl) == 1 else 1))
    if "screen" in meta:
        sys.path.insert(0, str(ROOT / "probes"))
        import screen_engine as _SE
        want("screen", meta.get("screen"),
             [_SE.label(m, t) for m, t in _SE.parse_screen(a.get("+PROBE.screen", ""))])
    if "cudagraphs" in meta:
        want("cudagraphs", meta.get("cudagraphs"), call["env"].get("WAM_CUDAGRAPHS", "0"))
    want("checkpoint", pv.get("ckpt"), os.path.basename(a.get("ckpt", "")) or None)
    if rt:
        so = a.get("EVALUATION.sigma_shift")
        want("sigma_shift_override", rt.get("sigma_shift_override"), float(so) if so is not None else None)
        if so is None and rt.get("sigma_shift_override") is not None:
            diffs.append(f"sigma_shift_override: shipped {rt['sigma_shift_override']}, script leaves the default")
        want("model_class", rt.get("model_class"), MODEL_CLASS.get(a.get("task")))
        want("action_infer_mode", rt.get("action_infer_mode"), a.get("+EVALUATION.action_infer_mode", "first_frame"))
    cfgs = meta.get("configs") or []
    if len(cfgs) == 1 and cfgs[0] not in ("stock", "screen"):
        k = int(a.get("EVALUATION.num_inference_steps", "10"))
        exp_prefix = int(a.get("+PROBE.cache_prefix", -1))
        exp_prefix = k - 2 if exp_prefix < 0 else exp_prefix
        want("cache_prefix", meta.get("cache_prefix"), exp_prefix)
    return diffs


def compare_gate(call, d):
    a, diffs = call["args"], []
    if os.path.basename(a.get("ckpt", "")) != d.get("ckpt"):
        diffs.append(f"checkpoint: shipped {d.get('ckpt')}, script {os.path.basename(a.get('ckpt', ''))}")
    so = a.get("EVALUATION.sigma_shift")
    got = (d.get("runtime") or {}).get("sigma_shift_override")
    if (float(so) if so is not None else None) != got:
        diffs.append(f"sigma_shift_override: shipped {got}, script {so}")
    return diffs


def gated(experiment):
    src = (ROOT / "repro" / "experiments" / f"{experiment}.sh").read_text()
    return re.search(r'^GATE=(yes|"\$\{GATE:-yes\}")', src, re.M) is not None


def main(verbose=False):
    needed = analysis_inputs()
    plan = read_plan()
    writes = {}          # "dir/name" -> [(plan row, call or None)]
    with tempfile.TemporaryDirectory() as tmp:
        for row in plan:
            calls = dry_run(row, Path(tmp) / row["dir"])
            for name, call in calls.items():
                writes.setdefault(f"{row['dir']}/{name}", []).append((row, call))
            if gated(row["experiment"]):
                writes.setdefault(f"{row['dir']}/gate.json", []).append((row, None))
    problems, ok = [], 0
    for rel in needed:
        d = json.loads((RESULTS / rel).read_text())
        cands = writes.get(rel)
        if not cands:
            problems.append(f"{rel}: no plan line writes it"); continue
        errs = []
        for row, call in cands:
            if call is None:           # run_experiment.sh's own identity gate
                e = [] if d.get("ckpt") == "libero_optional_idm_2cam224.pt" and \
                    (d.get("runtime") or {}).get("sigma_shift_override") is None else ["not the default gate"]
            elif "meta" in d:
                e = compare_run(call, d["meta"])
            else:
                e = compare_gate(call, d)
            if not e:
                break
            errs.append(f"[{row['tier']} {row['experiment']}] " + "; ".join(e))
        else:
            problems.append(f"{rel}: " + " | ".join(errs)); continue
        ok += 1
        if verbose:
            print(f"  ok  {rel:62s} <- {row['tier']:9s} {row['experiment']} {row['seeds'][:40]}")
    print(f"{ok} of {len(needed)} runs the analysis reads are regenerated by repro/regeneration_plan.txt "
          f"with the settings recorded in the shipped files")
    for p in problems:
        print("  PROBLEM " + p)
    hours = {}
    for row in plan:
        hours[row["tier"]] = hours.get(row["tier"], 0) + row["hours"]
    print("  GPU hours by tier (one RTX A6000): " + ", ".join(f"{t} {h:.1f}" for t, h in hours.items()))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))

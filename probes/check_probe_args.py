"""Does every argument an experiment passes actually reach its probe?

An experiment can name the right output file, at the right seed, and still fail
on a fresh host because it passes a setting the probe never reads. That failure
is invisible here: hydra accepts any +PROBE.key without complaint and the probe
simply uses its default, so the run completes and produces a plausible artefact
collected under the wrong configuration. An environment variable the probe does
not read fails the same way and even more quietly.

This was not hypothetical. An earlier version of the pi0.5 experiment passed
PI_BETA to express the substitution weight for the sweep. probes/pi05_libero.py
has no such variable -- it reads the weight from a file -- so every run of that
sweep would have been collected at the same weight, and the five rows would have
differed only by noise while appearing to sweep a parameter.

So: extract what each experiment passes (via DRY_RUN), extract what each probe
reads, and report the difference. Keys the probe reads but the experiment does
not pass are fine -- that is what defaults are for. Keys the experiment passes
that the probe never reads are the bug.

usage: check_probe_args.py [--verbose]
"""
from __future__ import annotations

import glob, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# hydra keys consumed by the model/framework rather than the probe itself
FRAMEWORK = {"ckpt", "model", "EVALUATION", "seed", "hydra"}


def experiments():
    return sorted(os.path.basename(f)[:-3]
                  for f in glob.glob(os.path.join(ROOT, "repro", "experiments", "*.sh")))


def dry_run(exp):
    env = dict(os.environ, DRY_RUN="1")
    r = subprocess.run(["bash", os.path.join(ROOT, "repro", "run_experiment.sh"), exp],
                       capture_output=True, text=True, env=env, timeout=180)
    return r.stdout if r.returncode == 0 else ""


def probe_reads(path):
    """(hydra PROBE.* keys, environment variables) the probe actually consults."""
    try:
        src = open(path, encoding="utf-8").read()
    except OSError:
        return None, None
    # probes name their config object inconsistently -- pr, probe, cfg -- and a
    # detector that knows only one of them reports the others' arguments as
    # unknown. step_sweep.py uses `probe.out` and was flagged on that basis.
    hydra = set(re.findall(r"\b(?:pr|probe|cfg)\.([a-z_0-9]+)", src))
    hydra |= set(re.findall(r"PROBE\.([a-z_0-9]+)", src))
    hydra |= set(re.findall(r"cfg\.get\(\s*[\"']([a-z_0-9]+)", src))
    envs = set(re.findall(r"environ(?:\.get)?\(\s*[\"']([A-Z_0-9]+)", src))
    envs |= set(re.findall(r"getenv\(\s*[\"']([A-Z_0-9]+)", src))
    return hydra, envs


def main(verbose):
    bad = 0
    print("  experiment      probe                        unknown arguments")
    print("  " + "-" * 72)
    for exp in experiments():
        out = dry_run(exp)
        if not out:
            print(f"  {exp:15s} (dry run produced nothing)"); bad += 1; continue
        seen = set()
        for line in out.splitlines():
            # `weights` runs its probe after cd'ing into probes/, so the path in
            # the command has no directory part.
            m = re.search(r"probes/([a-z_0-9]+\.py)", line) or \
                re.search(r"python3? +([a-z_0-9]+\.py)", line)
            if not m: continue
            probe = "probes/" + os.path.basename(m.group(1))
            if (exp, probe) in seen: continue
            seen.add((exp, probe))
            hyd, env = probe_reads(os.path.join(HERE, os.path.basename(probe)))
            if hyd is None:
                print(f"  {exp:15s} {probe:28s} PROBE FILE NOT FOUND"); bad += 1; continue
            # everything this experiment passes to that probe, across all its lines
            passed_h, passed_e = set(), set()
            for l2 in out.splitlines():
                if os.path.basename(probe) not in l2: continue
                passed_h |= set(re.findall(r"\+PROBE\.([a-z_0-9]+)=", l2))
                passed_e |= set(re.findall(r"\b([A-Z][A-Z_0-9]{2,})=", l2))
            unknown_h = sorted(k for k in passed_h if k not in hyd and k not in FRAMEWORK)
            unknown_e = sorted(k for k in passed_e
                               if k not in env and not k.startswith(("PROBE", "DRY")))
            if unknown_h or unknown_e:
                bad += 1
                items = ", ".join([f"+PROBE.{k}" for k in unknown_h] + list(unknown_e))
                print(f"  {exp:15s} {probe:28s} {items}")
            else:
                print(f"  {exp:15s} {probe:28s} ok")
            if verbose:
                print(f"      probe reads PROBE.*: {sorted(hyd) or '-'}")
                print(f"      probe reads env:     {sorted(env) or '-'}")
    print()
    if bad:
        print(f"  {bad} experiment/probe pair(s) pass something the probe never reads.")
        print("  hydra accepts an unknown +PROBE key silently and the probe uses its")
        print("  default, so the run succeeds and the artefact records the wrong setting.")
    else:
        print("  every argument every experiment passes is one its probe reads")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))

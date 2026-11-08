"""Would a fresh host actually regenerate the artefacts this repository ships?

verify.sh answers a different question: it re-derives the published numbers from
the committed artefacts. That proves the analysis is faithful to the data. It
says nothing about whether the collection scripts, run on a new GPU host, would
produce that data again -- and an experiment whose defaults write different
filenames, or a different number of seeds, reproduces something adjacent to the
paper rather than the paper.

So this asks each experiment, via DRY_RUN, what files it would write, and checks
those against what is in results/. Three outcomes matter:

  orphan   a shipped artefact no experiment would regenerate. Its number is in
           the write-up and nothing in the repository can produce it again.
  phantom  an experiment output with no shipped counterpart. Either a result was
           never committed, or the defaults drifted from what was collected.
  match    the experiment would rewrite that artefact, under that name.

Filenames are the unit because that is what the analysis globs. An experiment
that writes atten_libero_goal_seed7.json cannot reproduce a table built from
q_object_seed7.json even if the science underneath is identical.

usage: check_repro_map.py [--verbose]
"""
from __future__ import annotations

import glob, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")

# Artefacts that are inputs or gates rather than experiment outputs.
IGNORE = re.compile(r"gate.*\.json$|canonical_numbers\.json$|families_summary\.json$")


def experiments():
    return sorted(os.path.basename(f)[:-3]
                  for f in glob.glob(os.path.join(ROOT, "repro", "experiments", "*.sh")))


def is_pending(exp):
    """Has this experiment declared itself written-but-not-yet-collected?"""
    p = os.path.join(ROOT, "repro", "experiments", exp + ".sh")
    try: src = open(p, encoding="utf-8", errors="replace").read()
    except OSError: return False
    return re.search(r"^PENDING=yes\b", src, re.M) is not None


def param_sets(exp):
    """Environment settings under which this experiment has shipped artefacts.

    An experiment can be parameterised -- baselines.sh reads WAM_SEED and names
    its outputs with it -- and then its DEFAULT dry run declares only one of the
    collections it produced. Without this, every seed-7 artefact read as "cited
    but nothing would regenerate it", which is the opposite of true: the script
    regenerates them exactly, given the parameter it was run with.

    The experiment declares them, so the list is committed rather than inferred.
    """
    p = os.path.join(ROOT, "repro", "experiments", exp + ".sh")
    try:
        src = open(p, encoding="utf-8").read()
    except OSError:
        return [{}]
    m = re.search(r'^PARAM_SETS="([^"]*)"', src, re.M)
    if not m:
        return [{}]
    sets = [{}]
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            sets.append({k: v})
    return sets


def would_write(exp):
    """Filenames this experiment could produce, over every declared parameter set."""
    out, err = set(), None
    for extra in param_sets(exp):
        got, e = _would_write_once(exp, extra)
        if got is None:
            err = e
        else:
            out |= got
    return (out, None) if out else (None, err or "no outputs declared")


def _would_write_once(exp, extra):
    env = dict(os.environ, DRY_RUN="1", **extra)
    try:
        r = subprocess.run(["bash", os.path.join(ROOT, "repro", "run_experiment.sh"), exp],
                           capture_output=True, text=True, env=env, timeout=120)
    except Exception as e:
        return None, f"dry run failed: {e}"
    if r.returncode != 0:
        return None, f"dry run exited {r.returncode}: {(r.stderr or '').strip()[:120]}"
    out = set()
    for m in re.finditer(r"(?:\+PROBE\.out=|PI_OUT=|MERGED_OUT=)(\S+)", r.stdout):
        out.add(os.path.basename(m.group(1).strip("'\"")))
    # A shard that a post-collection merge consumes is not an output the
    # repository should ship; the merged file is. The experiment declares them.
    for m in re.finditer(r"INTERMEDIATE=(\S+)", r.stdout):
        out.discard(os.path.basename(m.group(1).strip("'\"")))
    # Experiments that write through a different mechanism name their target in
    # the dry-run text; catch the common "-> path.json" and "dst=" forms too.
    for m in re.finditer(r"(\S+\.json)\b", r.stdout):
        b = os.path.basename(m.group(1))
        if b.startswith(("shrink_",)): out.add(b)
    return out, None


def missing_inputs():
    """Files an experiment feeds a probe that this repository does not contain.

    The output audit cannot see these. An experiment can write exactly the right
    artefact under exactly the right name and still be unreproducible because an
    INPUT it reads was never committed -- the run succeeds, the probe silently
    falls back to a default, and the artefact comes out missing an arm.

    That is not hypothetical: placement passes +PROBE.oracle pointing at a cell
    set that no setup script creates and no probe generates. The shipped artefact
    has five placements, a fresh collection has four, and report_placement.py
    used to raise KeyError on the difference.

    Paths under $REPO must exist here. Paths under /workspace are provided by a
    setup script, so they are checked against what those scripts fetch.
    """
    env = dict(os.environ, DRY_RUN="1")
    provided = ""
    for f in glob.glob(os.path.join(ROOT, "setup", "*.sh")):
        try: provided += open(f, encoding="utf-8", errors="ignore").read()
        except OSError: pass
    out = []
    for exp in experiments():
        try:
            r = subprocess.run(["bash", os.path.join(ROOT, "repro", "run_experiment.sh"), exp],
                               capture_output=True, text=True, env=env, timeout=120)
        except Exception:
            continue
        for m in re.finditer(r"\+PROBE\.(manifest|shrink|oracle|video_root)=(\S+)", r.stdout):
            key, path = m.group(1), m.group(2).strip("'\"")
            if path.startswith(ROOT) or "/inputs/" in path or "/results/" in path:
                real = path if os.path.isabs(path) else os.path.join(ROOT, path)
                if not os.path.exists(real):
                    out.append((exp, key, os.path.relpath(real, ROOT)))
            elif path.startswith("/workspace") or path.startswith("/w"):
                base = os.path.basename(path)
                if base not in provided and path not in provided:
                    out.append((exp, key, path + "  (no setup script provides it)"))
    return out


def shipped():
    got = {}
    for f in glob.glob(os.path.join(RESULTS, "**", "*.*"), recursive=True):
        b = os.path.basename(f)
        if IGNORE.search(b) or not b.endswith((".json", ".npz")): continue
        got.setdefault(b, []).append(os.path.relpath(f, RESULTS))
    # weights are inputs, produced by the `weights` experiment
    for f in glob.glob(os.path.join(ROOT, "inputs", "weights", "*.json")):
        got.setdefault(os.path.basename(f), []).append(
            os.path.relpath(f, ROOT))
    return got


# The write-up lives outside the repository, so a reviewer will not have it and
# the path must not be baked in -- an absolute path under a home directory also
# carries the account name into a double-blind submission, which is how this
# file failed the anonymity check the first time it was written. Point WRITEUP
# at it to include it in the citation test; without it the test falls back to
# verify.sh and canonical_numbers.py, which is enough to classify every artefact
# the repository itself refers to.
ARTIFACT = os.environ.get("WRITEUP", "")


def is_cited(name, subdir=None):
    """Does any number from this artefact reach the write-up?

    Membership in canonical_numbers.json is not enough: several measurements are
    computed and kept without being quoted. What matters is whether the write-up
    would be wrong if the artefact could not be produced again.

    A directory-level rule was tried here and reverted: counting everything in a
    globbed directory as cited turned eight genuine archive files into false
    positives. The seed-suffixed artefacts that motivated it are handled properly
    instead, by having their experiment DECLARE the parameter sets it shipped
    (PARAM_SETS), so they are claimed rather than merely cited.
    """
    stem = name.rsplit(".", 1)[0]
    srcs = []
    for src in (os.path.join(ROOT, "repro", "verify.sh"),
                os.path.join(HERE, "canonical_numbers.py"), ARTIFACT):
        if not src: continue
        try:
            srcs.append(open(src, encoding="utf-8", errors="ignore").read())
        except OSError:
            continue
    for txt in srcs:
        if stem in txt: return True
    return False


def main(verbose):
    have = shipped()
    claimed, problems, pending = {}, [], []
    print("  experiment      would write   of which shipped")
    print("  " + "-" * 62)
    for exp in experiments():
        files, err = would_write(exp)
        if err:
            problems.append(f"{exp}: {err}"); print(f"  {exp:15s} {err}"); continue
        hit = sorted(f for f in files if f in have)
        miss = sorted(f for f in files if f not in have)
        for f in hit: claimed.setdefault(f, []).append(exp)
        print(f"  {exp:15s} {len(files):11d}   {len(hit)}")
        if miss:
            # An experiment declared PENDING has been written and validated but
            # not yet collected, so nothing is shipped to compare against. That
            # is a different state from an experiment that writes files nothing
            # ships, and burying one in the other is how a real phantom hides.
            # The declaration must be removed when the artefacts land, and the
            # summary below says so.
            (pending if is_pending(exp) else problems).append(
                f"{exp}: writes {len(miss)} file(s) that are not shipped yet: "
                + ", ".join(miss[:4]))
            if verbose:
                for f in miss: print(f"      phantom: {f}")

    # Not every unclaimed artefact is a problem, and reporting them together
    # would bury the ones that are. Three kinds are accounted for:
    #   VARIANT  produced by an experiment under an environment switch rather
    #            than its defaults -- mechanism collects one backbone per run.
    #   DERIVED  an analysis output, not a collected measurement. verify.sh
    #            regenerates these from the .npz files by running the probe.
    #   ARCHIVE  shipped but quoted nowhere: kept as evidence, not as a result.
    # What is left is the real answer: a cited artefact nothing can produce.
    variant = {}
    for exp, flag, names in (
            ("mechanism", "BACKBONE=pi05", ["pi05_families.npz", "pi05_families_repro.npz"]),
            ("mechanism", "BACKBONE=dz",   ["dz_families.npz", "dz_families_repro.npz"]),
            # The tag now selects the deployment checkpoint and episodes24_idm.json
            # itself, so this is a plain invocation rather than a described one.
            ("mechanism", "--seeds idm", ["fw_families_idm.npz"]),
            ("mechanism", "--seeds idm7", ["fw_families_idm_taps7.npz"]),
            ("mechanism", "--seeds repeat16", ["t1_seed0.npz"])):
        for n in names: variant[n] = f"{exp} {flag}"
    derived = {"beta_ceiling.json": "probes/beta_ceiling.py (verify.sh beta_ceiling)",
               "kalman_beta.json":  "probes/kalman_beta.py (verify.sh kalman)",
               "online_beta.json":  "probes/online_beta.py (verify.sh online)",
               "families_summary.json": "probes/report_families.py (verify.sh families)",
               "families_final.json": "probes/report_families.py (geometry + gains; "
                                      "the rows table is families_summary.json)",
               "alpha_constraint.json": "probes/alpha_constraint.py (verify.sh alpha)"}

    unclaimed = [b for b in have if b not in claimed]
    var  = sorted(b for b in unclaimed if b in variant)
    der  = sorted(b for b in unclaimed if b in derived)
    rest = sorted(b for b in unclaimed if b not in variant and b not in derived)
    cited = [b for b in rest if is_cited(b)]
    archive = [b for b in rest if b not in cited]

    print()
    if var:
        print(f"  reachable with a switch, not by defaults ({len(var)}):")
        for b in var: print(f"    {b:34s} {variant[b]}")
    if der:
        print(f"\n  analysis outputs, regenerated by verify.sh ({len(der)}):")
        for b in der: print(f"    {b:34s} {derived[b]}")
    if archive:
        print(f"\n  shipped but quoted nowhere -- evidence, not results ({len(archive)}):")
        for b in archive: print(f"    {b:34s} {have[b][0]}")
    orphans = cited
    if orphans:
        print(f"\n  NOT REPRODUCIBLE -- cited artefacts nothing would regenerate ({len(orphans)}):")
        for b in orphans: print(f"    {b:34s} {have[b][0]}")
    else:
        print("\n  every cited artefact is regenerated by an experiment")

    if pending:
        print("\n  PENDING -- declared, validated, not yet collected:")
        for p in pending: print(f"    {p}")
        print("  Remove PENDING=yes from the experiment once its artefacts are")
        print("  committed, or this stops distinguishing 'not run' from 'wrong'.")

    if problems:
        print("\n  PHANTOMS -- experiment outputs with nothing shipped to compare:")
        for p in problems: print(f"    {p}")

    mi = missing_inputs()
    if mi:
        print("\n  INPUTS an experiment reads that this repository does not contain:")
        for exp, key, path in mi:
            print(f"    {exp:12s} +PROBE.{key:11s} {path}")
        print("  A run still succeeds -- the probe falls back to a default and the")
        print("  artefact comes out missing whatever that input supplied.")

    print(f"\n  {len(claimed)} of {len(have)} shipped artefacts are claimed by an experiment")
    return 1 if (orphans or problems or mi) else 0


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))

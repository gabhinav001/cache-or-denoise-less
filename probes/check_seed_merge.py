"""Could a probe silently merge two seeds into one collection?

This has happened twice. Seed variants are shipped beside the seed-42 artefacts
under a _seedN suffix, and every closed-loop probe keys its arms by initial
state. The seeds visit the SAME states, so a probe that globs its directory
without filtering loads both files into one dictionary and the later one
overwrites the earlier. The output is a mixture reported as a single collection,
with no error and no missing file:

  deployfrontier  seed variants first shipped in per-seed subdirectories, where
                  check_repro_map -- which keys artefacts by basename -- could
                  not see them at all.
  statescale      report_statescale.load() globbed the directory and reported
                  2st at 97.2% where the seed-42 value is 96.6%. canonical had
                  already regenerated with the mixture.

Neither was caught by a number check, because the numbers were internally
consistent; they were simply the wrong numbers. So this checks the structure
instead: for every results directory holding seed variants, every probe that
globs it must show evidence of filtering by seed.

This is a static check and it is deliberately noisy in one direction: it flags a
glob it cannot prove is seed-aware, rather than staying silent about one it
cannot prove is unsafe. A probe that genuinely handles seeds passes by naming
them.
"""
from __future__ import annotations

import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
# Evidence of FILTERING, not merely of the string "_seed" appearing. The first
# version of this accepted any mention, and report_matrix.py passed it while
# merging: it contains rsplit("_seed", 1), which STRIPS the suffix so that a
# seed-7 file is treated as its seed-42 stem -- the precise mechanism of the bug,
# scored as protection against it. Stripping is now evidence of nothing.
AWARE = re.compile(
    r"seeds_present"                    # the shared helper
    r"|_seed%d|_seed\{|_seed\[0-9\]"    # a glob pattern selecting one seed
    r"|seed\s*=\s*42"                  # an explicit default-seed parameter
    r"|[\"']_seed[\"']\s+in\b"          # an exclusion test: if "_seed" in b: continue
    r"|meta.*\[.seed.\]\s*==|get\(.seed.\)\s*==",  # filtering on recorded seed
    re.I)

# A probe can only MERGE seeds if it accumulates measurements keyed by initial
# state. One that enumerates files for provenance, or summarises each separately,
# has nothing to overwrite. This is the real discriminator and it is checked
# directly rather than inferred from the glob.
KEYS_BY_STATE = re.compile(r"init_state")

# Globbing a mixed-seed directory is only destructive when the probe ACCUMULATES
# across files into a structure keyed by initial state. A probe that analyses one
# file at a time and keys its output by filename cannot merge anything, and that
# distinction is not reliably detectable from the source. These were checked by
# reading them and by confirming their output separates the seeds; the reason is
# recorded so the exemption can be re-checked rather than inherited.
PER_FILE_SAFE = {
    "report_cachesteps_ci.py": (
        "analyse() takes a single path and main() keys results by filename, so "
        "each artefact is summarised on its own; output lists cs4 (seed 42) and "
        "cs4_seed7 as separate rows"),
}


def mixed_seed_dirs():
    """results subdirs holding both unsuffixed and _seedN artefacts."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        if not os.path.isdir(d): continue
        files = glob.glob(os.path.join(d, "*.json"))
        suffixed = [f for f in files if re.search(r"_seed\d+\.json$", f)]
        plain = [f for f in files if not re.search(r"_seed\d+\.json$", f)
                 and os.path.basename(f) != "gate.json"]
        if not suffixed or not plain: continue
        seeds = set()
        for f in files:
            try: seeds.add((json.load(open(f)).get("meta") or {}).get("seed"))
            except Exception: pass
        seeds.discard(None)
        # without two RECORDED seeds this is not a mixed-seed directory at all;
        # closedloop and pi05 carry _seedN filenames but no recorded seed, and
        # flagging them buried the real cases in noise
        if len(seeds) < 2: continue
        # do the two groups actually share initial states? that is what makes a
        # blind merge destructive rather than merely untidy
        def states(fs):
            s = set()
            for f in fs:
                try:
                    for r in json.load(open(f))["rows"]: s.add(r.get("init_state"))
                except Exception: pass
            return s
        shared = states(suffixed) & states(plain)
        out[os.path.basename(d)] = {"seeds": sorted(seeds),
                                    "shared_states": len(shared)}
    return out


def main():
    dirs = mixed_seed_dirs()
    if not dirs:
        print("  no results directory holds seed variants -- nothing to merge")
        return 0
    print("  directories holding seed variants:")
    for d, info in dirs.items():
        print(f"    results/{d}: seeds {info['seeds']}, "
              f"{info['shared_states']} initial states shared between them")

    bad = []
    for probe in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        name = os.path.basename(probe)
        if name.startswith("check_"): continue
        try: src = open(probe, encoding="utf-8", errors="replace").read()
        except OSError: continue
        for d in dirs:
            # does this probe glob that directory at all?
            if not re.search(rf'["\']{re.escape(d)}["\']|{re.escape(d)}[/"\']', src):
                continue
            if not KEYS_BY_STATE.search(src): continue
            if AWARE.search(src): continue
            if name in PER_FILE_SAFE: continue
            bad.append((name, d))

    if bad:
        print(f"\n  {len(bad)} probe(s) glob a mixed-seed directory with no sign of "
              f"filtering by seed:")
        for n, d in bad:
            print(f"    {n} reads results/{d}")
        print("\n  A blind glob there merges the seeds: same states, same arm keys,")
        print("  later file wins, and the mixture reports as one collection. Take")
        print("  the seed explicitly, as report_statescale.load() now does.")
        return 1
    print("\n  every probe reading a mixed-seed directory either filters by seed")
    print("  or summarises one artefact at a time:")
    for n, why in sorted(PER_FILE_SAFE.items()):
        print(f"    {n}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

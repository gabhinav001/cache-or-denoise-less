"""Is any run counted twice in a pooled set -- or silently reused as a new one?

Two different failures, one test.

Pooling the same run twice inflates both the sample size and the significance,
and nothing in the output reveals it -- the p-value simply looks better. It
happened here: a re-run of seed 7, kept for comparison, was left in the directory
the frontier figures were pooled from, so five seeds were reported as six.

The second failure is subtler and cost more. A directory of "re-runs under the
fixed harness" contained two seeds byte-for-byte identical to the pre-fix data
they were supposed to supersede -- boxes provisioned before the fix landed, since
the driver rsyncs the repo at launch and a commit afterwards does not reach a
machine already running. The numbers were quoted as evidence that the fix changed
nothing, when in fact those runs never exercised it. Comparing within a directory
cannot see this: each set is internally distinct, and only the comparison
*between* directories shows the reuse.

Two artefacts are the same run if every (task, episode, config) has the same
outcome. That is stronger than comparing filenames, and it catches the case this
was: a deliberate repeat of a deterministic configuration, which produces a
byte-different file with identical contents. Step counts are reported alongside
because they are far more sensitive than a binary success -- two genuinely
independent runs of the same seed on different hardware agree on most outcomes
but almost never on the step count of every episode.

usage: check_duplicates.py <dir> [<dir> ...]
"""
from __future__ import annotations

import glob, hashlib, json, os, sys
from collections import defaultdict


def _rows(path):
    try:
        j = json.load(open(path))
    except Exception:
        return None
    rows = j.get("rows") if isinstance(j, dict) else None
    if not rows or not isinstance(rows[0], dict) or "task" not in rows[0]:
        return None
    return rows


def fingerprint(path):
    rows = _rows(path)
    if rows is None: return None
    key = sorted((r["task"], r["episode"], r["config"], r["success"]) for r in rows)
    return hashlib.md5(str(key).encode()).hexdigest()


def cells(path):
    """(task, episode, config) -> (success, steps), the comparable unit."""
    rows = _rows(path)
    if rows is None: return None
    return {(r["task"], r["episode"], r["config"]): (r["success"], r.get("steps"))
            for r in rows}


def overlap_agreement(A, B):
    """(n_shared, success agreement, step agreement) over the shared cells.

    Compared on the overlap rather than the whole file, because the case worth
    catching is a run still in flight -- 250 rows against the original's 500 --
    whose rows are nonetheless the identical computation. A whole-file
    fingerprint cannot see that: the row sets differ, so the hashes differ, and
    the reused run passes as new.
    """
    common = set(A) & set(B)
    if not common: return 0, None, None
    succ = sum(1 for k in common if A[k][0] == B[k][0]) / len(common)
    step = sum(1 for k in common if A[k][1] == B[k][1]) / len(common)
    return len(common), succ, step


# Experiments whose runs are COMPARED rather than pooled. This check exists to
# stop the same run being counted twice inside one pooled figure; where an
# experiment's whole design is to run the same configuration twice under one
# changed variable, identical outcomes are the RESULT, not a defect. rngdrift is
# exactly that: reseed on against reseed off, and the finding is that they match
# bit for bit. Nothing in it is ever pooled -- report_rngdrift.py compares the
# files pairwise -- so the hazard this guards against cannot arise.
#
# Narrow by construction: an experiment earns a place here only by never pooling,
# and the check still reports what it found so a reader sees the agreement.
# stocktime is the same shape: sfirst* and ffirst* are one configuration at one
# seed, differing ONLY in which arm runs first, and report_stocktime.py compares
# them pairwise to separate arm position from recording overhead. Their outcomes
# being identical is the point -- it says arm order moves wall clock and not
# results, which is a cleaner statement of A8 than anything in the frontier data.
# Nothing in stocktime is pooled.
# frontier_timing is a TIMING collection: its success column is never used for
# any figure (success comes from baselines.sh, per seed). Its lat4/lat6/lat8 runs
# are full compute at 4, 6 and 8 denoising steps and come back with identical
# outcomes -- which is expected, not a bug: 30 episodes at ~97% success cannot
# separate them, while their timings differ by 100 ms (203/255/307), proving the
# step override took effect. Nothing here is pooled.
# stockquiet is stocktime's design on a quiet host: q_sfirst and q_ffirst are one
# configuration differing ONLY in which arm runs first. Their outcomes come back
# identical -- 30/30 in both -- which is a second independent confirmation that
# arm order moves wall clock and not results, and it is why both orders can be
# averaged to cancel the position term. Nothing here is pooled.
# deployfrontier is a LATENCY collection at 30 episodes per arm, and its runs are
# one arm each at a different denoising-step count. Two of them come back with
# identical outcomes -- d3 and d8 both fail on episodes (2,3) and (2,6) -- which
# is what a hard episode does, not a duplicated run: their latencies differ by
# 1.8x (179.9 against 331.7 ms), so the step override plainly took effect. Its
# success column is never pooled; success comes from baselines.sh and
# cachesteps.sh at 100-200 episodes per arm.
# matched100_a6000 holds one run per arm, compared pairwise across arms and
# machines (report_budget_shift.py) and never pooled. Its causal_uncond_s1_stock2
# and causal_uncond_s5_stock2 are two uncached steps at shifts 1 and 5: both
# succeed on 47 of 50 states and fail on the same three, while their step counts
# differ in 30 of the 50 episodes, so the shift plainly took effect.
COMPARED_NOT_POOLED = {"rngdrift", "stocktime", "frontier_timing", "stockquiet",
                       "deployfrontier", "matched100_a6000"}


def main(dirs):
    bad = 0
    everywhere = defaultdict(list)   # fingerprint -> [relative path, ...]

    for d in dirs:
        files = [f for f in sorted(glob.glob(os.path.join(d, "*.json")))
                 if "gate" not in os.path.basename(f)]
        sig = defaultdict(list)
        for f in files:
            h = fingerprint(f)
            if h:
                sig[h].append(os.path.basename(f))
                everywhere[h].append(f)
        dupes = {k: v for k, v in sig.items() if len(v) > 1}
        n = sum(len(v) for v in sig.values())
        compared = os.path.basename(d) in COMPARED_NOT_POOLED
        if dupes and compared:
            print(f"  {os.path.basename(d)}: {n} runs, {len(dupes)} identical pair(s) "
                  f"-- compared, never pooled, so this is a result not a duplicate")
            for v in dupes.values():
                print(f"    identical outcomes: {', '.join(v)}")
        elif dupes:
            bad += 1
            print(f"  {os.path.basename(d)}: {n} runs, DUPLICATES FOUND")
            for v in dupes.values():
                print(f"    identical outcomes: {', '.join(v)}")
        else:
            print(f"  {os.path.basename(d)}: {n} runs, all distinct")

    # The same run present in two directories is not automatically wrong -- an
    # archived copy of a superseded set is deliberate. It is reported rather than
    # failed, because whether it is intended depends on what the directory claims
    # to be, and only the reader knows that.
    #
    # Every cross-directory pair is compared on its overlap, not by hash, so a
    # partial run reproducing a complete one is caught as well as an exact copy.
    loaded = {}
    for paths in everywhere.values():
        for p in paths:
            if p not in loaded:
                c = cells(p)
                if c: loaded[p] = c
    rel = lambda p: os.path.join(os.path.basename(os.path.dirname(p)),
                                 os.path.basename(p))
    same, shared_arms = [], 0
    names = sorted(loaded)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if os.path.dirname(a) == os.path.dirname(b): continue
            n, succ, step = overlap_agreement(loaded[a], loaded[b])
            # Identical outcomes AND identical step counts over a real overlap.
            # Success alone is too coarse: independent runs of the same seed on
            # different hardware often agree on most successes and essentially
            # never on every step count.
            if n < 50 or succ != 1.0 or step != 1.0: continue
            # Two *different* experiments that share an arm agree episode for
            # episode, because the harness is deterministic given the seed --
            # expected, and far too common to list. The reused-run case looks
            # different: the same filename in two directories, one claiming to
            # supersede the other.
            if os.path.basename(a) == os.path.basename(b):
                same.append((rel(a), rel(b), n))
            else:
                shared_arms += 1
    if same:
        print("\n  Same run under the same name in two directories:")
        for a, b, n in same:
            print(f"    {a} == {b}  ({n} shared episodes, identical successes and steps)")
        print("  A copy kept as an archive is fine. A directory of fresh re-runs is not:")
        print("  identical outcomes mean that run did not exercise whatever changed.")
    if shared_arms:
        print(f"\n  ({shared_arms} further cross-directory agreements between differently"
              f" named runs -- expected: a shared arm is deterministic given the seed.)")

    if bad:
        print(f"\n  {bad} director{'y' if bad==1 else 'ies'} pool the same run more than once.")
        print("  Remove the duplicate before quoting any pooled figure from it.")
        return 1
    return 0


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "*")))
    sys.exit(main([d for d in dirs if os.path.isdir(d)]))

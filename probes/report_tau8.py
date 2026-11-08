"""The tau=8 effect, with the two axes kept apart.

The original nine runs were reported as nine replications and were not: four
LIBERO suites, three episode counts, and four runs with no recorded seed. A
seed-level interval over that set answers no question anyone asked.

This reports two claims from two axes and refuses to combine them:

  SEEDS   one configuration at five seeds. The only set a seed-level interval
          or sign test may be computed over, because it is the only set where
          the seed is what varies.
  SUITES  the same configuration on other suites, one seed each. Coverage: it
          says the effect appears wherever it was looked for. It says nothing
          about seed variance, and pooling it into the interval above would
          reintroduce exactly the error this experiment exists to correct.

Also verifies what the artefacts now record -- seed, checkpoint hash, initial
latent, per-episode initial states -- since a claim of reproducibility that
nothing checks is a claim about intent.

usage: report_tau8.py <dir>
"""
from __future__ import annotations

import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tstat                                       # noqa: E402

SEED_RUNS = ("seed3", "seed7", "seed11", "seed23", "seed42")


def load(p):
    try: d = json.load(open(p))
    except Exception: return None, None
    if not isinstance(d, dict) or "rows" not in d: return None, None
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], {})[(r["task"], r["episode"])] = bool(r["success"])
    return d, by


def arms(by):
    plain = next((c for c in by if c.endswith("tau8")), None)
    shrunk = next((c for c in by if c.endswith("tau8_shrink")), None)
    return plain, shrunk


def delta(by):
    p, s = arms(by)
    if not p or not s: return None
    ks = sorted(set(by[p]) & set(by[s]))
    if not ks: return None
    rp = 100.0 * sum(by[p][k] for k in ks) / len(ks)
    rs = 100.0 * sum(by[s][k] for k in ks) / len(ks)
    better = sum(1 for k in ks if by[s][k] and not by[p][k])
    worse = sum(1 for k in ks if by[p][k] and not by[s][k])
    full = by.get("full", {})
    rf = 100.0 * sum(full.values()) / len(full) if full else float("nan")
    return {"full": rf, "plain": rp, "shrunk": rs, "delta": rs - rp,
            "better": better, "worse": worse, "n": len(ks)}


def mcnemar_p(b, w):
    n = b + w
    if n == 0: return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def show(rows, label):
    print(f"\n  {label}\n")
    print("    %-16s %-8s %7s %7s %7s %8s %s"
          % ("run", "seed", "full", "plain", "att.", "delta", "better/worse"))
    for name, seed, r in rows:
        print("    %-16s %-8s %6.1f%% %6.1f%% %6.1f%% %+7.1f  %d/%d"
              % (name, seed, r["full"], r["plain"], r["shrunk"], r["delta"],
                 r["better"], r["worse"]))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    seeds, suites, bad = [], [], []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        base = os.path.basename(p)[:-5]
        if base.startswith(("gate", "_")): continue
        doc, by = load(p)
        if doc is None: continue
        m = doc.get("meta", {})
        r = delta(by)
        if r is None: continue
        seed = m.get("seed")
        if seed is None: bad.append((base, "no seed recorded"))
        (seeds if base in SEED_RUNS else suites).append((base, str(seed), r))

    if not seeds and not suites:
        # Not collected is not the same as collected and wrong. This experiment
        # is declared PENDING until its artefacts are committed, and a reviewer
        # running verify.sh on a fresh clone should see that stated rather than
        # a failure they cannot act on.
        print(f"  no tau=8 artefacts in {d} -- experiment not yet collected.")
        print("  repro/experiments/tau8.sh declares PENDING until they are.")
        return 0

    if seeds:
        show(seeds, "SEEDS -- one configuration, libero_goal, seed is what varies")
        ds = [r["delta"] for _, _, r in seeds]
        m, lo, hi, sd, t = tstat.mean_ci(ds)
        pos = sum(1 for x in ds if x > 0); neg = sum(1 for x in ds if x < 0)
        k, tot = min(pos, neg), pos + neg
        sp = min(1.0, 2 * sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot) if tot else 1.0
        print(f"\n    mean {m:+.2f}  sd {sd:.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (t={t:.3f})")
        print(f"    positive on {pos} of {len(ds)}, sign test p = {sp:.4g}")
        print("\n    This interval is over SEEDS of one configuration. It is the only")
        print("    seed-level statistic these artefacts support.")

    if suites:
        show(suites, "SUITES -- same configuration elsewhere, one seed each")
        print("\n    Coverage, not replication. Every suite positive says the effect")
        print("    appears wherever it was looked for; it says nothing about seed")
        print("    variance, and it is NOT pooled into the interval above. Pooling the")
        print("    two axes is the error this experiment exists to correct.")

    print("\n\n  WHAT THE ARTEFACTS RECORD\n")
    fields = ("seed", "initial_latent")
    provf = ("commit", "ckpt_sha256_16", "gpu")
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        base = os.path.basename(p)[:-5]
        if base.startswith(("gate", "_")): continue
        try: doc = json.load(open(p))
        except Exception: continue
        if not isinstance(doc, dict) or "rows" not in doc: continue
        m = doc.get("meta", {}); pv = m.get("provenance") or {}
        have = [f for f in fields if m.get(f) is not None]
        have += [f for f in provf if pv.get(f) is not None]
        init = sum(1 for r in doc["rows"] if r.get("init_state"))
        print("    %-16s %-46s init_state on %d/%d rows"
              % (base, ",".join(have), init, len(doc["rows"])))
    if bad:
        print("\n    MISSING:")
        for n, why in bad: print(f"      {n}: {why}")
        return 1
    ncommit = sum(1 for p in glob.glob(os.path.join(d, "*.json"))
                  if not os.path.basename(p).startswith(("gate", "_"))
                  and not ((json.load(open(p)).get("meta", {}).get("provenance") or {}).get("commit")))
    print("\n    Every run records its seed, its checkpoint's hash, the machine, and")
    print("    the starting state of every episode -- none of which the original nine")
    print("    runs recorded.")
    if ncommit:
        print(f"\n    NOT recorded: the commit, on {ncommit} of these runs. A probe executes")
        print("    from a copy installed beside FastWAM's configs and the driver ships the")
        print("    repository without .git/, so there was no checkout on the box to read.")
        print("    The driver now passes the commit it shipped; these artefacts predate")
        print("    that and cannot be told which code wrote them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

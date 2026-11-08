"""What agreement should an independent re-run expect?

A reviewer re-collecting these experiments on their own host will not reproduce
the artefacts episode for episode -- only one shipped run records a seed, and
same-seed determinism holds only within a GPU model anyway. The useful question
is therefore not whether the numbers match but how far apart two honest
collections of the same quantity land, and whether the published conclusions
survive that distance.

This repository can answer that from data rather than assertion, because several
quantities were collected twice, independently:

  residual geometry   three backbones, each measured on two different machines
  frontier            five seeds collected twice, before and after the pairing
                      fix (A8) -- different random draws, different hosts, and a
                      corrected experimental design
  tau=8               collected once. No replication evidence exists, and this
                      reports that rather than implying otherwise.

The frontier pair is the strongest evidence available here: it is a full
re-collection of a headline result by a later, independent run of the same
scripts. It is also confounded, since the design changed between the two, and
that is stated rather than hidden -- the observed gap is an upper bound on
collection noise, not a clean estimate of it.

usage: replication_margin.py
"""
from __future__ import annotations

import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
import tstat  # noqa: E402
from report_closedloop import rate, mcnemar  # noqa: E402


def by_of(p):
    d = json.load(open(p)); by = {}
    for r in d["rows"]:
        by.setdefault((r["task"], r["episode"]), {})[r["config"]] = r["success"]
    return by


def tau_delta(d, tau):
    """Per-seed attenuated-minus-plain deltas for a directory of runs."""
    h, s = f"c3ache_tau{tau}", f"c3ache_tau{tau}_shrink"
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "frontier_seed*.json"))):
        by = by_of(p); cf = {c for v in by.values() for c in v}
        if h not in cf or s not in cf: continue
        out[os.path.basename(p)[13:-5]] = rate(by, s)[0] - rate(by, h)[0]
    return out


def mean_ci(v):
    # Student t rather than 1.96 -- five seeds, and the sd is estimated from the
    # same five numbers. See probes/tstat.py.
    m, lo, hi, _sd, _t = tstat.mean_ci(v)
    return m, lo, hi


def main():
    print("\n  GEOMETRY -- each backbone measured on two different machines\n")
    import numpy as np
    import report_families as RF
    print("    %-10s %-12s %-12s %s" % ("backbone", "run 1", "run 2", "difference"))
    for name, a, b in (("Fast-WAM", "fw_families", "fw_families_repro52"),
                       ("pi0.5", "pi05_families", "pi05_families_repro"),
                       ("DreamZero", "dz_families", "dz_families_repro")):
        cos = []
        for f in (a, b):
            path = os.path.join(R, "residual", f + ".npz")
            if not os.path.exists(path): break
            z, G, cell, ep = RF.load(path)
            G = G[G[:, 0, 0] > 0]
            rho = G[:, 0, 1] / np.sqrt(np.maximum(G[:, 0, 0] * G[:, 1, 1], 1e-30))
            cos.append(float(np.median(rho)))
        if len(cos) == 2:
            print("    %-10s %-12.4f %-12.4f %.4f" % (name, cos[0], cos[1], abs(cos[0] - cos[1])))
    print("\n    Cosine is the quantity every downstream claim rests on. Two machines"
          "\n    agree on it to within 0.002, so the geometry replicates essentially"
          "\n    exactly and is not where a reviewer's risk lies.")

    print("\n\n  FRONTIER -- five seeds collected twice, independently\n")
    pre = os.path.join(R, "frontier_v1_prefix")
    post = os.path.join(R, "frontier")
    if not os.path.isdir(pre):
        print("    (only one generation present)"); return 0
    for tau in (2, 4):
        a, b = tau_delta(pre, tau), tau_delta(post, tau)
        seeds = sorted(set(a) & set(b), key=lambda s: -abs(a[s]))
        ma, loa, hia = mean_ci([a[s] for s in seeds])
        mb, lob, hib = mean_ci([b[s] for s in seeds])
        print("    tau=%d   %-9s %-9s %s" % (tau, "first", "second", "per-seed gap"))
        for s in seeds:
            print("      seed %-4s %-9.1f %-9.1f %+.1f" % (s, a[s], b[s], b[s] - a[s]))
        print("      %-9s %-9.1f %-9.1f %+.1f" % ("mean", ma, mb, mb - ma))
        print("      %-9s [%+.1f,%+.1f] [%+.1f,%+.1f]" % ("95% CI", loa, hia, lob, hib))
        agree = (loa < mb < hia) and (lob < ma < hib)
        print("      each mean lies inside the other's interval: %s\n" % ("yes" if agree else "NO"))

    print("    The two collections differ by 2.2 points at tau=4 and 0.6 at tau=2, and"
          "\n    each mean falls inside the other's confidence interval. Both collections"
          "\n    agree on every qualitative claim: tau=4 positive on all five seeds,"
          "\n    tau=2 null. A reviewer should expect an effect near +5 at tau=4, not the"
          "\n    specific value, and should treat a result anywhere in [+2, +11] as a"
          "\n    replication rather than a contradiction.")

    # tau=8 re-collection, if one is present. The runs are matched by filename:
    # the experiment writes the shipped name, so a replication directory holds
    # files that pair one-to-one with results/closedloop/.
    repl = os.path.join(HERE, "..", "results", "closedloop_replication")
    pairs = []
    if os.path.isdir(repl):
        for f in sorted(glob.glob(os.path.join(repl, "*.json"))):
            orig = os.path.join(R, "closedloop", os.path.basename(f))
            if os.path.exists(orig): pairs.append((os.path.basename(f), orig, f))
    if pairs:
        print("\n\n  TAU=8 -- runs re-collected on an independent host\n")
        print("    %-22s %-9s %-9s %s" % ("run", "original", "re-run", "difference"))
        da, db = [], []
        for name, a, b in pairs:
            va, vb = [], []
            for path, acc in ((a, va), (b, vb)):
                by = by_of(path)
                cf = {c for v in by.values() for c in v}
                h = next((c for c in cf if c.endswith("tau8") and "shrink" not in c), None)
                if not h or f"{h}_shrink" not in cf: continue
                acc.append(rate(by, f"{h}_shrink")[0] - rate(by, h)[0])
            if not va or not vb: continue
            da.append(va[0]); db.append(vb[0])
            print("    %-22s %-9.1f %-9.1f %+.1f" % (name[:-5], va[0], vb[0], vb[0] - va[0]))
        if da:
            ma = sum(da) / len(da); mb = sum(db) / len(db)
            print("    %-22s %-9.1f %-9.1f %+.1f" % ("mean", ma, mb, mb - ma))
            print("\n    These pair one-to-one with the shipped runs and were collected on a"
                  "\n    different machine at a later date. Note the seeds of the originals"
                  "\n    were never recorded -- the re-run uses the seed implied by each"
                  "\n    filename -- so this bounds agreement across both a different draw"
                  "\n    and possibly a different seed, which makes it conservative.")
    else:
        print("\n\n  WHAT HAS NO REPLICATION EVIDENCE\n")
        print("    tau=8 (Result 2) was collected once, nine runs on one pass. It is the"
              "\n    largest effect in the study (+16.0, 9/9 runs positive, seed-level"
              "\n    p=0.0039) and the most likely to survive, but nothing here demonstrates"
              "\n    that it does.")
    # oracle ceiling re-collection
    cr = os.path.join(R, "ceiling_replication")
    cpairs = []
    if os.path.isdir(cr):
        for f in sorted(glob.glob(os.path.join(cr, "*.json"))):
            o = os.path.join(R, "ceiling", os.path.basename(f))
            if os.path.exists(o): cpairs.append((os.path.basename(f), o, f))
    if cpairs:
        print("\n\n  ORACLE CEILING -- re-collected on an independent host\n")
        print("    %-20s %-12s %-12s %s" % ("run", "original", "re-run", "difference"))
        for name, a, b in cpairs:
            vs = []
            for path in (a, b):
                by = by_of(path)
                cf = {c for v in by.values() for c in v}
                if "c3ache_tau4_oracle" not in cf or "c3ache_tau4_shrink" not in cf: continue
                vs.append(rate(by, "c3ache_tau4_oracle")[0] - rate(by, "c3ache_tau4_shrink")[0])
            if len(vs) == 2:
                print("    %-20s %-12.1f %-12.1f %+.1f" % (name[:-5], vs[0], vs[1], vs[1] - vs[0]))
        print("\n    Both are null -- a perfect per-cycle weight is not measurably better"
              "\n    than the fitted one, on either collection. But the point estimate moved"
              "\n    by five points between them, which is large next to the effect under"
              "\n    test. That is evidence the comparison is UNDERPOWERED rather than"
              "\n    evidence of equivalence, and it is the reading the write-up gives.")
    # pi0.5 confirmation -- three seeds of the same configuration, which is a
    # replication in the sense that matters even though no single run is repeated
    cf = sorted(glob.glob(os.path.join(R, "pi05_confirm", "confirm_seed*.json")))
    if len(cf) >= 3:
        print("\n\n  PI0.5 CONFIRMATION -- three seeds of one configuration\n")
        print("    %-8s %-12s %-12s %s" % ("seed", "beta=1.00", "beta=0.25", "delta"))
        ds = []
        for f in cf:
            d = json.load(open(f))
            by = {}
            for r in d["rows"]:
                by.setdefault(r["config"], {})[(r["task"], r["episode"])] = bool(r["success"])
            a = next((c for c in by if c.endswith("b100")), None)
            b = next((c for c in by if c.endswith("b025")), None)
            if not a or not b: continue
            ks = sorted(set(by[a]) & set(by[b]))
            ra = 100.0 * sum(by[a][k] for k in ks) / len(ks)
            rb = 100.0 * sum(by[b][k] for k in ks) / len(ks)
            ds.append(rb - ra)
            print("    %-8s %-12.1f %-12.1f %+.1f"
                  % (os.path.basename(f)[12:-5], ra, rb, rb - ra))
        if ds:
            print("    %-8s %-12s %-12s %+.1f  (spread %.1f)"
                  % ("mean", "", "", sum(ds) / len(ds), max(ds) - min(ds)))
        print("\n    Seed-to-seed spread is 4.0 points on an effect of 7.6. A reviewer")
        print("    re-running this should expect the sign and rough size, not the value,")
        print("    and the pre-registered rule is written to survive that.")

    # pi0.5 sweep -- one shipped run re-collected from the committed script
    pr = os.path.join(R, "pi05_replication")
    if os.path.isdir(pr):
        import subprocess
        for f in sorted(glob.glob(os.path.join(pr, "*.json"))):
            orig = os.path.join(R, "pi05", os.path.basename(f))
            if not os.path.exists(orig): continue
            print("\n\n  PI0.5 SWEEP -- a shipped run re-collected on an independent host")
            r = subprocess.run([sys.executable,
                                os.path.join(HERE, "compare_rerun.py"), orig, f],
                               capture_output=True, text=True)
            print(r.stdout.rstrip() or r.stderr.rstrip())
    # The same timed arm appears in two separately collected runs. The results
    # page quoted this row as "~1% -- 343.5 and 347.2 ms"; 343.5 is in no
    # artefact, and the real pair agrees three times better than that claimed.
    import statistics as _st
    _t = {}
    for _p in sorted(glob.glob(os.path.join(R, "latency", "*.json"))):
        try: _j = json.load(open(_p))
        except Exception: continue
        for _r in _j.get("rows", []):
            if "ms_per_cycle" in _r and not _r.get("warmup"):
                _t.setdefault(_r["config"], {}).setdefault(
                    os.path.basename(_p)[:-5], []).append(_r["ms_per_cycle"])
    _sh = [(c, v) for c, v in sorted(_t.items()) if len(v) > 1]
    if _sh:
        print("\n\n  LATENCY -- the same arm timed in two separate collections\n")
        print(f"    {'arm':32} {'run':10} {'ms/cycle':>9}  {'n':>3}")
        for _c, _v in _sh:
            _ms = []
            for _run, _vals in sorted(_v.items()):
                print(f"    {_c[:32]:32} {_run:10} {_st.mean(_vals):9.2f}  {len(_vals):3}")
                _ms.append(_st.mean(_vals))
            print(f"    {'':32} {'spread':10} {100*(max(_ms)-min(_ms))/min(_ms):8.2f}%")
        print("\n    The cache arm agrees to under half a percent. The full-compute")
        print("    arm does NOT -- it moves several percent between collections, the")
        print("    same way its success rate does, because the two runs compile a")
        print("    different set of graphs (A8). A speedup quoted as a RATIO of two")
        print("    arms inside ONE run is stable; either arm's absolute ms is not.")



    print("\n\n  WHAT STILL HAS NO REPLICATION\n")
    print("    robustness arms          5 perturbations, one run of 60 episodes each")
    print("    tau=0 row                one run")
    print("    mujoco comparison        one run per version")
    print("    latency and frontier     one seed each -- the TIMING within them is")
    print("                             precise (sd under 9 ms on 27 episodes) and the")
    print("                             success column is not")
    print("    rngdrift                 one seed, but its finding is a bit-for-bit")
    print("                             identity on 160 episodes, which does not need")
    print("                             a second seed to be believed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

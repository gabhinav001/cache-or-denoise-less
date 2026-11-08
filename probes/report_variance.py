"""Aggregate the repeats of one experiment and report the spread, not just a mean.

The point of this script is that a reproduction is not a number, it is a number
and a spread. Closed-loop success on this benchmark moves up to 10 points across
seeds; two of the claims in an earlier draft of this work were written from
single runs and did not survive being repeated. So every table printed here
shows each repeat individually, the spread across them, and -- where the repeats
vary a seed rather than a machine -- the pooled paired test over all of them.

Pooling is legitimate here because the arms are matched within each run: every
episode contributes a (task, initial state, seed) pair evaluated under both
substitution rules, and discordant pairs from different seeds are still
independent draws from the same comparison.

Published values are carried inline so a reader can see immediately whether a
fresh run landed where the paper says it should. They are reference points, not
assertions: a repeat that misses them by less than the documented noise is a
successful reproduction, and one that misses by more is the interesting case.

usage: report_variance.py <kind> <dir-or-files...>
       kind = closedloop | residual | curve | mujoco | ceiling | fidelity
"""
from __future__ import annotations

import glob, json, math, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_closedloop as _RC
from report_closedloop import rate, mcnemar


def load(path):
    """report_closedloop.load, with the MAX_TASKS restriction applied."""
    d = json.load(open(path))
    d = _clip(d)
    by = {}
    for r in d["rows"]:
        by.setdefault((r["task"], r["episode"]), {})[r["config"]] = r["success"]
    return d, by

# What the paper reports, for side-by-side comparison. (value, tolerance, note)
PUBLISHED = {
    "closedloop.rescue_frac":    (0.761, 0.08, "287 of 377 discordant pairs, 9 runs / 3 seeds"),
    "closedloop.delta_range":    ((8.5, 39.0), 3.0, "per-run delta at tau=8, 9 runs"),
    "residual.fw.cosine":   (0.803, 0.010, "Fast-WAM, 52 episodes"),
    "residual.fw.beta":     (0.800, 0.010, "Fast-WAM"),
    "residual.pi05.cosine": (0.7453, 0.010, "pi0.5, 52 episodes"),
    "residual.pi05.beta":   (0.7368, 0.010, "pi0.5"),
    "residual.dz.cosine":   (0.7413, 0.010, "DreamZero, 20 episodes"),
    "residual.dz.beta":     (0.7374, 0.010, "DreamZero"),
    # Was 0.0: the original claim was that attenuation leaves the frontier
    # unmoved. It does not -- the attenuated arm at tau=4 loses to the plain arm
    # at tau=2 by 2.8 points on average, which is the finding, and a reference of
    # 0 made the harness print DIFFERS on every run while still exiting 0.
    "frontier.dominates": (-2.8, 2.0, "attenuated tau=4 minus plain tau=2, per seed"),
    "curve.tau4":  ((85.0, 85.0), 5.0, "seeds 42 and 7 both landed on 85.0"),
    "curve.tau0":  ((11.2, 13.8), 5.0, "never-refresh, the headline setting"),
    "mujoco.gap":  (22.6, 5.0, "3.3.2 minus 2.3.7, same box"),
    "ceiling.compiled": ((1.503, 1.584), 0.10, "cache ceiling over a compiled baseline"),
    "fidelity.gain": ((15.8, 16.3), 2.0, "percent reduction in action deviation"),
}


# A run interrupted partway covers only its first N tasks. Tasks differ in
# difficulty, so pooling a 9-task run with a 5-task one compares different work.
# MAX_TASKS restricts every artefact to a common prefix so the comparison is like
# for like; set it from the environment.
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0")) or None


def _clip(d):
    if MAX_TASKS is None: return d
    d = dict(d); d["rows"] = [r for r in d["rows"] if r["task"] < MAX_TASKS]
    return d


def files(args, ext):
    out = []
    for a in args:
        if os.path.isdir(a): out += sorted(glob.glob(os.path.join(a, "**", "*" + ext), recursive=True))
        elif a.endswith(ext): out.append(a)
    return out


def spread(vals):
    """Precision follows the magnitude: a ceiling of 1.503 and a success rate of
    85.0 need different numbers of digits to be readable."""
    if not vals: return "n/a"
    d = 3 if max(abs(v) for v in vals) < 10 else 1
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}  (SINGLE RUN -- not a reproduction)"
    return (f"{min(vals):.{d}f} to {max(vals):.{d}f}  "
            f"(spread {max(vals)-min(vals):.{d}f})")


DISAGREEMENTS = []


def cmp_line(key, got):
    """Compare a re-derived figure with the value the write-up states.

    A disagreement is recorded, not just printed. The harness used to print
    DIFFERS and exit 0, which meant the one check whose entire purpose is to
    catch the paper drifting from its artefacts could fail silently for weeks --
    and did.
    """
    if key not in PUBLISHED: return ""
    ref, tol, note = PUBLISHED[key]
    if isinstance(ref, tuple):
        lo, hi = min(ref), max(ref)
        t = tol or 0
        ok = (lo - t) <= got <= (hi + t)
        if not ok: DISAGREEMENTS.append((key, got, f"{lo:g}-{hi:g}", note))
        return f"   published {lo:g}-{hi:g} ({note}) -> {'AGREES' if ok else 'DIFFERS'}"
    ok = tol is None or abs(got - ref) <= tol
    if not ok: DISAGREEMENTS.append((key, got, f"{ref:g}", note))
    return f"   published {ref:g} ({note}) -> {'AGREES' if ok else 'DIFFERS'}"


# --------------------------------------------------------------- closed loop
def _tau_of(cfg):
    if "tau" not in cfg: return None
    t = cfg.split("tau")[-1].replace("_shrink", "")
    return int(t) if t.isdigit() else None


def do_closedloop(paths, tau=8):
    if not paths: return print("no .json artefacts found")
    print(f"\n{'run':34s} {'tau':>4s} {'full':>7s} {'cached':>7s} {'+weight':>8s} "
          f"{'delta':>7s} {'resc/brok':>10s} {'p':>9s}")
    print("-" * 96)
    # a single artefact can hold several refresh intervals; each is its own
    # comparison, and mixing them silently reports the wrong arm
    # [rescued, broken, deltas, then (full, plain, attenuated) per run]
    bytau = defaultdict(lambda: [0, 0, []])
    for p in paths:
        try: d, by = load(p)
        except Exception: continue
        cfgs = sorted({c for v in by.values() for c in v})
        rf = rate(by, "full")[0] if "full" in cfgs else float("nan")
        for t in sorted({_tau_of(c) for c in cfgs} - {None}):
            hold, shr = f"c3ache_tau{t}", f"c3ache_tau{t}_shrink"
            if hold not in cfgs or shr not in cfgs: continue
            rh = rate(by, hold)[0]; rs = rate(by, shr)[0]
            n01, n10, pv = mcnemar(by, hold, shr)     # n01 = rescued by the weight
            mark = " *" if t == tau else "  "
            print(f"{os.path.basename(p)[:34]:34s} {t:4d} {rf:7.1f} {rh:7.1f} {rs:8.1f} "
                  f"{rs-rh:+7.1f} {n01:5d}/{n10:<4d} {pv:9.1e}{mark}")
            bytau[t][0] += n01; bytau[t][1] += n10; bytau[t][2].append(rs - rh)
            bytau[t].append((rf, rh, rs)) if len(bytau[t]) > 3 else bytau[t].append((rf, rh, rs))
    if not bytau: return print("no paired hold/shrink arms found")
    print("-" * 96)
    print("  * = the refresh interval the headline result is stated at\n")

    print(f"  {'tau':>4s} {'runs':>5s} {'full':>6s} {'plain':>6s} {'+beta':>6s} "
          f"{'rescued':>8s} {'broken':>7s} {'pooled p':>10s}  delta")
    for t in sorted(bytau):
        R, B, ds = bytau[t][0], bytau[t][1], bytau[t][2]
        trip = bytau[t][3:]
        mf = sum(x[0] for x in trip) / len(trip) if trip else float("nan")
        mh = sum(x[1] for x in trip) / len(trip) if trip else float("nan")
        ms = sum(x[2] for x in trip) / len(trip) if trip else float("nan")
        n, k = R + B, min(R, B)
        pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        print(f"  {t:4d} {len(ds):5d} {mf:6.1f} {mh:6.1f} {ms:6.1f} "
              f"{R:8d} {B:7d} {pv:10.1e}  {spread(ds)}")
    if len(bytau) > 1:
        print("\n  The gradient across tau is the boundary condition: the correction is\n"
              "  worth more the staler the cached residual is.")

    if tau in bytau:
        R, B, ds = bytau[tau][0], bytau[tau][1], bytau[tau][2]
        frac = R / (R + B) if (R + B) else float("nan")
        print(f"\n  at tau={tau}:")
        print(f"    runs             {len(ds)}")
        print(f"    per-run delta    {spread(ds)}")
        print("   " + cmp_line("closedloop.delta_range", sum(ds) / len(ds)).strip())
        print(f"    rescued fraction {frac:.3f}  ({R} of {R+B} discordant pairs)")
        print("   " + cmp_line("closedloop.rescue_frac", frac).strip())
        if len(ds) == 1:
            print("\n    ONE RUN ONLY. Closed-loop success moves up to 10 points across\n"
                  "    seeds; a single run cannot establish an effect of this size.\n"
                  '    Re-run with --repeats "42 7 11".')


# ------------------------------------------------------------------ residual
def _backbone(name):
    n = os.path.basename(name)
    if n.startswith("pi05"): return "pi05"
    if n.startswith("dz"):   return "dz"
    # the idm checkpoint is a different model; pooling it into Fast-WAM's spread
    # would report a checkpoint difference as reproduction noise
    if "idm" in n: return "fw_idm"
    return "fw"


def do_residual(paths):
    """Reproduce the geometry and the one-scalar gain using the PAPER'S estimator.

    Deliberately calls report_families.summarize rather than recomputing, so a
    reproduction cannot pass by accidentally measuring something easier. The
    geometry lines below are copied from the same source and use the median, as
    the paper does -- the mean over cells is a different and worse-behaved
    statistic because a handful of near-zero-norm cells dominate it.
    """
    import io, contextlib
    import numpy as np
    import report_families as RF
    from analyze_families import unpack

    if not paths: return print("no .npz artefacts found")
    verbose = os.environ.get("VERBOSE", "") not in ("", "0")
    groups = defaultdict(list)
    for p in paths: groups[_backbone(p)].append(p)

    for bb in sorted(groups):
        print(f"\n=== {bb} ===")
        print(f"{'run':32s} {'obs':>9s} {'cosine':>8s} {'deg':>5s} {'magnitude':>10s} "
              f"{'beta':>8s} {'beta<1':>7s} {'gain %':>7s}")
        print("-" * 92)
        rows = []
        for p in sorted(groups[bb]):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    res = RF.summarize(p, os.path.basename(p))
                z, G, cell, ep = RF.load(p)
                G = G[z_keep(G)]
            except Exception as e:
                # A section that cannot compute must fail the run. This printed
                # "failed:" and carried on, so a broken analysis looked like a
                # successful verification with a couple of odd-looking lines --
                # which is exactly how a tuple-width change went unnoticed.
                print(f"{os.path.basename(p)[:32]:32s}  failed: {e}")
                DISAGREEMENTS.append((f"residual:{os.path.basename(p)}",
                                      float("nan"), "computable", str(e)))
                continue
            if verbose: print(buf.getvalue())
            beta = G[:, 0, 1] / np.maximum(G[:, 1, 1], 1e-30)
            rho  = G[:, 0, 1] / np.sqrt(np.maximum(G[:, 0, 0] * G[:, 1, 1], 1e-30))
            nr   = np.sqrt(G[:, 0, 0] / np.maximum(G[:, 1, 1], 1e-30))
            cos, mag, bt = float(np.median(rho)), float(np.median(nr)), float(np.median(beta))
            frac = 100 * float((beta < 1).mean())
            # Rows carry their own held-out baseline, because not every row holds
            # out the same thing -- the disjoint-task rows hold out TASKS and must
            # be scored against the hold error on those tasks. Match the plain
            # per-cell row explicitly rather than by prefix: "shrink1" also prefixes
            # the disjoint-task and quartile rows.
            gain = next((100 * (1 - h / bh)
                         for n, o, h, bh in res["rows"]
                         if n.startswith("shrink1 [") and h), float("nan"))
            deg = float(np.degrees(np.arccos(min(max(cos, -1), 1))))
            # error from substituting a SINGLE global weight, which is what the
            # "the optimum is interior" claim rests on
            # SAME estimator report_families uses: the mean of per-observation
            # relative errors, not a pooled ratio. The two differ by ~0.10 here,
            # and mixing them is how a table ends up internally inconsistent.
            tt, t1, oo = G[:, 0, 0], G[:, 0, 1], G[:, 1, 1]
            curve = {b: float(np.mean(np.sqrt(np.maximum(
                        tt - 2 * b * t1 + b * b * oo, 0.0) / np.maximum(tt, 1e-30))))
                     for b in (0.0, 0.8, 1.0)}
            rows.append((os.path.basename(p), res["n"], cos, deg, mag, bt, frac, gain, curve))
            print(f"{os.path.basename(p)[:32]:32s} {res['n']:9,d} {cos:8.4f} {deg:5.1f} "
                  f"{mag:10.4f} {bt:8.4f} {frac:6.1f}% {gain:7.2f}")
        if len(rows) < 2:
            print("\n  ONE RUN ONLY -- this measurement is deterministic given its episode\n"
                  "  set, so a reproduction must run on a DIFFERENT machine, not again here.")
        else:
            print("-" * 92)
            for i, name in ((2, "cosine"), (4, "magnitude"), (5, "beta"), (7, "gain %")):
                v = [r[i] for r in rows]
                print(f"  {name:10s} {min(v):.4f} to {max(v):.4f}   spread {max(v)-min(v):.4f}")
        if rows:
            c = rows[0][8]
            print(f"\n  error from one GLOBAL weight (the optimum is interior):")
            print(f"    beta=0 (skip the block) {c[0.0]:.4f}   "
                  f"beta=1 (published) {c[1.0]:.4f}   beta=0.8 {c[0.8]:.4f}")
            print(cmp_line(f"residual.{bb}.cosine", rows[0][2]))
            print(cmp_line(f"residual.{bb}.beta", rows[0][5]))


def z_keep(G):
    import numpy as np
    return G[:, 0, 0] > 0





# ---------------------------------------------------------------- generalize
def do_generalize(paths):
    """Does the correction survive an observation shift?

    Two things must both be read, and reading only one is how this measurement
    goes wrong. First, is the shift real? If full compute is untouched, nothing
    was perturbed and the row says nothing about robustness. Second, given a real
    shift, does the correction still recover what caching costs?
    """
    if not paths: return print("no .json artefacts found")
    rows = []
    for p in sorted(paths):
        try: d, by = load(p)
        except Exception: continue
        pert = (json.load(open(p)).get("meta", {}) or {}).get("perturb", "?")
        cfgs = {c for v in by.values() for c in v}
        hold = next((c for c in cfgs if c.endswith(("tau2", "tau4", "tau8"))), None)
        shr = f"{hold}_shrink" if hold else None
        if not hold or shr not in cfgs: continue
        rf = rate(by, "full")[0]; rh = rate(by, hold)[0]; rs = rate(by, shr)[0]
        n01, n10, pv = mcnemar(by, hold, shr)
        rows.append((pert, rf, rh, rs, rs - rh, n01, n10, pv))
    if not rows: return print("no paired arms found")

    clean = next((r for r in rows if r[0] in ("none", "?")), None)
    base_full = clean[1] if clean else float("nan")
    base_delta = clean[4] if clean else float("nan")

    print(f"\n{'perturbation':16s} {'full':>7s} {'shift?':>8s} {'cached':>7s} "
          f"{'+weight':>8s} {'delta':>7s} {'resc/brok':>10s} {'p':>8s}")
    print("-" * 82)
    for pert, rf, rh, rs, d, n01, n10, pv in rows:
        drop = base_full - rf
        shift = "baseline" if pert in ("none", "?") else (
            f"-{drop:.1f}pts" if drop > 1.0 else "NONE")
        print(f"{pert:16s} {rf:7.1f} {shift:>8s} {rh:7.1f} {rs:8.1f} {d:+7.1f} "
              f"{n01:5d}/{n10:<4d} {pv:8.3f}")
    print("-" * 82)

    real = [r for r in rows if r[0] not in ("none", "?") and (base_full - r[1]) > 1.0]
    inert = [r for r in rows if r[0] not in ("none", "?") and (base_full - r[1]) <= 1.0]
    if inert:
        print(f"\n  {len(inert)} perturbation(s) left full compute intact and so did not")
        print( "  actually shift the distribution; they cannot support a robustness")
        print(f"  claim either way: {', '.join(r[0] for r in inert)}")
    if not real:
        return print("\n  NO REAL SHIFT TESTED -- increase the magnitudes and re-run.")
    ds = [r[4] for r in real]
    R = sum(r[5] for r in real); B = sum(r[6] for r in real)
    n, k = R + B, min(R, B)
    pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"\n  under a genuine shift ({len(real)} perturbations):")
    print(f"    correction delta   {spread(ds)}")
    print(f"    clean baseline     {base_delta:+.1f}")
    print(f"    pooled             {R} rescued / {B} broken, p={pv:.1e}")
    mean = sum(ds) / len(ds)
    if mean > 0 and pv < 0.05:
        print("    -> the correction survives the shift")
    elif mean > 0:
        print("    -> positive but not distinguishable from noise at this n")
    else:
        print("    -> the correction does NOT survive the shift")


# ------------------------------------------------------------- applicability
def _cells(paths, hold, shr):
    """Every (run, task) pair carrying all three arms, as episode dicts."""
    out = []
    for p in sorted(paths):
        try: d = json.load(open(p))
        except Exception: continue
        bytask = defaultdict(lambda: defaultdict(dict))
        for r in d["rows"]:
            bytask[r["task"]][(r["task"], r["episode"])][r["config"]] = r["success"]
        for _, by in bytask.items():
            cfgs = {c for v in by.values() for c in v}
            if hold in cfgs and shr in cfgs and "full" in cfgs:
                out.append(by)
    return out


def _tally(chooser, scorer, hold, shr, eps):
    """Bucket on `chooser` episodes, count rescued/broken on `scorer` episodes."""
    g = {"caching costs something": [0, 0, 0], "caching costs nothing": [0, 0, 0]}
    for ch, sc in zip(chooser, scorer):
        try:
            loss = rate(ch, "full")[0] - rate(ch, hold)[0]
        except Exception:
            continue
        n01, n10, _ = mcnemar(sc, hold, shr)
        k = "caching costs something" if loss > eps else "caching costs nothing"
        g[k][0] += 1; g[k][1] += n01; g[k][2] += n10
    return g


def _exact(R, B):
    tot, k = R + B, min(R, B)
    return min(1.0, 2 * sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot) if tot else 1.0


def do_applicability(paths, tau=8, eps=0.1, draws=400, seed=7):
    """Where does the correction help, and where does it hurt?

    Splits every (run, task) pair by whether caching cost anything on that task.
    The intuition is sound -- if the cached arm already matches full compute
    there is no loss to recover -- but the obvious way to measure it is wrong,
    and this reports both so the difference is visible.

    The split variable is full - cached and the outcome is shrink - cached. Both
    contain the cached arm, and computing them on the SAME episodes makes the
    comparison circular: selecting tasks where cached happened to match full
    selects the upper tail of that arm's sampling noise, and the shrink arm,
    which was not part of the selection, regresses to the mean. The bucket then
    reports harm that is an artefact of how it was chosen.

    Measured here: in-sample, the "costs nothing" bucket reads 2 rescued / 18
    broken, p = 4e-04, which this function used to print as the verdict
    "HARMFUL". Choosing the bucket on half the episodes and scoring it on the
    disjoint half reverses it to roughly 13 / 10, and across 400 random splits
    it never once reaches p < 0.05 in the harmful direction. There is no
    evidence the correction hurts where caching is free.

    What survives is the other half of the finding, which does not depend on the
    split at all: the benefit tracks how much caching costs, so it is large at
    tau=8 and absent at tau=2. That is a between-configuration comparison with
    independent arms, and it is the form the usability argument should take.
    """
    hold = f"c3ache_tau{tau}"; shr = f"{hold}_shrink"
    cells = _cells(paths, hold, shr)
    if not cells: return print("no .json artefacts found")

    ins = _tally(cells, cells, hold, shr, eps)
    print(f"\n  splitting {sum(v[0] for v in ins.values())} (run, task) pairs at tau={tau}\n")
    print(f"  {'regime':26s} {'pairs':>6s} {'rescued':>8s} {'broken':>7s} {'p':>10s}  verdict")
    print("  " + "-" * 78)
    for k, (n, R, B) in ins.items():
        pv = _exact(R, B)
        # A verdict is never issued from this row: the split and the outcome were
        # computed on the same episodes, so "HARMFUL" here would be circular.
        note = "(in-sample -- biased, see below)"
        print(f"  {k:26s} {n:6d} {R:8d} {B:7d} {pv:10.1e}  {note}")

    rng = random.Random(seed)
    acc = {k: [0.0, 0.0] for k in ins}; harm = {k: 0 for k in ins}; ben = {k: 0 for k in ins}
    for _ in range(draws):
        ch, sc = [], []
        for by in cells:
            ks = list(by); rng.shuffle(ks); h = len(ks) // 2
            if h < 1: continue
            ch.append({k: by[k] for k in ks[:h]}); sc.append({k: by[k] for k in ks[h:]})
        g = _tally(ch, sc, hold, shr, eps)
        for k, (_, R, B) in g.items():
            acc[k][0] += R / draws; acc[k][1] += B / draws
            if _exact(R, B) < 0.05:
                (harm if B > R else ben)[k] += 1
    print(f"\n  bucket chosen on half the episodes, scored on the disjoint half"
          f" ({draws} random splits):\n")
    print(f"  {'regime':26s} {'rescued':>8s} {'broken':>7s}   significant splits")
    print("  " + "-" * 78)
    for k in ins:
        R, B = acc[k]
        print(f"  {k:26s} {R:8.1f} {B:7.1f}   beneficial {ben[k]:3d}/{draws}, "
              f"harmful {harm[k]:3d}/{draws}")
    if harm["caching costs nothing"] == 0:
        print("\n  The correction is NOT shown to hurt where caching is free. The in-sample"
              "\n  split says so only because it selects on the arm it then scores against.")
    return ins, acc


# ------------------------------------------------------------------ frontier
SAVED = {1: 0, 2: 40, 4: 60, 8: 70, 0: 80}

def do_frontier(paths):
    """Does attenuation move the compute/quality frontier?

    The within-tau delta (attenuated vs plain at the SAME tau) is not the
    question a practitioner asks. They ask: at a given quality, how much compute
    can I save? So the comparison that matters is the attenuated arm at a HIGHER
    tau against the plain arm at a LOWER one -- more compute saved, and the
    question is whether quality holds. Both arms live in the same run, on the
    same tasks, initial states and seed, so the comparison is paired.
    """
    if not paths: return print("no .json artefacts found")
    rows = []
    for p in sorted(paths):
        try: d, by = load(p)
        except Exception: continue
        cfgs = {c for v in by.values() for c in v}
        taus = sorted({_tau_of(c) for c in cfgs} - {None})
        full = rate(by, "full")[0] if "full" in cfgs else float("nan")
        for hi in taus:
            shr = f"c3ache_tau{hi}_shrink"
            if shr not in cfgs: continue
            for lo in taus:
                pl = f"c3ache_tau{lo}"
                if lo >= hi or pl not in cfgs: continue
                if SAVED.get(hi, 0) <= SAVED.get(lo, 0): continue
                a = rate(by, pl)[0]; b = rate(by, shr)[0]
                n01, n10, pv = mcnemar(by, pl, shr)
                rows.append((os.path.basename(p), lo, hi, a, b, b - a,
                             SAVED.get(lo, 0), SAVED.get(hi, 0), n01, n10, pv, full))
    if not rows: return print("no higher-tau attenuated vs lower-tau plain pairs found")

    print(f"\n{'run':24s} {'plain':>12s} {'attenuated':>14s} {'quality':>8s} {'extra':>7s} "
          f"{'win/lose':>9s} {'p':>8s}")
    print(f"{'':24s} {'tau  saved':>12s} {'tau  saved':>14s} {'delta':>8s} {'saved':>7s}")
    print("-" * 92)
    for r in rows:
        f, lo, hi, a, b, d, sl, sh, n01, n10, pv, full = r
        mark = "" if d >= 0 else "  <-- worse"
        print(f"{f[:24]:24s} {lo:4d} {sl:4d}% {a:5.1f} {hi:4d} {sh:4d}% {b:5.1f} "
              f"{d:+8.1f} {sh-sl:+6d}% {n01:4d}/{n10:<4d} {pv:8.3f}{mark}")
    print("-" * 92)

    # group by the (lo,hi) pair so seeds are pooled like with like
    pairs = defaultdict(list)
    for r in rows: pairs[(r[1], r[2])].append(r)
    for (lo, hi), rs in sorted(pairs.items()):
        ds = [r[5] for r in rs]
        R = sum(r[8] for r in rs); B = sum(r[9] for r in rs)
        n, k = R + B, min(R, B)
        pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        extra = SAVED.get(hi, 0) - SAVED.get(lo, 0)
        mean = sum(ds) / len(ds)
        print(f"\n  attenuated tau={hi} vs plain tau={lo}  (+{extra}% compute saved)")
        print(f"    per-seed quality delta   {spread(ds)}")
        print(f"    mean                     {mean:+.1f} points")
        print(f"    pooled                   {R} better / {B} worse, p={pv:.3f}")
        better = sum(1 for d in ds if d > 0); equal = sum(1 for d in ds if d == 0)
        print(f"    seeds better/equal/worse {better}/{equal}/{len(ds)-better-equal}")
        if mean > 0 and better > len(ds) - better:
            print("    -> DOMINATES: more compute saved AND better quality")
        elif abs(mean) <= 2.0:
            print("    -> comparable quality for more compute saved (within noise)")
        else:
            print("    -> does NOT dominate: the extra saving costs quality")
        print("   " + cmp_line("frontier.dominates", mean).strip())


# --------------------------------------------------------------------- curve
def do_curve(paths):
    if not paths: return print("no .json artefacts found")
    saved = {"full": 0, "1": 0, "2": 40, "4": 60, "8": 70, "0": 80}
    per = defaultdict(dict)
    for p in paths:
        d, by = load(p)
        for c in sorted({c for v in by.values() for c in v}):
            per[os.path.basename(p)][c] = rate(by, c)[0]
    cfgs = sorted({c for v in per.values() for c in v},
                  key=lambda c: (c != "full", c))
    print(f"\n{'run':30s}" + "".join(f"{c:>14s}" for c in cfgs))
    print(f"{'block evals saved':30s}" + "".join(
        f"{str(saved.get(c.split('tau')[-1], '?'))+'%':>14s}" for c in cfgs))
    print("-" * (30 + 14 * len(cfgs)))
    for run, v in per.items():
        print(f"{run[:30]:30s}" + "".join(f"{v.get(c, float('nan')):14.1f}" for c in cfgs))
    if len(per) < 2:
        return print("\n  ONE SEED ONLY -- re-run with --repeats \"42 7\".")
    for c in cfgs:
        v = [x[c] for x in per.values() if c in x]
        if len(v) > 1 and max(v) - min(v) > 0:
            print(f"  {c:14s} spread across seeds: {max(v)-min(v):.1f} points")
    t4 = [v.get("c3ache_tau4") for v in per.values() if "c3ache_tau4" in v]
    t0 = [v.get("c3ache_tau0") for v in per.values() if "c3ache_tau0" in v]
    if t4: print(cmp_line("curve.tau4", sum(t4) / len(t4)))
    if t0: print(cmp_line("curve.tau0", sum(t0) / len(t0)))


# -------------------------------------------------------------------- mujoco
def do_mujoco(paths):
    if not paths: return print("no .json artefacts found")
    got = {}
    for p in paths:
        d, by = load(p)
        v = os.path.basename(p).replace("mujoco_", "").replace(".json", "")
        got[v] = rate(by, "full")[0]
    for v in sorted(got, reverse=True):
        print(f"  mujoco {v:8s} full-compute success {got[v]:5.1f}%")
    if len(got) == 2:
        a, b = sorted(got, reverse=True)
        gap = got[a] - got[b]
        print(f"\n  gap {gap:+.1f} points  (same box, same seed, same checkpoint)")
        print(cmp_line("mujoco.gap", gap))
    else:
        print("\n  BOTH ARMS MUST RUN ON ONE BOX or the gap is confounded with\n"
              "  cross-machine variance, which is itself ~15 points.")


# ------------------------------------------------------------------- ceiling
def do_ceiling(paths):
    """Compiled-vs-eager ceiling. The sweep script already fits t(n) and records
    the verdict, so this reads it rather than refitting."""
    if not paths: return print("no .json artefacts found")
    ceils, comps = [], []
    print(f"\n{'run':34s} {'eager ms':>9s} {'compiled':>9s} {'fixed':>7s} "
          f"{'per-step':>9s} {'R2':>7s} {'compile x':>10s} {'ceiling':>8s}")
    print("-" * 100)
    for p in sorted(paths):
        d = json.load(open(p))
        v, f = d.get("verdict"), d.get("fits")
        if not v or not f:
            print(f"  {os.path.basename(p)[:34]:34s}  not a step sweep (no verdict/fits)"); continue
        c = f.get("compiled", {}); e = f.get("eager", {})
        ceil = v["cache_ceiling_vs_compiled_x"]
        ceils.append(ceil); comps.append(v["compile_speedup_x"])
        print(f"{os.path.basename(p)[:34]:34s} {e.get('total_at_10_steps_ms',float('nan')):9.1f} "
              f"{v['honest_baseline_ms']:9.1f} {v['uncacheable_fixed_ms_compiled']:7.1f} "
              f"{c.get('per_denoise_step_ms',float('nan')):9.2f} {c.get('r2',float('nan')):7.4f} "
              f"{v['compile_speedup_x']:10.2f} {ceil:8.3f}")
    if not ceils: return
    print("-" * 100)
    print(f"  cache ceiling over a compiled baseline   {spread(ceils)}")
    print(f"  compile speedup alone                    {spread(comps)}")
    print(cmp_line("ceiling.compiled", sum(ceils) / len(ceils)))
    print(f"\n  A reported speedup above the ceiling cannot have been measured against\n"
          f"  a compiled baseline. Published claim: 2.51x.")
    if len(ceils) < 2:
        print("  ONE MACHINE ONLY -- the ceiling is hardware-dependent; run two.")


# ------------------------------------------------------------------ fidelity
def do_fidelity(paths):
    """Held-out cycles ONLY. Cycles from episodes the weights were fit on are
    marked in_fit and must be excluded, or the gain is optimistic."""
    gains = []
    if not paths: return print("no .json artefacts found")
    for p in sorted(paths):
        d = json.load(open(p))
        rows = [r for r in d.get("rows", []) if not r.get("in_fit", False)]
        if not rows:
            print(f"  {os.path.basename(p)[:34]:34s}  no held-out rows"); continue
        h = sum(r["hold"] for r in rows) / len(rows)
        s_ = sum(r["shrink"] for r in rows) / len(rows)
        wins = sum(1 for r in rows if r["shrink"] < r["hold"])
        g = 100 * (1 - s_ / h); gains.append(g)
        print(f"  {os.path.basename(p)[:34]:34s} held-out {len(rows):4d}  hold {h:.4f}  "
              f"attenuated {s_:.4f}  gain {g:+5.1f}%  wins {wins}/{len(rows)}")
    if gains:
        print(f"\n  gain {spread(gains)}")
        print(cmp_line("fidelity.gain", sum(gains) / len(gains)))
    if len(gains) < 2:
        print("  ONE MACHINE ONLY -- run a second to establish agreement.")


KINDS = {"closedloop": (do_closedloop, ".json"), "residual": (do_residual, ".npz"),
         "frontier": (do_frontier, ".json"), "applicability": (do_applicability, ".json"),
         "generalize": (do_generalize, ".json"),
         "curve": (do_curve, ".json"), "mujoco": (do_mujoco, ".json"),
         "ceiling": (do_ceiling, ".json"), "fidelity": (do_fidelity, ".json")}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in KINDS:
        sys.exit(f"usage: report_variance.py <{'|'.join(KINDS)}> <dir-or-files...>")
    fn, ext = KINDS[sys.argv[1]]
    paths = [p for p in files(sys.argv[2:], ext) if "gate" not in os.path.basename(p)]
    fn(paths)
    if DISAGREEMENTS:
        print(f"\n  {len(DISAGREEMENTS)} figure(s) disagree with the stated value:")
        for key, got, ref, note in DISAGREEMENTS:
            print(f"    {key}: re-derived {got:.4g}, stated {ref}  ({note})")
        print("  Either the write-up is stale or the artefacts changed. Fix one.")
        sys.exit(1)

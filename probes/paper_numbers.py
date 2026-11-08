"""Every number in the paper's text, the snapshot entry it is read from, and a check.

The text and tables of the paper quote one analysis snapshot,
results/budget_shift/summary.json, which probes/report_budget_shift.py derives
from the raw run files. The tables and figures are generated whole from that
snapshot (probes/make_tables.py, probes/make_figure.py); this file covers the
numbers in the prose, in the order they appear. Each number has a kind:

  outcome    success rates, discordant pairs and the statistics computed from
             them, and the gate's replay error. Rollouts are deterministic on an
             RTX A6000 (Fast-WAM's fixed per-call noise; fresh noise is seeded per
             episode), so a regeneration on that GPU must reproduce these exactly.
  latency    milliseconds and ratios of them. The same runs regenerate them, but
             the values belong to the machine: compare them within the new host.
  hosts      tallies over the machines that ran the shipped collections (hosts,
             revisions, paired re-runs, cross-host latency ratios). They are
             recomputed from the shipped logs; a new host adds its own comparison.
  formula    arithmetic on the sampler grid (Eq. 1) or on other numbers here.
             Where the snapshot records the value (the harness writes each run's
             cached share), the check reads it from there.
  protocol   design constants set by the experiment scripts (states per task,
             tau, steps, jitter), read from the snapshot where it records them.
  cited      reported by other papers or read from upstream Fast-WAM's configs.
  definition notation (sigma = 1 at pure noise, h_0, SHA-256).

usage:
  python3 probes/paper_numbers.py                               # the shipped snapshot
  python3 probes/paper_numbers.py --summary regen/summary.json  # a regenerated one
  python3 probes/paper_numbers.py --markdown repro/PAPER_NUMBERS.md

With --summary, outcome, formula and protocol numbers must equal the printed
value; latency and hosts numbers are reported next to the printed value.
Exit status is 1 if a number that must reproduce does not.
"""
from __future__ import annotations

import argparse
import math
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "budget_shift" / "summary.json"
EXACT = ("outcome", "formula", "protocol")
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}

# ---------------------------------------------------------------- snapshot paths
# A path is a /-separated walk through summary.json; {a/b} is one key that
# contains a slash. A list is indexed either by position, [3], or by the fields of
# the element wanted, [label=...] or [ckpt=idm&shift=1.0&prefix=8], so a path keeps
# pointing at the same comparison when a partial regeneration leaves out others in
# the same list.
_TRACE: list[str] | None = None


def _segments(path):
    out, cur, i = [], "", 0
    while i < len(path):
        ch = path[i]
        if ch == "{":
            j = path.index("}", i)
            cur += path[i + 1:j]; i = j + 1
            continue
        if ch == "[":
            j = path.index("]", i)
            if cur:
                out.append(cur); cur = ""
            out.append(path[i:j + 1]); i = j + 1
            continue
        if ch == "/":
            if cur:
                out.append(cur); cur = ""
        else:
            cur += ch
        i += 1
    if cur:
        out.append(cur)
    return out


def _match(elem, sel):
    for cond in sel.split("&"):
        k, v = cond.split("=", 1)
        x = elem.get(k) if isinstance(elem, dict) else None
        if x is None or (str(x) != v and json.dumps(x) != v):
            return False
    return True


def g(S, path):
    """The value at a snapshot path; KeyError if the snapshot does not hold it."""
    node, resolved = S, []
    for seg in _segments(path):
        if seg.startswith("["):
            inner = seg[1:-1]
            if not isinstance(node, list):
                raise KeyError(f"{path}: {'/'.join(resolved)} is not a list")
            if inner.lstrip("-").isdigit():
                idx = int(inner)
            else:
                hits = [i for i, e in enumerate(node) if _match(e, inner)]
                if len(hits) != 1:
                    raise KeyError(f"{path}: {len(hits)} entries match [{inner}]")
                idx = hits[0]
            node = node[idx]
            resolved.append(f"[{idx}]")
        else:
            if not isinstance(node, dict) or seg not in node:
                raise KeyError(f"{path}: no '{seg}' under {'/'.join(resolved) or 'the top level'}")
            node = node[seg]
            resolved.append(seg)
    if _TRACE is not None:
        _TRACE.append("/".join(resolved).replace("/[", "["))
    return node


# ---------------------------------------------------------------- the grid (Eq. 1)
def sigma(K, m, s):
    """Noise level after m of K Euler steps on the shifted grid, Eq. 1."""
    u = 1 - m / K
    return s * u / (1 + (s - 1) * u)


def share(K, m, s):
    """Share of the flow path integrated by the first m of K steps, 1 - sigma_m."""
    return 1 - sigma(K, m, s)


def arm_share(arm):
    """Cached share of a 10-step arm, from its name: {ckpt}_s{S}_{pM|whole}_tau{T}."""
    m = re.match(r"(idm|uncond)_s(\d)_(p(\d)|whole|stock\d+)", arm)
    if not m or m.group(3).startswith("stock"):
        return 0.0
    return share(10, int(m.group(4)) if m.group(4) else 8, float(m.group(2)))


# ---------------------------------------------------------------- numbers
class Num:
    def __init__(self, printed, kind, how, get=None, fmt="%.0f", scale=1.0, sign=1.0, word=False):
        self.printed, self.kind, self.how = printed, kind, how
        self.get, self.fmt, self.scale, self.sign, self.word = get, fmt, scale, sign, word

    def value(self, S):
        """The number as the paper would print it from this snapshot, or None for constants."""
        if self.get is None:
            return None
        x = self.get(S) if callable(self.get) else g(S, self.get)
        if isinstance(x, str):
            return x
        x = self.sign * self.scale * x
        if self.word:
            return WORDS.get(int(round(x)), str(x))
        return self.fmt % x


def V(printed, path, kind="outcome", **kw):
    """A value read from one snapshot path."""
    return Num(printed, kind, path, get=path, **kw)


def C(printed, fn, how, kind="outcome", **kw):
    """A value computed from snapshot entries (every read goes through g)."""
    return Num(printed, kind, how, get=fn, **kw)


def K(printed, how, kind="protocol"):
    """A constant: a design choice, a citation or notation. Not read from the snapshot."""
    return Num(printed, kind, how)


class Entry:
    def __init__(self, where, text, *nums, runs=None):
        self.where, self.text, self.nums, self.runs = where, text, list(nums), runs


E = Entry
F10 = "frontier/idm/arms/uncached 10 steps"
N10 = "frontier/native/arms/uncached 10 steps"
F2, N2 = "frontier/idm/arms/uncached 2 steps", "frontier/native/arms/uncached 2 steps"
F1, N1 = "frontier/idm/arms/uncached 1 step", "frontier/native/arms/uncached 1 step"
FT8, NT8 = "frontier/idm/arms/10 steps, cache tau=8", "frontier/native/arms/10 steps, cache tau=8"
FT0, NT0 = "frontier/idm/arms/10 steps, cache tau=0", "frontier/native/arms/10 steps, cache tau=0"
FT2, FT4 = "frontier/idm/arms/10 steps, cache tau=2", "frontier/idm/arms/10 steps, cache tau=4"
FATT = "frontier/idm/arms/10 steps, attenuated cache tau=8"
PC = "primary_contrasts[label={}]"
CAM = "followup/camera/cam0015"
HO, HS = "heldout/shift", "heldout/streams"                 # results/budget_shift/heldout_a6000, repro/experiments/heldout.sh
IDLE_J = "idle_host/jitter/idm"      # results/budget_shift/idle_a6000, repro/experiments/idle_host.sh
SW = "shift_sweep[ckpt={}&shift={}&prefix={}]"
FVF = "noise/fixed_vs_fresh[arm={}]"


def _fresh_losses(S, lo, hi=1.0):
    """Points lost to fresh noise by the tau=8 caches whose share lies in [lo, hi]."""
    rows = g(S, "noise/fixed_vs_fresh")
    return [(-r["fresh_minus_fixed"]["diff"], r["fresh_minus_fixed"]["p"], arm_share(r["arm"]))
            for r in rows if lo - 1e-9 <= arm_share(r["arm"]) <= hi + 1e-9 and arm_share(r["arm"]) > 0]


def _fixed_drops(S):
    """(share, sigma where fresh steps resume, points lost against the uncached control) per tau=8 cache."""
    rows = g(S, "shift_sweep")
    ctrl = {(r["ckpt"], r["shift"]): r["success"]["pct"] for r in rows if r["prefix"] == 0}
    return [(r["fraction"], round(1 - r["fraction"], 4), ctrl[(r["ckpt"], r["shift"])] - r["success"]["pct"])
            for r in rows if r["prefix"] > 0 and (r["ckpt"], r["shift"]) in ctrl]


def _moved_later(S):
    """Trajectory changes in the process-sharing pilot, for the arms that did not run first."""
    return [a["steps_changed"] for a in g(S, "process_sharing/reseed_on/arms") if a["position"] != [0, 0]]


def _two_vs_ten_elsewhere(S):
    """|2 - 10 steps| in points, under fresh noise and on LIBERO-Long."""
    paths = ["noise/contrasts_under_fresh_noise[label=idm_s1_stock2_rn vs idm_s1_stock10_rn]/diff",
             "noise/contrasts_under_fresh_noise[label=uncond_s5_stock2_rn vs uncond_s5_stock10_rn]/diff",
             "libero_long/idm/contrasts[label=idm: 2 vs 10 steps]/diff",
             "libero_long/uncond/contrasts[label=uncond: 2 vs 10 steps]/diff"]
    return max(abs(g(S, p)) for p in paths)


def _composed_vs_two(S, ck):
    a = "idm" if ck == "idm" else "uncond"
    return 100 * (1 - g(S, f"matched_composed/arms/{a}_whole_2step_p1_tau4/latency/ep_mean")
                  / g(S, f"matched_composed/arms/{a}_stock_2step/latency/ep_mean"))


def _xhost(S):
    return [x["ratio"] for x in g(S, "latency_across_hosts")] + \
           [x["ratio"] for x in g(S, "followup/latency_vs_frontier_host")]


def _reuse_refresh(S, key):
    return [x[key] for x in g(S, "followup/composed_reuse_over_refresh")]


IDLE_T = "idle_host/timing"      # results/budget_shift/idle_a6000, repro/experiments/idle_host.sh


def _rr_all(S):
    """reuse/refresh ratios on every machine that timed the composed cache: (host, ratio)."""
    return [(x["host"], x["reuse_over_refresh"]) for x in g(S, "followup/composed_reuse_over_refresh")] + \
           [("idle A6000", x["reuse_over_refresh"]) for x in g(S, f"{IDLE_T}/rounds")]


def _idle_round_saving(S, pick):
    """Saving of one idle-machine round against its two uncached steps, in percent."""
    return 100 * (1 - pick(x["ratio"] for x in g(S, f"{IDLE_T}/rounds")))


SCR = "screen"


def _drift(S, fresh):
    """Per screening cell: the stored residual's relative difference from the current call's,
    averaged over the cached steps 0-7 of tau = 8 reuse calls (screen/cells/*/drift8_by_step)."""
    return [sum(c["drift8_by_step"][:8]) / 8 for c in g(S, f"{SCR}/cells").values() if c["fresh_noise"] == fresh]


def _equal_share(S):
    """(share, success %) for Optional-IDM caches near 54-64% of the path on four grids:
    s=1 with 6 cached steps, s=2 with 7, s=3 with 8, s=5 with 9 (shift_sweep, heldout)."""
    pts = [(g(S, f"{SW.format('idm', s_, m_)}/fraction"), g(S, f"{SW.format('idm', s_, m_)}/success/pct"))
           for s_, m_ in ((1.0, 6), (3.0, 8), (5.0, 9))]
    pts.append((g(S, f"{HO}/share_p7"), g(S, f"{HO}/p7/pct")))
    return pts


def _val_err(S):
    return [abs(v["pred_D"] - v["observed"]["pct"]) for v in g(S, f"{SCR}/validation")]


_MIXED = ("idm_s1_p7_tau8 vs idm_s1_stock10", "uncond_s1_p7_tau8 vs uncond_s1_stock10", "idm_s3_p9_tau8 vs idm_s3_stock10")


def _idle_round(S, i):
    """Saving of round i (0-based) of the idle machine's timing, in percent."""
    return 100 * (1 - g(S, f"{IDLE_T}/rounds")[i]["ratio"])


def _wilson95(k, n):
    """95% Wilson score interval for k successes in n, as fractions."""
    z = 1.959964
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h


def _contains(k, n, x):
    lo, hi = _wilson95(k, n)
    return lo <= x <= hi


def _idle_two_step_spread(S):
    """How much the idle machine's two-step calls varied between rounds, in percent."""
    v = [x["stock2_ms"] for x in g(S, f"{IDLE_T}/rounds")]
    return 100 * (max(v) - min(v)) / min(v)


def _within_after_fresh(S):
    """Caches whose fresh-noise success stays within 8 points of their fixed-noise success."""
    return [sh for loss, _, sh in _fresh_losses(S, 0.0) if abs(loss) <= 8]


def _small_loss_share(S):
    """Largest cached share below the first share where a cache lost more than two points (fixed noise)."""
    drops = _fixed_drops(S)
    first_big = min(sh for sh, _, d in drops if d > 2)
    return max(sh for sh, _, d in drops if sh < first_big - 1e-9)


def _drops_between(S, lo, hi):
    """Points lost by the tau=8 caches whose share lies in [lo, hi] (fixed noise)."""
    return [d for sh, _, d in _fixed_drops(S) if lo - 1e-9 <= sh <= hi + 1e-9]


def _composed_vs_two_followup(S):
    """Saving of the composed cache against two steps in the follow-up machine's undisturbed rounds."""
    return 100 * (1 - max(g(S, "followup/timing/ratios/{comp4/stock2}/per_round")))


def _composed_cost(S):
    """Points of success the composed cache gave up against two uncached steps, on the same host."""
    return max(-g(S, "matched_composed/contrasts/idm_tau4_vs_2step/diff"),
               -g(S, "matched_composed/contrasts/native_tau4_vs_2step/diff"))


def _ceil3(x):
    return "%.3f" % (math.ceil(x * 1000 - 1e-9) / 1000)


CAMERA = "the camera jitter chosen by the pilot: perturb=cam:0.015 (metres), followup.sh @hard"
MC = "matched_composed/contrasts/{}"
SC = "followup/share_contrasts[label={}]"


def _small_two_vs_ten_lower(S):
    """Tango 90% lower bounds of the 50-state two-vs-ten comparisons outside Table 1:
    camera jitter (both checkpoints), fresh noise (both) and LIBERO-Long (both)."""
    cam = {c["label"]: c for c in g(S, f"{CAM}/contrasts")}
    nc = {c["label"]: c for c in g(S, "noise/contrasts_under_fresh_noise")}
    lg = g(S, "libero_long")
    return [cam["cam0015: idm_stock2 vs idm_stock10"]["tango90"][0], cam["cam0015: uncond_stock2 vs uncond_stock10"]["tango90"][0],
            nc["idm_s1_stock2_rn vs idm_s1_stock10_rn"]["tango90"][0], nc["uncond_s5_stock2_rn vs uncond_s5_stock10_rn"]["tango90"][0],
            lg["idm"]["contrasts"][0]["tango90"][0], lg["uncond"]["contrasts"][0]["tango90"][0]]



LD = "latent_delay"       # results/budget_shift/latent_delay_a6000, repro/experiments/latent_delay.sh


def _delay5_losses(S):
    """Points that delay 5 cost two and ten uncached steps against delay 2."""
    return [-g(S, f"{LD}/delay/two_d5_vs_two_d2/diff"), -g(S, f"{LD}/delay/ten_d5_vs_ten_d2/diff")]


def _latent_ok(S):
    """P9 and P10 held, and the per-episode latent at least matched the fixed latent at both shares."""
    return (g(S, f"{LD}/predictions/P9/met") and g(S, f"{LD}/predictions/P10/met")
            and all(g(S, f"{LD}/latent/p{m}/episode_vs_fixed/diff") >= 0 for m in (6, 8)))


def _cell_drift(S, cell):
    """Mean drift of the tau = 8 store over cached steps 0-7, for one screening cell."""
    d = g(S, f"{SCR}/cells/{cell}/drift8_by_step")
    return sum(d[:8]) / 8


def _composed_ceiling_followup(S):
    """The composed cache's largest saving at tau = 4 with the follow-up machine's two-point fit, in percent."""
    return 100 * 0.75 / (g(S, "followup/cost_two_point/fixed_in_step_costs") + 2)


GR = "delay_graphs"       # results/budget_shift/delay_graphs_a6000, repro/experiments/delay_graphs.sh


def _rate_ld(S, k):
    return g(S, f"{LD}/delay/rates/{k}/pct")


def _delay5_cost(S):
    """Points that delay 5 cost two and ten uncached steps against no delay (Optional-IDM)."""
    return [_rate_ld(S, "two_d0") - _rate_ld(S, "two_d5"), _rate_ld(S, "ten_d0") - _rate_ld(S, "ten_d5")]


def _short_delay_cost(S, ten_only=False):
    """Points lost against no delay at delays 2-4, over every arm run (both checkpoints)."""
    gi, go = g(S, f"{GR}/idm"), g(S, f"{GR}/orig")
    out = [_rate_ld(S, "ten_d0") - _rate_ld(S, "ten_d2"), -gi["ten_d3_vs_ten_d0"]["diff"],
           gi["ten_d0"]["pct"] - gi["ten_d4"]["pct"], -go["ten_d3_vs_ten_d0"]["diff"]]
    return out if ten_only else out + [_rate_ld(S, "two_d0") - _rate_ld(S, "two_d2")]


def _pt(S, ck):
    return g(S, f"per_task/{ck}")


def _tau8_split(S):
    """Optional-IDM's tau = 8 cache by task: (low tasks' successes, losses on the rest) against ten steps."""
    p = _pt(S, "idm")
    c8, c10 = p["success"]["10 steps, cache tau=8"], p["success"]["uncached 10 steps"]
    low = [x for x in c8 if x <= 5]
    rest = [y - x for x, y in zip(c8, c10) if x > 5]
    return low, rest


ENTRIES = [
    # ------------------------------------------------------------ abstract
    E("Abstract", "Fast-WAM, a 6B-parameter world action model",
      K("6", "Fast-WAM: 5B video transformer + 1B action transformer [fastwam2026]", "cited")),
    E("Abstract", "Two uncached steps keep the success of ten: on all 500 initial states they are non-inferior within two percentage points, for one checkpoint and noise seed",
      V("500", "earlier/statescale/full_vs_2/n", "protocol"),
      C("two", lambda S: 2 if max(abs(x) for x in g(S, "earlier/statescale/full_vs_2/tango90")) < 2 else None,
        "earlier/statescale/full_vs_2/tango90 lies within +-2 points", "outcome", word=True)),
    E("Abstract", "Under PyTorch's default compile mode, they also cut a replanning call on an RTX A6000 from 202 to 92 ms",
      V("202", f"{N10}/latency/ep_mean", "latency"), V("92", f"{N2}/latency/ep_mean", "latency")),
    E("Abstract", "A cache that keeps two fresh steps cannot be faster than two uncached steps: every call still encodes the images, runs the video expert and computes those two steps",
      C("cannot be faster", lambda S: "cannot be faster" if g(S, f"{FT8}/latency/reuse_mean") >= g(S, f"{F2}/latency/ep_mean")
        and g(S, f"{NT8}/latency/reuse_mean") >= g(S, f"{N2}/latency/ep_mean") else "can be faster",
        "a reuse call of the tau=8 cache costs at least a two-step call, on both checkpoints (Table 1)", "latency")),
    E("Abstract", "Caching the first of two steps, refreshed every fourth call, saves at most 11–14% of a call by our cost model",
      C("11", lambda S: 100 * max(g(S, "latency_model/idm/composed_tau4_max_saving"), g(S, "latency_model/native/composed_tau4_max_saving")),
        "latency_model/{idm,native}/composed_tau4_max_saving (Table 1 machine, tau = 4)", "latency"),
      C("14", _composed_ceiling_followup, "0.75 / (F/a + 2) with followup/cost_two_point/fixed_in_step_costs (tau = 4)", "latency")),
    E("Abstract", "under Fast-WAM's current time shift, C3ache's eight cached steps cover 80% of the path instead of 44%, and the cached policy succeeds on 63% of states against 95% uncached",
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100),
      V("44", f"{SW.format('native', 5.0, 8)}/fraction", "formula", scale=100),
      V("63", f"{FT8}/success/pct"), V("95", f"{F10}/success/pct")),
    E("Abstract", "Resampling the noise on every call cost caches covering 60% or more of the path 14–40 points",
      C("60", lambda S: 100 * min(sh for l, _, sh in _fresh_losses(S, 0.0) if l > 8), "smallest share that loses more than 8 points to fresh noise", "formula"),
      C("14", lambda S: min(l for l, _, _ in _fresh_losses(S, 0.6)), "smallest fresh-noise loss at shares >= 60% (noise/fixed_vs_fresh)"),
      C("40", lambda S: max(l for l, _, _ in _fresh_losses(S, 0.6)), "largest fresh-noise loss at shares >= 60%")),
    # ------------------------------------------------------------ introduction
    E("1 Introduction", "A ten-step replanning call of Fast-WAM takes about 200 ms on a desktop GPU, four control steps of a 20 Hz robot",
      C("200", lambda S: 100 * round(g(S, f"{N10}/latency/ep_mean") / 100), "ten-step call on the Table 1 host (202 ms), to the nearest 100 ms", "latency"),
      C("four", lambda S: round(g(S, f"{N10}/latency/ep_mean") / 50), "the ten-step call in 50 ms control steps, rounded (202 / 50)", "latency", word=True),
      K("20", "LIBERO control rate [libero2023]", "cited")),
    E("1 Introduction", "reports up to 2.5× faster inference", K("2.5", "C3ache [c3ache2026]", "cited")),
    E("1 Introduction", "already reach 95.8% success on LIBERO-Goal", K("95.8", "Yang et al. [sbintuitions2026]", "cited")),
    E("1 Introduction", "a share that Fast-WAM's latest time shift raised from 44% to 80% for eight cached steps, and the cached policy falls to 63% success",
      V("44", f"{SW.format('native', 5.0, 8)}/fraction", "formula", scale=100),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100),
      V("63", f"{FT8}/success/pct")),
    # ------------------------------------------------------------ setup
    E("Fig. 1 caption", "on 100 LIBERO-Goal states (one RTX A6000). Blue: uncached, 1–10 steps",
      V("100", f"{F2}/success/n", "protocol"), K("one", "a single machine (Table 1 host)", "protocol"),
      V("1", "latency_model/idm/steps[0]", "protocol"), V("10", "latency_model/idm/steps[3]", "protocol")),
    E("Fig. 1 caption", "squares: 8 of 10 steps cached; diamond: attenuated; triangle: 2 of 4 cached). Bars: 95% Wilson intervals",
      K("8", "whole:TAU caches all but the last two steps (budget_shift.sh; meta.cache_prefix of each run)"),
      K("10", "EVALUATION.num_inference_steps=10"), K("2", "4-step cache: prefix 2 (meta.cache_prefix)"),
      K("4", "EVALUATION.num_inference_steps=4"), K("95", "report_budget_shift.wilson (z = 1.96)")),
    E("2 Setup", "pairs a 5B video transformer with a 1B action transformer",
      K("5", "[fastwam2026]; a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited"),
      K("1", "[fastwam2026]; a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited")),
    E("2 Setup", "denoises a chunk of 32 actions with the 30-block action expert",
      K("32", "upstream configs/data/libero_2cam.yaml: num_frames 33, i.e. 32 actions; a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited"),
      K("30", "upstream configs/model/*.yaml: action expert num_layers 30; a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited")),
    E("2 Setup", "The robot executes 10 of them (0.5 s at 20 Hz)",
      K("10", "upstream configs/sim_libero.yaml: replan_steps 10 (meta.replan of every run); a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited"),
      K("0.5", "10 actions at 20 Hz", "formula"), K("20", "LIBERO control rate [libero2023]", "cited")),
    E("2 Setup", "from noise (σ=1) to actions (σ=0)", K("1", "flow-matching convention", "definition"),
      K("0", "flow-matching convention", "definition")),
    E("2 Setup", "we call 1-σ_m the cache's share", K("1", "definition of the share", "definition")),
    E("2 Setup", "Both released LIBERO checkpoints use K=10",
      K("10", "upstream configs/train.yaml: eval_num_inference_steps 10; a regenerating host's identity gate re-reads it (gate*.json model_facts)", "cited")),
    E("2 Setup", "runs at s=5, the shift its release specifies",
      K("5", "upstream README: scheduler shift 5.0 for the original checkpoint (repro/experiments/baseline_audit.sh)", "cited")),
    E("2 Setup", "we run it in first-frame mode at s=1, the current shift (commit 0050a44)",
      K("1", "upstream configs/model/*.yaml: action_scheduler infer_shift 1.0", "cited"),
      K("0050", "upstream commit 0050a44 'adjust all action shift to 1.0'", "cited")),
    E("2 Setup", "over the whole 30-block stack", K("30", "upstream action expert num_layers 30", "cited")),
    E("2 Setup", "We cache steps 0–7 of 10, C3ache's longer range, and refresh them every τ calls, by default τ=8",
      K("0", "the longer of C3ache's two cache-step ranges, [0,7] (the other is [0,6])"), K("7", "the same"), K("10", "K"),
      K("8", "whole:8 (budget_shift.sh)")),
    E("2 Setup", "With τ=0 the cache is filled only when an episode starts, C3ache's headline setting (2.51× on an A100)",
      K("0", "whole:0 (budget_shift.sh)"), K("2.51", "C3ache [c3ache2026]", "cited")),
    E("Appendix A", "The eight residuals of the default cache take 0.5 MB in bf16",
      C("0.5", lambda S: "%.1f" % ((max([x for x in (e.get("whole_cache_bytes") for e in g(S, "gates")) if x] or [8 * 32 * 1024 * 2])) / 2 ** 20),
        "8 cached steps x a [1, 32, 1024] stack residual x 2 bytes (bf16) = 0.5 MiB; the identity gate "
        "records the entry shape and, since this revision, gates[*]/whole_cache_bytes", "formula")),
    E("Appendix A", "record errors of at most 0.17%",
      C("0.17", lambda S: "%.2f" % (100 * max(e["whole_identity_err"] for e in g(S, "gates"))),
        "largest gates[*]/whole_identity_err (relative L2 error of the replayed actions)")),
    E("2 Setup", "on 10 initial states for each of the 10 LIBERO-Goal tasks (100 states), or on the first 5 per task (50 states)",
      K("10", "+PROBE.n_episodes=10 (budget_shift.sh @frontier100)"), K("10", "+PROBE.n_tasks=10"),
      V("100", f"{F2}/success/n", "protocol"), K("5", "+PROBE.n_episodes=5"),
      V("50", f"{SW.format('idm', 1.0, 0)}/success/n", "protocol")),
    E("2 Setup", "so rollouts nearly repeat across machines (7 of 11725 paired outcomes changed over thirteen machines)",
      V("7", "determinism_exhaustive/outcome_changes", "hosts"),
      V("11725", "determinism_exhaustive/paired_episodes", "hosts"),
      C("thirteen", lambda S: len(g(S, "determinism_exhaustive/hosts")), "determinism_exhaustive/hosts (count)", "hosts", word=True)),
    E("2 Setup", "non-inferiority within five points", K("five", "EQ_MARGIN = 5.0 (report_budget_shift.py)")),
    E("2 Setup", "a margin we fixed beforehand as roughly what 100 paired states resolve near 95% success. Since it lets a 5% failure rate double, we report each interval",
      V("100", f"{F2}/success/n", "protocol"),
      C("roughly", lambda S: "roughly" if 3 <= max(
          (b - a) / 2 for a, b in (g(S, PC.format(f"{c}: uncached 2 steps vs uncached 10 steps") + "/tango90") for c in ("idm", "native"))) <= 6
        else "not", "half-width of the two 2-vs-10 Tango 90% intervals on 100 states (4.1 and 3.8 points)", "outcome"),
      V("95", f"{F2}/success/pct"),
      C("5", lambda S: 100 - g(S, f"{F2}/success/pct"), "failure rate at the two-step success rate, 100 - 95", "outcome")),
    E("2 Setup", "Holm's correction covers McNemar tests of equal success for the eleven primary contrasts",
      C("eleven", lambda S: len(g(S, "primary_contrasts")), "primary_contrasts (count)", "protocol", word=True)),
    E("Table 1 caption", "on 100 LIBERO-Goal states (one RTX A6000), Optional-IDM (IDM) at s=1 and the original at s=5",
      V("100", f"{F2}/success/n", "protocol"), K("one", "a single machine (Table 1 host)", "protocol"),
      K("1", "EVALUATION.sigma_shift=1.0"), K("5", "EVALUATION.sigma_shift=5.0")),
    # ------------------------------------------------------------ section 3
    E("3 Fewer steps or a cache?", "Two uncached steps succeed on 95 and 96 of 100 states (Optional-IDM and original), against 95 and 95 with ten, at 94 and 92 ms per call instead of 202, and both are non-inferior within five points (Tango 90% intervals [-4.1, 4.1] and [-2.6, 5.0])",
      V("95", f"{F2}/success/k"), V("96", f"{N2}/success/k"), V("100", f"{F2}/success/n", "protocol"),
      V("95", f"{F10}/success/k"), V("95", f"{N10}/success/k"),
      V("94", f"{F2}/latency/ep_mean", "latency"), V("92", f"{N2}/latency/ep_mean", "latency"),
      V("202", f"{F10}/latency/ep_mean", "latency"),
      K("five", "EQ_MARGIN = 5.0"), K("90", "report_budget_shift.tango_ci(conf=0.90)"),
      V("-4.1", PC.format("idm: uncached 2 steps vs uncached 10 steps") + "/tango90[0]", fmt="%.1f"),
      V("4.1", PC.format("idm: uncached 2 steps vs uncached 10 steps") + "/tango90[1]", fmt="%.1f"),
      V("-2.6", PC.format("native: uncached 2 steps vs uncached 10 steps") + "/tango90[0]", fmt="%.1f"),
      V("5.0", PC.format("native: uncached 2 steps vs uncached 10 steps") + "/tango90[1]", fmt="%.1f")),
    E("3 Fewer steps or a cache?", "On all 500 initial states, in an earlier Optional-IDM collection that our 100 states reproduce exactly, two steps succeed on 96.6% and ten on 96.4% (90% interval [-1.4, 1.8])",
      V("500", "earlier/statescale/full_vs_2/n", "protocol"),
      V("100", "earlier/statescale_reproduced[0]/n", "protocol"),
      C("exactly", lambda S: "exactly" if sum(r["success_flips"] for r in g(S, "earlier/statescale_reproduced")) == 0
        else "%d outcome changes" % sum(r["success_flips"] for r in g(S, "earlier/statescale_reproduced")),
        "earlier/statescale_reproduced[*]/success_flips (0 of 100 for 2 and for 10 steps)", "hosts"),
      V("96.6", "earlier/statescale/full_vs_2/a_pct", fmt="%.1f"), V("96.4", "earlier/statescale/full_vs_2/b_pct", fmt="%.1f"),
      K("90", "tango_ci(conf=0.90)"),
      V("-1.4", "earlier/statescale/full_vs_2/tango90[0]", fmt="%.1f"), V("1.8", "earlier/statescale/full_vs_2/tango90[1]", fmt="%.1f")),
    E("Appendix A", "Yang reports 95.6% for one step on 500 episodes without naming the checkpoint. Our 93% with the original checkpoint lies within the sampling error of that figure on 100 states, while 88% with Optional-IDM does not; a single step is the same at every shift, so the gap lies in the checkpoint or the evaluation",
      K("95.6", "[sbintuitions2026]: naive one-step LIBERO-Goal", "cited"),
      K("500", "[sbintuitions2026]: 10 tasks x 50 trials per suite", "cited"),
      V("93", f"{N1}/success/pct"),
      C("within", lambda S: "within" if _contains(g(S, f"{N1}/success/k"), g(S, f"{N1}/success/n"), 0.956) else "outside",
        "95% Wilson interval of the original checkpoint's one-step success on 100 states contains 0.956", "outcome"),
      V("100", f"{N1}/success/n", "protocol"),
      V("88", f"{F1}/success/pct"),
      C("gap", lambda S: "gap" if not _contains(g(S, f"{F1}/success/k"), g(S, f"{F1}/success/n"), 0.956) else "no gap",
        "95% Wilson interval of Optional-IDM's one-step success on 100 states excludes 0.956", "outcome"),
      K("the same at every shift", "Eq. 1 with K=1: sigma_0 = 1 and sigma_1 = 0 at every s", "formula")),
    E("3 Fewer steps or a cache?", "though the closest, Optional-IDM's τ=2 cache (97 against 95 states, at 158 instead of 94 ms), could still be a few points better ([-6.9, 2.5])",
      K("2", "whole:2"), V("97", f"{FT2}/success/k"), V("95", f"{F2}/success/k"),
      V("158", f"{FT2}/latency/ep_mean", "latency"), V("94", f"{F2}/latency/ep_mean", "latency"),
      V("-6.9", PC.format("idm: uncached 2 steps vs 10 steps, cache tau=2") + "/tango90[0]", fmt="%.1f"),
      V("2.5", PC.format("idm: uncached 2 steps vs 10 steps, cache tau=2") + "/tango90[1]", fmt="%.1f")),
    E("3 Fewer steps or a cache?", "With the original checkpoint the τ=8 cache costs one point (C3ache reports none), while Optional-IDM degrades as refreshes grow rarer",
      K("8", "whole:8"),
      C("one", lambda S: g(S, f"{N10}/success/pct") - g(S, f"{NT8}/success/pct"), "uncached ten steps minus the tau=8 cache, original checkpoint", word=True),
      K("none", "[c3ache2026]", "cited"),
      C("degrades", lambda S: "degrades" if g(S, f"{FT2}/success/pct") > g(S, f"{FT4}/success/pct") > g(S, f"{FT8}/success/pct") > g(S, f"{FT0}/success/pct")
        else "does not degrade", "Optional-IDM tau=2 > 4 > 8 > never refreshed (Table 1)", "outcome")),
    E("3 Fewer steps or a cache?", "two uncached steps beat its τ=8 cache by 32 points (p<10^-4) and its attenuated version by 14 (p=0.02)",
      K("8", "whole:8"),
      V("32", PC.format("idm: uncached 2 steps vs 10 steps, cache tau=8") + "/diff"),
      C("<10^-4", lambda S: "<10^-4" if g(S, PC.format("idm: uncached 2 steps vs 10 steps, cache tau=8") + "/p_holm") < 1e-4
        else "=%.4f" % g(S, PC.format("idm: uncached 2 steps vs 10 steps, cache tau=8") + "/p_holm"),
        PC.format("idm: uncached 2 steps vs 10 steps, cache tau=8") + "/p_holm"),
      V("14", PC.format("idm: uncached 2 steps vs 10 steps, attenuated cache tau=8") + "/diff"),
      V("0.02", PC.format("idm: uncached 2 steps vs 10 steps, attenuated cache tau=8") + "/p_holm", fmt="%.2f")),
    E("3 Fewer steps or a cache?", "plus bookkeeping (100 vs. 94 ms), and a refresh call costs the full schedule (210 ms)",
      V("100", f"{FT8}/latency/reuse_mean", "latency"), V("94", f"{F2}/latency/ep_mean", "latency"),
      V("210", f"{FT8}/latency/refresh_mean", "latency")),
    E("3 Fewer steps or a cache?", "gives F ≈ 4.8a–4.9a, so about 70% of a two-step call is work no action cache reduces",
      V("4.8", "latency_model/native/fixed_in_step_costs", "latency", fmt="%.1f"),
      V("4.9", "latency_model/idm/fixed_in_step_costs", "latency", fmt="%.1f"),
      C("70", lambda S: 10 * round(10 * g(S, "latency_model/native/share_outside_denoising_at_2_steps")),
        "latency_model/{idm,native}/share_outside_denoising_at_2_steps (0.712, 0.706), to the nearest 10%", "latency")),
    E("3 Fewer steps or a cache?", "Upstream CUDA graphs cut two- and ten-step calls to 86 and 143 ms, while the fastest cache for the original checkpoint, which our implementation cannot run under graphs, takes 105 ms",
      V("86", "latency_model/native_cudagraphs/ms[1]", "latency"), V("143", "latency_model/native_cudagraphs/ms[3]", "latency"),
      C("fastest", lambda S: "fastest" if g(S, "frontier/native/arms/4 steps, cache tau=4/latency/ep_mean") == min(
          v["latency"]["ep_mean"] for k_, v in g(S, "frontier/native/arms").items() if "cache" in k_ and not k_.startswith("uncached"))
        else "not fastest", "smallest mean latency among the original checkpoint's cached arms in Table 1", "latency"),
      V("105", "frontier/native/arms/4 steps, cache tau=4/latency/ep_mean", "latency")),
    E("Appendix A", "where ten steps cost 2.5 times as much as two, our τ=8 cache ran 2.0× faster than ten steps, close to the 2.13× that C3ache reports",
      V("2.5", "followup/cost_two_point/ten_over_two", "latency", fmt="%.1f"), K("8", "whole:8"),
      C("2.0", lambda S: 1 / g(S, "followup/timing/ratios/{tau8/stock10}/mean"), "1 / followup/timing/ratios/{tau8/stock10}/mean", "latency", fmt="%.1f"),
      K("2.13", "[c3ache2026]", "cited")),
    E("3 Fewer steps or a cache?", "Serving the first of the two steps from a cache refreshed every fourth call (τ=4) skips one action-expert pass on three calls in four",
      K("4", "whole:4@prefix=1 with K=2"), K("three", "tau=4: three reuse calls per refresh"), K("four", "tau=4")),
    E("3 Fewer steps or a cache?", "The cost model, assuming a cached step costs nothing and the rest of the call is unchanged, bounds the saving at 11% of a call on the Table 1 machine (14% with the follow-up machine's fit, and 6% if both calls ran under CUDA graphs)",
      C("11", lambda S: 100 * max(g(S, "latency_model/idm/composed_tau4_max_saving"), g(S, "latency_model/native/composed_tau4_max_saving")),
        "latency_model/{idm,native}/composed_tau4_max_saving", "latency"),
      C("14", _composed_ceiling_followup, "0.75 / (F/a + 2) with followup/cost_two_point/fixed_in_step_costs", "latency"),
      V("6", "latency_model/native_cudagraphs/composed_tau4_max_saving", "latency", scale=100)),
    E("3 Fewer steps or a cache?", "It saved 10% of a call on an idle A6000 machine (95% interval [-1.8, 22.2]% over three rounds) and 2–9% on loaded ones, so neither difference is resolved",
      V("10", f"{IDLE_T}/saving_pct", "latency"),
      K("95", "t interval over rounds (report_budget_shift.py section 13)"),
      V("-1.8", f"{IDLE_T}/saving_ci95[0]", "latency", fmt="%.1f"),
      V("22.2", f"{IDLE_T}/saving_ci95[1]", "latency", fmt="%.1f"),
      C("three", lambda S: g(S, f"{IDLE_T}/n_rounds"), "idle_host/timing/n_rounds", "protocol", word=True),
      C("2", lambda S: min(_composed_vs_two(S, "idm"), _composed_vs_two(S, "native"), _composed_vs_two_followup(S)),
        "smallest saving on the shared and slow-CPU machines", "latency"),
      C("9", lambda S: max(_composed_vs_two(S, "idm"), _composed_vs_two(S, "native"), _composed_vs_two_followup(S)),
        "largest saving on the shared and slow-CPU machines", "latency"),
      C("neither", lambda S: "neither" if g(S, f"{IDLE_T}/saving_ci95")[0] <= 0 <= g(S, f"{IDLE_T}/saving_ci95")[1]
        and g(S, MC.format("idm_tau4_vs_2step") + "/tango90")[0] <= -5 <= 0 <= g(S, MC.format("idm_tau4_vs_2step") + "/tango90")[1]
        else "not neither", "the idle saving's 95% interval spans zero, and the success interval spans zero and reaches the 5-point margin", "outcome")),
    E("3 Fewer steps or a cache?", "This composed cache succeeded on 94 and 95 of 100 states against 95 and 96 for two uncached steps (Tango 90% interval [-5.0, 2.6] for both)",
      V("94", MC.format("idm_tau4_vs_2step") + "/a_pct"), V("95", MC.format("native_tau4_vs_2step") + "/a_pct"),
      V("100", "matched_composed/arms/idm_whole_2step_p1_tau4/rate/n", "protocol"),
      V("95", MC.format("idm_tau4_vs_2step") + "/b_pct"), V("96", MC.format("native_tau4_vs_2step") + "/b_pct"),
      K("90", "tango_ci(conf=0.90)"),
      V("-5.0", MC.format("idm_tau4_vs_2step") + "/tango90[0]", fmt="%.1f"),
      V("2.6", MC.format("idm_tau4_vs_2step") + "/tango90[1]", fmt="%.1f"),
      C("both", lambda S: "both" if g(S, MC.format("idm_tau4_vs_2step") + "/tango90") == g(S, MC.format("native_tau4_vs_2step") + "/tango90")
        else "differ", "the two checkpoints' Tango 90% intervals are identical (1 vs 2 discordant states each)", "outcome")),
    E("Appendix A", "refreshes put the τ=8 cache's p95 at 214 ms (mean 120 ms), against 95 ms for two uncached steps",
      K("8", "whole:8"), V("214", f"{NT8}/latency/cyc_p95", "latency"), V("120", f"{NT8}/latency/ep_mean", "latency"),
      V("95", f"{N2}/latency/cyc_p95", "latency")),
    E("3 Fewer steps or a cache?", "Two steps also clear a two-point margin on 250 states under 1.5 cm camera jitter (70.0 against 64.8%, [1.4, 9.3])",
      C("clear a two-point margin", lambda S: "clear a two-point margin" if g(S, f"{IDLE_J}/contrast/tango90")[0] > -2 else "miss a two-point margin", "the jitter interval's lower bound is above -2 as well", "outcome"),
      V("250", f"{IDLE_J}/contrast/n", "protocol"),
      C("1.5", lambda S: 100 * float(g(S, f"{CAM}/perturb").split(":")[1]), f"{CAM}/perturb (cam:0.015 m)", "protocol", fmt="%.1f"),
      V("70.0", f"{IDLE_J}/stock2/pct", fmt="%.1f"), V("64.8", f"{IDLE_J}/stock10/pct", fmt="%.1f"),
      V("1.4", f"{IDLE_J}/contrast/tango90[0]", fmt="%.1f"), V("9.3", f"{IDLE_J}/contrast/tango90[1]", fmt="%.1f")),
    E("3 Fewer steps or a cache?", "That is the firmer result, since Optional-IDM's 100-state interval falls to -6.5 without its most favorable task",
      C("firmer", lambda S: "firmer" if g(S, "earlier/statescale/full_vs_2/tango90")[0] > g(S, "per_task/idm/loo_lowest_bound") else "weaker",
        "the 500-state lower bound exceeds the lowest leave-one-task-out bound on 100 states", "outcome"),
      V("100", f"{F2}/success/n", "protocol"),
      V("-6.5", "per_task/idm/loo_lowest_bound", fmt="%.1f")),
    E("3 Fewer steps or a cache?", "In a synchronous loop the faster call shortens each chunk only 1.18-fold (594 against 702 ms)",
      C("1.18", lambda S: (500 + g(S, f"{F10}/latency/ep_mean")) / (500 + g(S, f"{F2}/latency/ep_mean")), "702 / 594 ms per chunk", "latency", fmt="%.2f"),
      C("594", lambda S: 500 + g(S, f"{F2}/latency/ep_mean"), "10 actions x 50 ms + the two-step call (94 ms)", "latency"),
      C("702", lambda S: 500 + g(S, f"{F10}/latency/ep_mean"), "10 actions x 50 ms + the ten-step call (202 ms)", "latency")),
    E("3 Fewer steps or a cache?", "Rarer refreshes raise the ceiling only toward 15–19%",
      C("15", lambda S: 100 / (min(g(S, "latency_model/idm/fixed_in_step_costs"), g(S, "latency_model/native/fixed_in_step_costs")) + 2),
        "1 / (F/a + 2) on the Table 1 machine: one pass of two saved on every call", "latency"),
      C("19", lambda S: 100 / (g(S, "followup/cost_two_point/fixed_in_step_costs") + 2), "the same with the follow-up machine's fit", "latency")),
    E("Appendix A", "Under fresh noise and on LIBERO-Long, two and ten steps differ by at most four points",
      C("four", _two_vs_ten_elsewhere, "largest |2 - 10 steps| under fresh noise (noise/contrasts_under_fresh_noise) and on LIBERO-Long (libero_long/*/contrasts)", word=True)),
    # ------------------------------------------------------------ section 4
    E("4 When does a cache lose success?", "succeeds on 9% of states with Optional-IDM at s=1 but on 93% with the original at s=5. To separate checkpoint from shift, we ran each checkpoint at both shifts with the default τ=8 cache on the first 50 states",
      V("9", f"{FT0}/success/pct"), K("1", "EVALUATION.sigma_shift=1.0"), V("93", f"{NT0}/success/pct"), K("5", "EVALUATION.sigma_shift=5.0"),
      K("8", "whole:8"), V("50", f"{SW.format('idm', 1.0, 0)}/success/n", "protocol")),
    E("4 When does a cache lose success?", "Moving Optional-IDM from s=1 to s=5 raises its cached success from 54 to 92%, and moving the original from s=5 to s=1 lowers it from 92 to 72%",
      K("1", "shift"), K("5", "shift"), V("54", f"{SW.format('idm', 1.0, 8)}/success/pct"), V("92", f"{SW.format('idm', 5.0, 8)}/success/pct"),
      K("5", "shift"), K("1", "shift"), V("92", f"{SW.format('native', 5.0, 8)}/success/pct"), V("72", f"{SW.format('native', 1.0, 8)}/success/pct")),
    E("4 When does a cache lose success?", "Both changes exceed the shift's effect on uncached success (at most two points), by 40 and 20 points (95% intervals [24, 56] and [6, 34])",
      C("two", lambda S: max(abs(g(S, f"{SW.format(c, 1.0, 0)}/success/pct") - g(S, f"{SW.format(c, 5.0, 0)}/success/pct")) for c in ("idm", "native")),
        "shift_sweep uncached 10 steps, s=1 against s=5, both checkpoints", word=True),
      V("40", "shift_interaction[label=idm: shift effect on the tau=8 cache minus on uncached 10-step]/did"),
      V("20", "shift_interaction[label=native: shift effect on the tau=8 cache minus on uncached 10-step]/did"),
      K("95", "percentile bootstrap, 10,000 resamples"),
      V("24", "shift_interaction[label=idm: shift effect on the tau=8 cache minus on uncached 10-step]/ci95[0]"),
      V("56", "shift_interaction[label=idm: shift effect on the tau=8 cache minus on uncached 10-step]/ci95[1]"),
      V("6", "shift_interaction[label=native: shift effect on the tau=8 cache minus on uncached 10-step]/ci95[0]"),
      V("34", "shift_interaction[label=native: shift effect on the tau=8 cache minus on uncached 10-step]/ci95[1]")),
    E("4 When does a cache lose success?", "cached at s=1, the original keeps 18 points more than Optional-IDM (72 vs. 54%, p=0.02), against two points uncached",
      K("1", "shift"),
      V("18", "checkpoint_at_s1[label=idm_s1_whole_tau8 vs uncond_s1_whole_tau8]/diff", sign=-1),
      V("72", "checkpoint_at_s1[label=idm_s1_whole_tau8 vs uncond_s1_whole_tau8]/b_pct"),
      V("54", "checkpoint_at_s1[label=idm_s1_whole_tau8 vs uncond_s1_whole_tau8]/a_pct"),
      V("0.02", "checkpoint_at_s1[label=idm_s1_whole_tau8 vs uncond_s1_whole_tau8]/p", fmt="%.2f"),
      V("two", "checkpoint_at_s1[label=idm_s1_stock10 vs uncond_s1_stock10]/diff", word=True)),
    E("Fig. 2 caption", "(τ=8, 50 states; 0% is uncached). Triangles: s=3. Crosses: fresh noise per call. Bars: 95% Wilson intervals",
      K("8", "whole:8"), V("50", f"{SW.format('idm', 1.0, 0)}/success/n", "protocol"), K("0", "share of an uncached policy", "definition"),
      K("3", "EVALUATION.sigma_shift=3.0 (followup.sh @share)"),
      K("95", "make_figure.wilson95 (z = 1.96); intervals recorded in figure_data.json")),
    E("4 When does a cache lose success?", "At s=5 the grid of Eq. 1 crowds its early points near σ=1, so caching steps 0–7 integrates only 44% of the flow path, while at s=1 the same steps integrate 80%",
      K("5", "shift"), K("1", "pure noise", "definition"), K("0", "C3ache's longer cache-step range, [0,7]"), K("7", "the same"),
      V("44", f"{SW.format('native', 5.0, 8)}/fraction", "formula", scale=100), K("1", "shift"),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100)),
    E("4 When does a cache lose success?", "a third shift, s=3, no fixed-noise cache covering 64% of the path or less lost more than two points",
      K("3", "EVALUATION.sigma_shift=3.0"),
      C("64", lambda S: 100 * _small_loss_share(S), "largest share below the first cache that lost more than two points (shift_sweep)", "formula"),
      C("two", lambda S: max(_drops_between(S, 0.0, _small_loss_share(S))), "largest loss at those shares (shift_sweep)", word=True)),
    E("4 When does a cache lose success?", "At 70–75% the losses were mixed (2–8 points, intervals reaching -18.4), and at 80% both checkpoints lost 22–42",
      C("70", lambda S: 100 * arm_share("idm_s1_p7_tau8"), "Eq. 1, s=1, 7 of 10", "formula"),
      C("75", lambda S: 100 * arm_share("idm_s3_p9_tau8"), "Eq. 1, s=3, 9 of 10", "formula"),
      C("2", lambda S: min(-g(S, SC.format(x) + "/diff") for x in _MIXED), "smallest loss among the 70-75% caches", "outcome"),
      C("8", lambda S: max(-g(S, SC.format(x) + "/diff") for x in _MIXED), "largest loss among the 70-75% caches", "outcome"),
      C("-18.4", lambda S: min(g(S, SC.format(x) + "/tango90")[0] for x in _MIXED),
        "lowest Tango 90% bound among the 70-75% caches (Optional-IDM, 70%)", "outcome", fmt="%.1f"),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100),
      C("22", lambda S: min(_drops_between(S, 0.8, 0.8)), "smallest loss at 80% (shift_sweep)"),
      C("42", lambda S: max(_drops_between(S, 0.8, 0.8)), "largest loss at 80% (shift_sweep)")),
    E("4 When does a cache lose success?", "The eight cached steps that fail at s=1 cover 57% at s=3 and succeed on 92% of states (19 to 0 discordant pairs)",
      K("eight", "prefix=8"), K("1", "shift"),
      V("57", f"{SW.format('idm', 3.0, 8)}/fraction", "formula", scale=100), K("3", "shift"),
      V("92", f"{SW.format('idm', 3.0, 8)}/success/pct"),
      V("19", SC.format("idm_s3_p8_tau8 vs idm_s1_whole_tau8") + "/a_only"),
      V("0", SC.format("idm_s3_p8_tau8 vs idm_s1_whole_tau8") + "/b_only")),
    E("4 When does a cache lose success?", "For an unseen shift, s=2, the observed losses (2 and 26 points at 54 and 82% of the path) agreed with predictions committed before the runs, but only as point estimates: 50 states establish neither bound",
      K("2", "EVALUATION.sigma_shift=2.0 (repro/experiments/heldout.sh)"),
      C("agreed", lambda S: "agreed" if g(S, f"{HO}/P1_within_4_points") and g(S, f"{HO}/P2_loss_at_least_20") else "disagreed",
        "heldout/shift/P1_within_4_points and P2_loss_at_least_20 (observed differences)", "outcome"),
      V("2", f"{HO}/contrast_p7/diff", sign=-1.0), V("26", f"{HO}/contrast_p9/diff", sign=-1.0),
      C("54", lambda S: 100 * g(S, f"{HO}/share_p7"), "Eq. 1 at s=2 after 7 of 10 steps (heldout/shift/share_p7)", "formula"),
      C("82", lambda S: 100 * g(S, f"{HO}/share_p9"), "Eq. 1 at s=2 after 9 of 10 steps (heldout/shift/share_p9)", "formula"),
      V("50", f"{HO}/contrast_p7/n", "protocol"),
      C("neither", lambda S: "neither" if g(S, f"{HO}/contrast_p7/tango90")[0] < -4 and g(S, f"{HO}/contrast_p9/tango90")[1] > -20
        and g(S, f"{HO}/contrast_p9/tango90")[1] < 0 else "not neither",
        "P1's interval reaches below -4 and P2's above -20, while P2's excludes zero", "outcome")),
    E("4 When does a cache lose success?", "In exploratory runs that resample it on every call, uncached policies stayed within two points but caches covering 60% or more of the path lost 14–40 (uncorrected p ≤ 0.039), and a pre-registered second stream repeated the 14-point loss at 60% (pooled p=0.006",
      C("two", lambda S: max(abs(r["fresh_minus_fixed"]["diff"]) for r in g(S, "noise/fixed_vs_fresh") if arm_share(r["arm"]) == 0),
        "largest |fresh - fixed| of the uncached arms (noise/fixed_vs_fresh)", word=True),
      C("60", lambda S: 100 * min(sh for l, _, sh in _fresh_losses(S, 0.0) if sh > 0.45), "smallest share among the losing fresh-noise caches", "formula"),
      C("14", lambda S: min(l for l, _, sh in _fresh_losses(S, 0.0) if sh > 0.45), "smallest fresh-noise loss at shares of 60% and more", "outcome"),
      C("40", lambda S: max(l for l, _, sh in _fresh_losses(S, 0.0) if sh > 0.45), "largest fresh-noise loss at shares of 60% and more", "outcome"),
      C("0.039", lambda S: max(p for l, p, sh in _fresh_losses(S, 0.0) if sh > 0.45), "largest McNemar p of those losses", "outcome", fmt="%.3f"),
      V("14", f"{HS}/seed7/contrast/diff", sign=-1.0),
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      V("0.006", f"{HS}/pooled/p_by_state", fmt="%.3f")),
    E("4 When does a cache lose success?", "The 60 and 80% caches then succeeded on 96 and 64% of states, against 96 and 54% with the run's fixed latent and 82 and 26% with fresh noise (p=0.065 and p<0.001 against fresh noise)",
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      C("80", lambda S: 100 * share(10, 8, 1.0), "Eq. 1, first 8 of 10 steps at s=1", "formula"),
      V("96", f"{LD}/latent/p6/episode/pct"), V("64", f"{LD}/latent/p8/episode/pct"),
      V("96", f"{LD}/latent/p6/fixed/pct"), V("54", f"{LD}/latent/p8/fixed/pct"),
      V("82", f"{LD}/latent/p6/fresh/pct"), V("26", f"{LD}/latent/p8/fresh/pct"),
      V("0.065", f"{LD}/latent/p6/episode_vs_fresh/p", fmt="%.3f"),
      C("<0.001", lambda S: "<0.001" if g(S, f"{LD}/latent/p8/episode_vs_fresh/p") < 0.001 else "=%.3f" % g(S, f"{LD}/latent/p8/episode_vs_fresh/p"),
        "latent_delay/latent/p8/episode_vs_fresh/p", "outcome")),
    E("4 When does a cache lose success?", "stored residuals drift by 56–66% between calls with a fixed latent",
      C("56", lambda S: 100 * min(_drift(S, False)), "smallest fixed-noise cell mean drift (screen/cells/*/drift8_by_step)", "outcome"),
      C("66", lambda S: 100 * max(_drift(S, False)), "largest fixed-noise cell mean drift", "outcome")),
    E("4 When does a cache lose success?", "65% with a per-episode latent",
      V("65", f"{LD}/screen_episode/drift8_steps0_7", scale=100)),
    E("4 When does a cache lose success?", "and 85–97% with fresh noise",
      C("85", lambda S: 100 * min(_drift(S, True)), "smallest fresh-noise cell mean drift", "outcome"),
      C("97", lambda S: 100 * max(_drift(S, True)), "largest fresh-noise cell mean drift", "outcome")),
    E("Appendix A", "Descriptively, with thresholds chosen after the runs and not among the predictions, all 16 configurations with D<0.13 kept at least 90% success and all six above 0.37 at most 72%",
      V("16", f"{SCR}/bands/below_0.13/n", "outcome"), K("0.13", "band edge, descriptive (screen/bands/note)", "definition"),
      V("90", f"{SCR}/bands/below_0.13/min_success_pct"),
      V("six", f"{SCR}/bands/above_0.37/n", word=True), K("0.37", "band edge, descriptive (screen/bands/note)", "definition"),
      V("72", f"{SCR}/bands/above_0.37/max_success_pct")),
    E("4 When does a cache lose success?", "Fits on the resulting action error and on the share predicted 61 and 62% for a held-out configuration that reached 20%",
      V("61", f"{SCR}/validation[1]/pred_D"), V("62", f"{SCR}/validation[1]/pred_share"), V("20", f"{SCR}/validation[1]/observed/pct")),
    # ------------------------------------------------------------ section 5
    E("5 Checklist", "Eight cached steps covered 44, 57 and 80% of the path at s=5, 3 and 1",
      V("44", f"{SW.format('native', 5.0, 8)}/fraction", "formula", scale=100), V("57", f"{SW.format('idm', 3.0, 8)}/fraction", "formula", scale=100),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100), K("5", "shift"), K("3", "shift"), K("1", "shift")),
    E("5 Checklist", "Fixed noise hid losses of 14–40 points",
      C("14", lambda S: min(l for l, _, _ in _fresh_losses(S, 0.6)), "smallest fresh-noise loss at shares >= 60%"),
      C("40", lambda S: max(l for l, _, _ in _fresh_losses(S, 0.6)), "largest fresh-noise loss at shares >= 60%")),
    E("5 Checklist", "Sharing a process changed 17–19 of 20 pilot trajectories",
      C("17", lambda S: min(_moved_later(S)), "process_sharing/reseed_on/arms/*/steps_changed, arms that did not run first", "hosts"),
      C("19", lambda S: max(_moved_later(S)), "the same, largest", "hosts"),
      V("20", "process_sharing/reseed_on/arms[arm=c3ache_tau4]/episodes", "protocol")),
    E("5 Checklist", "and latency varied 2.4-fold across machines",
      C("2.4", lambda S: max(_xhost(S)), "largest ratio in latency_across_hosts and followup/latency_vs_frontier_host", "hosts", fmt="%.1f")),
    E("Limitations", "one noise seed (one latent at 500 states)",
      V("500", "earlier/statescale/full_vs_2/n", "protocol")),
    E("Appendix A", "Caching each block separately instead of the whole stack kept attention running, so a reuse call took 146 against 101 ms",
      V("146", "latency/lat_idm_block_skip_tau8/reuse_mean", "latency"), V("101", "latency/idm_whole_agg_tau8/reuse_mean", "latency")),
    E("Appendix A", "In a synchronous loop, two steps finish a chunk 1.18 times faster than ten (594 against 702 ms at 20 Hz)",
      C("1.18", lambda S: (500 + g(S, f"{F10}/latency/ep_mean")) / (500 + g(S, f"{F2}/latency/ep_mean")),
        "702 / 594 ms per chunk", "latency", fmt="%.2f"),
      C("594", lambda S: 500 + g(S, f"{F2}/latency/ep_mean"), "10 actions x 50 ms + the two-step call (Optional-IDM, 94 ms)", "latency"),
      C("702", lambda S: 500 + g(S, f"{F10}/latency/ep_mean"), "10 actions x 50 ms + the ten-step call (202 ms)", "latency"),
      K("20", "LIBERO control rate", "cited")),
    E("3 Fewer steps or a cache?", "The delay d is fixed per configuration, the mean call time of Table 1 rounded up to 50 ms steps",
      K("50", "control period at 20 Hz", "formula")),
    E("3 Fewer steps or a cache?", "Two steps at delay 2 succeeded on 48 of 50 states and ten steps at delay 5 on 43, the direction we predicted, but the exact test does not resolve the gap (p=0.12)",
      C("2", lambda S: math.ceil(g(S, f"{F2}/latency/ep_mean") / 50), "the two-step call (94 ms) in 50 ms control steps, rounded up", "latency"),
      V("48", f"{LD}/delay/rates/two_d2/k"), V("50", f"{LD}/delay/rates/two_d2/n", "protocol"),
      C("5", lambda S: math.ceil(g(S, f"{F10}/latency/ep_mean") / 50), "the ten-step call (202 ms) in 50 ms control steps, rounded up", "latency"),
      V("43", f"{LD}/delay/rates/ten_d5/k"),
      C("the direction we predicted", lambda S: "the direction we predicted" if g(S, f"{LD}/predictions/P11/met") else "against our prediction",
        "latent_delay/predictions/P11", "outcome"),
      C("does not resolve", lambda S: "does not resolve" if g(S, f"{LD}/delay/two_d2_vs_ten_d5/p") >= 0.05 else "resolves", "exact McNemar p >= 0.05", "outcome"),
      V("0.12", f"{LD}/delay/two_d2_vs_ten_d5/p", fmt="%.2f")),
    E("3 Fewer steps or a cache?", "Against no delay, delay 5 cost two steps 12 points and ten steps 10 (p=0.11 and 0.18), and delays of 2–4, including the 3 that CUDA graphs give ten steps, changed ten-step success by at most 6",
      K("5", "delay=5"),
      C("12", lambda S: _delay5_cost(S)[0], "two steps: no delay minus delay 5"),
      C("10", lambda S: _delay5_cost(S)[1], "ten steps: no delay minus delay 5"),
      V("0.11", f"{LD}/delay/two_d5_vs_two_d0/p", fmt="%.2f"), V("0.18", f"{LD}/delay/ten_d5_vs_ten_d0/p", fmt="%.2f"),
      K("2", "delay=2"), K("4", "delay=4"),
      C("3", lambda S: math.ceil(g(S, "latency_model/native_cudagraphs/ms[3]") / 50), "143 ms in 50 ms control steps, rounded up", "latency"),
      C("6", lambda S: max(_short_delay_cost(S, ten_only=True)), "largest ten-step loss against no delay at delays 2-4 (both checkpoints)")),
    E("3 Fewer steps or a cache?", "non-inferior even at a two-point margin",
      C("two-point", lambda S: "two-point" if g(S, "earlier/statescale/full_vs_2/tango90")[0] > -2 else "wider",
        "earlier/statescale/full_vs_2/tango90 lower bound above -2", "outcome")),
    E("Appendix A", "at a three-point margin it still holds for the original checkpoint ([-2.6, 5.0]) but not for Optional-IDM on 100 states ([-4.1, 4.1]), whose 500-state interval, [-1.4, 1.8], clears even two points",
      C("holds", lambda S: "holds" if g(S, PC.format("native: uncached 2 steps vs uncached 10 steps") + "/tango90")[0] > -3
        and g(S, PC.format("idm: uncached 2 steps vs uncached 10 steps") + "/tango90")[0] <= -3 else "differs",
        "original's lower bound above -3, Optional-IDM's not", "outcome"),
      V("-2.6", PC.format("native: uncached 2 steps vs uncached 10 steps") + "/tango90[0]", fmt="%.1f"),
      V("5.0", PC.format("native: uncached 2 steps vs uncached 10 steps") + "/tango90[1]", fmt="%.1f"),
      V("100", f"{F2}/success/n", "protocol"),
      V("-4.1", PC.format("idm: uncached 2 steps vs uncached 10 steps") + "/tango90[0]", fmt="%.1f"),
      V("4.1", PC.format("idm: uncached 2 steps vs uncached 10 steps") + "/tango90[1]", fmt="%.1f"),
      V("500", "earlier/statescale/full_vs_2/n", "protocol"),
      V("-1.4", "earlier/statescale/full_vs_2/tango90[0]", fmt="%.1f"), V("1.8", "earlier/statescale/full_vs_2/tango90[1]", fmt="%.1f")),
    E("Appendix A", "Pooled over both seeds, the fresh-noise loss has p=0.003 with the 100 state-seed pairs as the unit and p=0.006 with the 50 initial states as the unit",
      V("0.003", f"{HS}/pooled/p", fmt="%.3f"), V("100", f"{HS}/pooled/n", "protocol"),
      V("0.006", f"{HS}/pooled/p_by_state", fmt="%.3f"), V("50", f"{HS}/pooled/states", "protocol")),
    E("Appendix A", "On every task, two and ten steps differ by at most two of ten states for either checkpoint",
      C("two", lambda S: max(abs(x) for ck in ("idm", "native") for x in _pt(S, ck)["two_minus_ten"]), "per_task/*/two_minus_ten, largest magnitude", word=True)),
    E("Appendix A", "the lowest Tango 90% bound of two against ten steps is -3.8 for the original checkpoint but -6.5 for Optional-IDM, reached when the task where two steps gained two states is the one left out",
      K("90", "tango_ci(conf=0.90)"),
      V("-3.8", "per_task/native/loo_lowest_bound", fmt="%.1f"), V("-6.5", "per_task/idm/loo_lowest_bound", fmt="%.1f"),
      C("two", lambda S: _pt(S, "idm")["two_minus_ten"][min(range(len(_pt(S, "idm")["loo_tango90_two_vs_ten"])),
                                                           key=lambda j: _pt(S, "idm")["loo_tango90_two_vs_ten"][j][0])],
        "two minus ten steps on the task whose omission gives the lowest bound", word=True)),
    E("Appendix A", "For Optional-IDM the 500-state comparison is the firmer evidence",
      V("500", "earlier/statescale/full_vs_2/n", "protocol")),
    E("Appendix A", "The τ=8 cache's losses with Optional-IDM come from five tasks, which it completes on 1–5 of ten states, while it loses at most two states on each of the other five",
      K("8", "whole:8"),
      C("five", lambda S: len(_tau8_split(S)[0]), "per_task/idm: tasks where the tau = 8 cache succeeds on at most 5 of 10", word=True),
      C("1", lambda S: min(_tau8_split(S)[0]), "fewest successes among those tasks"),
      C("5", lambda S: max(_tau8_split(S)[0]), "most successes among those tasks"),
      C("two", lambda S: max(_tau8_split(S)[1]), "largest loss against ten steps on the other tasks", word=True),
      C("five", lambda S: len(_tau8_split(S)[1]), "the other tasks (count)", word=True)),
    E("Table A8 caption", "Successes of 10 per LIBERO-Goal task behind Table 1 (one RTX A6000): two and ten uncached steps and the τ=8 cache, Optional-IDM (IDM) at s=1 and the original checkpoint at s=5",
      V("10", "per_task/idm/states_per_task[0]", "protocol"), K("one", "a single machine (Table 1 host)", "protocol"),
      K("8", "whole:8"), K("1", "shift"), K("5", "shift")),
    E("Appendix A", "Under upstream CUDA graphs, two- and ten-step calls take 86 and 143 ms on the Table 1 machine, delays of 2 and 3 control steps",
      V("86", "latency_model/native_cudagraphs/ms[1]", "latency"), V("143", "latency_model/native_cudagraphs/ms[3]", "latency"),
      C("2", lambda S: math.ceil(g(S, "latency_model/native_cudagraphs/ms[1]") / 50), "86 ms in 50 ms steps, rounded up", "latency"),
      C("3", lambda S: math.ceil(g(S, "latency_model/native_cudagraphs/ms[3]") / 50), "143 ms in 50 ms steps, rounded up", "latency")),
    E("Appendix A", "ran ten steps at delay 3 for both checkpoints and at delay 4 for Optional-IDM",
      K("3", "delay=3 (repro/experiments/delay_graphs.sh)"), K("4", "delay=4 (repro/experiments/delay_graphs.sh)")),
    E("Appendix A", "Ten steps with Optional-IDM then succeeded on 96, 96, 90, 94 and 86% of the 50 states at delays of 0, 2, 3, 4 and 5, and with the original checkpoint on 94% at delays 0 and 3",
      V("96", f"{GR}/idm/curve_ten/0"), V("96", f"{GR}/idm/curve_ten/2"), V("90", f"{GR}/idm/curve_ten/3"),
      V("94", f"{GR}/idm/curve_ten/4"), V("86", f"{GR}/idm/curve_ten/5"), V("50", f"{GR}/idm/ten_d3/n", "protocol"),
      K("0", "delay"), K("2", "delay"), K("3", "delay"), K("4", "delay"), K("5", "delay"),
      C("94", lambda S: g(S, f"{GR}/orig/ten_d0/pct") if g(S, f"{GR}/orig/ten_d0/pct") == g(S, f"{GR}/orig/ten_d3/pct") else None,
        "delay_graphs/orig/ten_d0 and ten_d3 (equal)"),
      K("0", "delay"), K("3", "delay")),
    E("Appendix A", "Against two steps at delay 2, ten steps at delay 3 lost 6 points with Optional-IDM (p=0.25) and none with the original checkpoint",
      K("2", "delay"), K("3", "delay"),
      V("6", f"{GR}/idm/two_d2_vs_ten_d3/diff"), V("0.25", f"{GR}/idm/two_d2_vs_ten_d3/p", fmt="%.2f"),
      C("none", lambda S: "none" if g(S, f"{GR}/orig/two_d2_vs_ten_d3/diff") == 0 else "some", "delay_graphs/orig/two_d2_vs_ten_d3/diff == 0", "outcome")),
    E("Appendix A", "Only delay 5 cost both schedules consistently",
      K("5", "delay=5"),
      C("consistently", lambda S: "consistently" if min(_delay5_cost(S)) >= 10 and max(_short_delay_cost(S)) <= 6 else "not consistently",
        "delay 5 cost both schedules at least ten points against no delay; delays 2-4 cost at most six", "outcome")),
    E("Appendix A", "By exact McNemar tests at the 5% level, these states resolve the per-episode latent's gain over fresh noise at 80% of the path and the cost of delay 5 against delay 2 for two steps, but not the gain at 60% or the P11 comparison, whose 90% Tango intervals exclude zero while their exact tests do not",
      K("5", "two-sided level of the exact McNemar tests", "protocol"),
      C("80", lambda S: 100 * share(10, 8, 1.0), "Eq. 1, first 8 of 10 steps at s=1", "formula"),
      C("resolve", lambda S: "resolve" if g(S, f"{LD}/latent/p8/episode_vs_fresh/p") < 0.05 and g(S, f"{LD}/delay/two_d5_vs_two_d2/p") < 0.05
        and g(S, f"{LD}/latent/p6/episode_vs_fresh/p") >= 0.05 and g(S, f"{LD}/delay/two_d2_vs_ten_d5/p") >= 0.05
        and g(S, f"{LD}/latent/p6/episode_vs_fresh/tango90")[0] > 0 and g(S, f"{LD}/delay/two_d2_vs_ten_d5/tango90")[0] > 0
        else "do not resolve as stated", "exact p < 0.05 for the first two, >= 0.05 for the last two, whose Tango 90% lower bounds exceed zero", "outcome"),
      K("5", "delay=5"), K("2", "delay=2"),
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      K("90", "tango_ci(conf=0.90)")),
    E("Appendix A", "All runs use the 50 states of Fig. 2 and Optional-IDM at s=1 unless noted",
      V("50", f"{LD}/latent/p6/episode/n", "protocol"), K("1", "shift")),
    E("Appendix A", "P9 predicted that this per-episode latent would keep the 60 and 80% caches at least eight points above fresh noise, and it did (+14 and +38)",
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      C("80", lambda S: 100 * share(10, 8, 1.0), "Eq. 1, first 8 of 10 steps at s=1", "formula"),
      K("eight", "P9 threshold (repro/experiments/latent_delay.sh)", "protocol"),
      C("it did", lambda S: "it did" if g(S, f"{LD}/predictions/P9/met") else "it did not", "latent_delay/predictions/P9/met", "outcome"),
      V("+14", f"{LD}/latent/p6/episode_vs_fresh/diff", fmt="%+.0f"), V("+38", f"{LD}/latent/p8/episode_vs_fresh/diff", fmt="%+.0f")),
    E("Appendix A", "P10 predicted success within ten points of the run's fixed latent, which held, at 80% only at its boundary (0 and +10)",
      K("ten", "P10 threshold (repro/experiments/latent_delay.sh)", "protocol"),
      C("held", lambda S: "held" if g(S, f"{LD}/predictions/P10/met") else "failed", "latent_delay/predictions/P10/met", "outcome"),
      C("80", lambda S: 100 * share(10, 8, 1.0), "Eq. 1, first 8 of 10 steps at s=1", "formula"),
      C("only at its boundary", lambda S: "only at its boundary" if abs(g(S, f"{LD}/latent/p8/episode_vs_fixed/diff")) == 10 else "inside it",
        "|80% per-episode minus fixed latent| equals the ten-point margin", "outcome"),
      V("0", f"{LD}/latent/p6/episode_vs_fixed/diff"), V("+10", f"{LD}/latent/p8/episode_vs_fixed/diff", fmt="%+.0f")),
    E("Appendix A", "the new chunk's actions d to 9 after that",
      K("9", "replan_steps - 1: actions 0-9 of each chunk are executed (fw_libero.delayed_actions)")),
    E("Appendix A", "The delays, 2 and 5 steps, are the two- and ten-step call times of Table 1 rounded up to 50 ms steps",
      C("2", lambda S: math.ceil(g(S, f"{F2}/latency/ep_mean") / 50), "94 ms in 50 ms control steps, rounded up", "latency"),
      C("5", lambda S: math.ceil(g(S, f"{F10}/latency/ep_mean") / 50), "202 ms in 50 ms control steps, rounded up", "latency"),
      K("50", "control period at 20 Hz", "formula")),
    E("Appendix A", "P11 predicted that two steps at delay 2 would succeed at least as often as ten at delay 5, and they did (48 against 43 of 50 states)",
      K("2", "delay=2"), K("5", "delay=5"),
      C("they did", lambda S: "they did" if g(S, f"{LD}/predictions/P11/met") else "they did not", "latent_delay/predictions/P11/met", "outcome"),
      V("48", f"{LD}/delay/rates/two_d2/k"), V("43", f"{LD}/delay/rates/ten_d5/k"), V("50", f"{LD}/delay/rates/two_d2/n", "protocol")),
    E("Appendix A", "including the original checkpoint's τ=8 cache, delayed 5 steps after refresh calls and 2 after reuse calls",
      K("8", "whole:8"), K("5", "delay=5/2 (latent_delay.sh)"), K("2", "delay=5/2 (latent_delay.sh)")),
    E("Table A7 caption", "given in points with its Tango 90% interval", K("90", "tango_ci(conf=0.90)")),
    E("Table A7 caption", "The last block uses the delays of CUDA-graph execution, 2 for two steps and 3 for ten",
      C("2", lambda S: math.ceil(g(S, "latency_model/native_cudagraphs/ms[1]") / 50), "86 ms in 50 ms steps, rounded up", "latency"),
      C("3", lambda S: math.ceil(g(S, "latency_model/native_cudagraphs/ms[3]") / 50), "143 ms in 50 ms steps, rounded up", "latency")),
    E("Table A7 caption", "(50 LIBERO-Goal states; Optional-IDM at s=1, the original checkpoint at s=5)",
      V("50", f"{LD}/delay/rates/two_d2/n", "protocol"), K("1", "shift"), K("5", "shift")),
    E("Table A7 caption", "the first d of the ten actions executed after a call come from the previous chunk; 5/2 is 5 after refresh calls and 2 after reuse calls",
      K("5", "delay=5/2"), K("2", "delay=5/2"), K("5", "delay=5/2"), K("2", "delay=5/2")),
    E("Appendix A", "all calls of each episode started from one latent, and the stored residuals differed from the current call's by 65%, against 63 and 85% for the same policy under a fixed latent and under fresh noise, with D at eight cached steps 0.40 against 0.40 and 0.52",
      C("all calls of each episode started from one latent", lambda S: "all calls of each episode started from one latent"
        if g(S, f"{LD}/screen_episode/one_latent_per_episode") and g(S, f"{LD}/screen_episode/distinct_latents") == g(S, f"{LD}/screen_episode/uncached/n")
        else "latents varied", "latent_delay/screen_episode: every call's latent digest equals its episode's, and the 50 episodes have distinct latents", "outcome"),
      V("65", f"{LD}/screen_episode/drift8_steps0_7", scale=100),
      C("63", lambda S: 100 * _cell_drift(S, "scr_idm_s1"), "screen/cells/scr_idm_s1/drift8_by_step, steps 0-7", "outcome"),
      C("85", lambda S: 100 * _cell_drift(S, "scr_idm_s1_rn"), "screen/cells/scr_idm_s1_rn/drift8_by_step, steps 0-7", "outcome"),
      V("0.40", f"{LD}/screen_episode/D/p8_tau8", fmt="%.2f"),
      V("0.40", f"{SCR}/cells/scr_idm_s1/D/p8_tau8", fmt="%.2f"),
      V("0.52", f"{SCR}/cells/scr_idm_s1_rn/D/p8_tau8", fmt="%.2f")),
    E("Appendix A", "which succeeded on 88% of states against 94% for its two uncached steps at delay 2 and 92% without delay",
      V("88", f"{LD}/cache_delay/rates/cache_d52/pct"), V("94", f"{LD}/cache_delay/rates/two_d2/pct"),
      K("2", "delay=2"), V("92", f"{LD}/cache_delay/rates/cache_sync/pct")),
    # ------------------------------------------------------------ Appendix A (the paper's supplementary results)
    E("Appendix A", "Table A1 lists the eleven contrasts that Holm's correction covers in Sec. 3, on the 100 LIBERO-Goal states of Table 1",
      C("eleven", lambda S: len(g(S, "primary_contrasts")), "primary_contrasts (count)", "protocol", word=True),
      V("100", f"{F2}/success/n", "protocol")),
    E("Appendix A", "Only the three contrasts involving the τ=8 cache and its attenuated version with Optional-IDM survive the correction",
      C("three", lambda S: sum(1 for c in g(S, "primary_contrasts") if c["p_holm"] < 0.05), "primary_contrasts with p_holm < 0.05", word=True),
      K("8", "whole:8")),
    E("Table A1 caption", "The eleven primary contrasts of Sec. 3 (100 LIBERO-Goal states)",
      V("100", f"{F2}/success/n", "protocol")),
    E("Appendix A", "Table A2 gives every point of Fig. 2. The s=3 points and the 70–75% shares come from a follow-up machine, which also reproduced the outcomes of 330 episodes",
      K("3", "shift"),
      C("70", lambda S: 100 * arm_share("idm_s1_p7_tau8"), "Eq. 1, s=1, 7 of 10", "formula"),
      C("75", lambda S: 100 * arm_share("idm_s3_p9_tau8"), "Eq. 1, s=3, 9 of 10", "formula"),
      V("330", "followup/determinism_total/paired_episodes", "hosts")),
    E("Appendix A", "Under fixed noise, no cache covering 64% of the path or less lost more than two points, the caches at 70–75% lost 2–8, and those at 80% lost 22–42",
      C("64", lambda S: 100 * _small_loss_share(S), "largest share below the first cache that lost more than two points (shift_sweep)", "formula"),
      C("two", lambda S: max(_drops_between(S, 0.0, _small_loss_share(S))), "largest loss at those shares (shift_sweep)", word=True),
      C("70", lambda S: 100 * arm_share("idm_s1_p7_tau8"), "Eq. 1, s=1, 7 of 10", "formula"),
      C("75", lambda S: 100 * arm_share("idm_s3_p9_tau8"), "Eq. 1, s=3, 9 of 10", "formula"),
      C("2", lambda S: min(_drops_between(S, 0.7, 0.75)), "smallest loss at 70-75% (shift_sweep)"),
      C("8", lambda S: max(_drops_between(S, 0.7, 0.75)), "largest loss at 70-75% (shift_sweep)"),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100),
      C("22", lambda S: min(_drops_between(S, 0.8, 0.8)), "smallest loss at 80% (shift_sweep)"),
      C("42", lambda S: max(_drops_between(S, 0.8, 0.8)), "largest loss at 80% (shift_sweep)")),
    E("Appendix A", "Under fresh noise, losses of 14 points or more appeared from a share of 60% on",
      C("14", lambda S: min(l for l, _, _ in _fresh_losses(S, 0.6)), "smallest fresh-noise loss at shares >= 60%"),
      C("60", lambda S: 100 * min(sh for l, _, sh in _fresh_losses(S, 0.0) if l > 8), "smallest share that loses more than 8 points to fresh noise", "formula")),
    E("Appendix A", "Two fresh steps at s=5 integrate 56% of the path, consistent with the original checkpoint's tolerance of even a cache filled once per episode (93% in Table 1)",
      K("5", "shift"), C("56", lambda S: 100 * (1 - g(S, f"{SW.format('native', 5.0, 8)}/fraction")), "sigma_8 at s=5 = 1 - cached share", "formula"),
      V("93", f"{NT0}/success/pct")),
    E("Table A2 caption", "Every point of Fig. 2 (τ=8, 50 LIBERO-Goal states)",
      K("8", "whole:8"), V("50", f"{SW.format('idm', 1.0, 0)}/success/n", "protocol")),
    E("Appendix A", "Table A3 compares two and ten uncached steps beyond the 100 states of Table 1",
      V("100", f"{F2}/success/n", "protocol")),
    E("Appendix A", "A pilot of ten episodes per magnitude chose the camera jitter: ten steps succeeded on 9, 9 and 5 of 10 episodes at 7.5, 10 and 15 mm",
      V("ten", "followup/camera_pilot/campilot_0075_idm_stock10/n", "protocol", word=True),
      V("9", "followup/camera_pilot/campilot_0075_idm_stock10/k"), V("9", "followup/camera_pilot/campilot_010_idm_stock10/k"),
      V("5", "followup/camera_pilot/campilot_015_idm_stock10/k"), V("10", "followup/camera_pilot/campilot_015_idm_stock10/n", "protocol"),
      K("7.5", "followup.sh @pilot: perturb=cam:0.0075 (m)"), K("10", "perturb=cam:0.01"), K("15", "perturb=cam:0.015")),
    E("Appendix A", "With a second seed, two steps succeed on 97.2% of all 500 LIBERO-Goal initial states, against 96.6% with the first",
      V("97.2", "earlier/statescale/7/a_pct", fmt="%.1f"), V("500", "earlier/statescale/7/n", "protocol"),
      V("96.6", "earlier/statescale/full_vs_2/a_pct", fmt="%.1f")),
    E("Appendix A", "On LIBERO-Long the τ=8 cache behaves as it does on LIBERO-Goal", K("8", "whole:8 (budget_shift.sh @long)")),
    E("Appendix A", "It keeps every state with the original checkpoint, which caches 44% of the path at s=5, and loses 18 points with Optional-IDM, which caches 80% at s=1 (p=0.004 against ten steps)",
      C("every", lambda S: "every" if g(S, "libero_long/uncond/whole_tau8/pct") == 100 else "%.0f%%" % g(S, "libero_long/uncond/whole_tau8/pct"),
        "libero_long/uncond/whole_tau8/pct (100%)"),
      V("44", f"{SW.format('native', 5.0, 8)}/fraction", "formula", scale=100), K("5", "shift"),
      V("18", "libero_long/idm/contrasts[label=idm: 10 steps vs tau=8 cache]/diff"),
      V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100), K("1", "shift"),
      V("0.004", "libero_long/idm/contrasts[label=idm: 10 steps vs tau=8 cache]/p", fmt="%.3f")),
    E("Table A3 caption", "Two against ten uncached steps beyond Table 1, with one step and the τ=8 cache where run. Success is in percent, and the interval is a Tango 90% interval",
      K("8", "whole:8"), K("90", "tango_ci(conf=0.90)")),
    E("Table A3 caption", "The 250-state jitter row ran on the idle machine and repeats every two- and ten-step outcome of the 50-state row on the states the two share",
      V("250", f"{IDLE_J}/contrast/n", "protocol"), V("50", f"{CAM}/rates/idm_stock10/n", "protocol"),
      C("repeats", lambda S: "repeats" if g(S, f"{IDLE_J}/reproduced_outcome_changes") == 0 else "does not repeat",
        f"{IDLE_J}/reproduced_outcome_changes (0 of {IDLE_J}/reproduced_episodes)", "hosts")),
    E("Appendix A", "ran the same calls about twice as slowly as the Table 1 machine, the idle machine 1.5 times as slowly",
      C("1.5", lambda S: g(S, f"{IDLE_T}/stock2_ms_mean") / g(S, f"{F2}/latency/ep_mean"),
        "idle_host/timing/stock2_ms_mean over the Table 1 machine's Optional-IDM two-step call", "hosts", fmt="%.1f"),
      C("twice", lambda S: "twice" if min(x["ratio"] for x in g(S, "followup/latency_vs_frontier_host")) >= 1.9 else "less than twice",
        "followup/latency_vs_frontier_host/*/ratio (1.95-2.36)", "hosts")),
    E("Appendix A", "At s=2, ten uncached steps succeeded on 48 of 50 states, the cache over the first seven steps on 47 (Tango 90% interval [-9.7, 5.1]), and the cache over the first nine on 35, losing 15 states and gaining 2 ([-38.5, -13.7])",
      K("2", "EVALUATION.sigma_shift=2.0 (repro/experiments/heldout.sh)"),
      V("48", f"{HO}/stock10/k"), V("50", f"{HO}/stock10/n", "protocol"), K("seven", "prefix=7"),
      V("47", f"{HO}/p7/k"), K("90", "tango_ci(conf=0.90)"),
      V("-9.7", f"{HO}/contrast_p7/tango90[0]", fmt="%.1f"), V("5.1", f"{HO}/contrast_p7/tango90[1]", fmt="%.1f"),
      K("nine", "prefix=9"), V("35", f"{HO}/p9/k"),
      V("15", f"{HO}/contrast_p9/b_only"), V("2", f"{HO}/contrast_p9/a_only"),
      V("-38.5", f"{HO}/contrast_p9/tango90[0]", fmt="%.1f"), V("-13.7", f"{HO}/contrast_p9/tango90[1]", fmt="%.1f")),
    E("Appendix A", "With run seed 7, the 60% cache succeeded on 45 states under fixed noise and 38 under fresh noise ([-25.4, -3.5]; p=0.065 on its own), against 48 and 41 with seed 42",
      K("7", "seed=7 (repro/experiments/heldout.sh)"),
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      V("45", f"{HS}/seed7/fixed/k"), V("38", f"{HS}/seed7/fresh/k"),
      V("-25.4", f"{HS}/seed7/contrast/tango90[0]", fmt="%.1f"), V("-3.5", f"{HS}/seed7/contrast/tango90[1]", fmt="%.1f"),
      V("0.065", f"{HS}/seed7/contrast/p", fmt="%.3f"),
      V("48", f"{HS}/seed42/fixed/k"), V("41", f"{HS}/seed42/fresh/k"), K("42", "seed=42 (budget_shift.sh default)")),
    E("Appendix A", "Table A5 gives the success and timing of the composed cache (K=2, step 0 cached, τ=4)",
      K("2", "EVALUATION.num_inference_steps=2"), K("0", "prefix=1"), K("4", "whole:4")),
    E("Appendix A", "reuse calls cost 83–87% of a refresh call on all four machines, a mean saving of 10–13% at τ=4",
      C("83", lambda S: 100 * min(r for _, r in _rr_all(S)), "followup/composed_reuse_over_refresh and idle_host/timing/rounds, reuse_over_refresh, smallest", "latency"),
      C("87", lambda S: 100 * max(r for _, r in _rr_all(S)), "the same, largest", "latency"),
      C("four", lambda S: len({h for h, _ in _rr_all(S)}), "machines in the same lists", "hosts", word=True),
      C("10", lambda S: 75 * (1 - max(r for _, r in _rr_all(S))), "mean saving at tau=4, (3/4)(1 - reuse/refresh), smallest", "latency"),
      C("13", lambda S: 75 * (1 - min(r for _, r in _rr_all(S))), "the same, largest", "latency"),
      K("4", "whole:4")),
    E("Appendix A", "against two uncached steps timed separately, the saving averaged 10% on the idle machine (5–14% in single rounds, while its two-step calls varied by 5% between rounds), 7–9% on the shared machine and 2% in two of three interleaved rounds on the follow-up machine",
      V("10", f"{IDLE_T}/saving_pct", "latency"),
      C("5", lambda S: _idle_round_saving(S, max), "1 - idle_host/timing/rounds/*/ratio, the round with the largest ratio", "latency"),
      C("14", lambda S: _idle_round_saving(S, min), "the same, the round with the smallest ratio", "latency"),
      C("5", _idle_two_step_spread, "idle_host/timing/rounds/*/stock2_ms, (largest - smallest) / smallest", "latency"),
      C("7", lambda S: _composed_vs_two(S, "idm"), "1 - matched_composed idm_whole_2step_p1_tau4 / idm_stock_2step (mean latency)", "latency"),
      C("9", lambda S: _composed_vs_two(S, "native"), "1 - matched_composed uncond_whole_2step_p1_tau4 / uncond_stock_2step", "latency"),
      C("2", _composed_vs_two_followup, "1 - followup/timing/ratios/{comp4/stock2}, rounds 1 and 3", "latency"),
      C("two", lambda S: sum(1 for r in g(S, "followup/timing/ratios/{comp4/stock2}/per_round") if round(100 * (1 - r)) == 2), "rounds with a 2% saving", "latency", word=True),
      C("three", lambda S: len(g(S, "followup/timing/ratios/{comp4/stock2}/per_round")), "rounds", "protocol", word=True)),
    E("Table A4 caption", "On the Table 1 machine F/a comes from a least-squares fit over 1–10 steps. On the follow-up machine it comes from the 2- and 10-step calls, because its 1-step arm",
      V("1", "latency_model/idm/steps[0]", "protocol"), V("10", "latency_model/idm/steps[3]", "protocol"),
      K("2", "followup/cost_two_point (2 and 10 steps)"), K("10", "followup/cost_two_point"), K("1", "tm1_idm_stock1")),
    E("Table A5 caption", "The composed cache (K=2, step 0 cached, τ=4). Success: on the 100 LIBERO-Goal states, where two uncached steps succeed on 95% (IDM) and 96% (orig.)",
      K("2", "EVALUATION.num_inference_steps=2"), K("0", "prefix=1"), K("4", "whole:4"),
      V("100", "matched_composed/arms/idm_whole_2step_p1_tau4/rate/n", "protocol"),
      V("95", MC.format("idm_tau4_vs_2step") + "/b_pct"), V("96", MC.format("native_tau4_vs_2step") + "/b_pct")),
    E("Table A5 caption", "Saving: mean saving per call at τ=4 implied by the two. vs. 2 steps: composed-cache latency",
      K("4", "whole:4"), K("2", "the two-step baseline")),
    E("Appendix A", "The planned re-runs, 47 comparisons over seven A6000 machines and five revisions of the harness, changed none of 2730 paired outcomes and 14 lengths",
      V("47", "followup/determinism_all/comparisons", "hosts"),
      C("seven", lambda S: len(g(S, "followup/determinism_all/hosts")), "followup/determinism_all/hosts (count)", "hosts", word=True),
      C("five", lambda S: len(g(S, "followup/determinism_all/harness_revisions")), "followup/determinism_all/harness_revisions (count)", "hosts", word=True),
      V("none", "followup/determinism_all/outcome_changes", "hosts", word=True),
      V("2730", "followup/determinism_all/paired_episodes", "hosts"),
      V("14", "followup/determinism_all/step_count_changes", "hosts")),
    E("Appendix A", "at most four points lost with the first seven of ten steps cached at s=2 (54% of the path) and at least 20 with the first nine (82%)",
      K("four", "prediction P1, repro/experiments/heldout.sh"), K("seven", "prefix=7"), K("ten", "EVALUATION.num_inference_steps=10"),
      K("2", "EVALUATION.sigma_shift=2.0 (repro/experiments/heldout.sh)"),
      C("54", lambda S: 100 * g(S, f"{HO}/share_p7"), "Eq. 1 at s=2 after 7 of 10 steps (heldout/shift/share_p7)", "formula"),
      K("20", "prediction P2, repro/experiments/heldout.sh"), K("nine", "prefix=9"),
      C("82", lambda S: 100 * g(S, f"{HO}/share_p9"), "Eq. 1 at s=2 after 9 of 10 steps (heldout/shift/share_p9)", "formula")),
    E("Appendix A", "replays that call's Euler loop from the same initial latent under each configuration (three to nine of ten steps cached at τ=8, and eight at τ=2, 4 and 0)",
      K("three", "screen_engine.DEFAULT_CONFIGS"), K("nine", "screen_engine.DEFAULT_CONFIGS"), K("ten", "EVALUATION.num_inference_steps=10"),
      K("8", "screen_engine.DEFAULT_CONFIGS"), K("eight", "screen_engine.DEFAULT_CONFIGS"), K("2", "screen_engine.DEFAULT_CONFIGS"),
      K("4", "screen_engine.DEFAULT_CONFIGS"), K("0", "screen_engine.DEFAULT_CONFIGS")),
    E("Appendix A", "On one episode, the replay at the first reuse call matched the action of a closed-loop τ=8 cache to the last bit, as the first call matched the cache's refresh",
      K("one", "repro/experiments/screen.sh @smoke (tasks=1, one episode)"), K("8", "smoke_whole8: whole:8"),
      C("to the last bit", lambda S: "to the last bit" if g(S, f"{SCR}/smoke/first_reuse_max_abs_diff") == 0 and g(S, f"{SCR}/smoke/first_call_max_abs_diff") == 0
        else "approximately", "screen/smoke: both maximum differences are zero", "outcome")),
    E("Appendix A", "the two differed by 0.013 at the next call and by up to 0.95 later. The screened rollouts execute only uncached actions and repeated all 400 outcomes of the uncached runs they duplicate. Replaying ten configurations took 2.4 times as long as the uncached call",
      V("0.013", f"{SCR}/smoke/later_max_abs_diff[1]", fmt="%.3f"),
      C("0.95", lambda S: max(g(S, f"{SCR}/smoke/later_max_abs_diff")), "largest replay-versus-cache difference in the smoke episode", "outcome", fmt="%.2f"),
      C("400", lambda S: sum(c["vs_shipped_uncached"]["paired"] for c in g(S, f"{SCR}/cells").values() if "vs_shipped_uncached" in c),
        "screen/cells/*/vs_shipped_uncached/paired, summed (0 outcome changes)", "protocol"),
      C("ten", lambda S: len(g(S, f"{SCR}/cells/scr_idm_s1/D")), "configurations screened per call", "protocol", word=True),
      C("2.4", lambda S: sum(c["replay_cost_ratio"] for c in g(S, f"{SCR}/cells").values()) / len(g(S, f"{SCR}/cells")),
        "mean of screen/cells/*/replay_cost_ratio (2.36-2.43)", "latency", fmt="%.1f")),
    E("Appendix A", "and all five held as stated",
      C("five", lambda S: sum(v["met"] for v in g(S, f"{SCR}/predictions").values()), "screen/predictions/*/met (count)", "outcome", word=True)),
    E("Appendix A", "At s=1 with eight cached steps, D was 0.401 for Optional-IDM and 0.378 for the original checkpoint",
      K("1", "shift"), K("eight", "prefix 8"),
      V("0.401", f"{SCR}/predictions/P4/idm", fmt="%.3f"), V("0.378", f"{SCR}/predictions/P4/original", fmt="%.3f")),
    E("Appendix A", "Fresh noise raised D at 4, 6, 7 and 8 cached steps (P5), and at eight cached steps D fell from 0.401 at s=1 to 0.287, 0.121 and 0.029 at s=2, 3 and 5",
      K("4", "P5, repro/experiments/screen.sh"), K("6", "P5"), K("7", "P5"), K("8", "P5"),
      C("raised", lambda S: "raised" if g(S, f"{SCR}/predictions/P5/met") else "did not raise", "screen/predictions/P5", "outcome"),
      K("eight", "P6"), V("0.401", f"{SCR}/predictions/P6/s1_s2_s3_s5[0]", fmt="%.3f"), K("1", "shift"),
      V("0.287", f"{SCR}/predictions/P6/s1_s2_s3_s5[1]", fmt="%.3f"), V("0.121", f"{SCR}/predictions/P6/s1_s2_s3_s5[2]", fmt="%.3f"),
      V("0.029", f"{SCR}/predictions/P6/s1_s2_s3_s5[3]", fmt="%.3f"), K("2", "shift"), K("3", "shift"), K("5", "shift")),
    E("Appendix A", "Over the 24 τ=8 configurations of Table ?, Spearman's ρ between D and success was -0.762, against -0.738 for the share (P7), and the bootstrap 90% interval of the difference over states, [-0.09, 0.04], includes zero",
      V("24", f"{SCR}/n_primary", "protocol"), K("8", "tau"),
      V("-0.762", f"{SCR}/rho_primary/D", fmt="%.3f"), V("-0.738", f"{SCR}/rho_primary/share", fmt="%.3f"),
      K("90", "report_screen.py: 5th and 95th percentiles of 2000 resamples"),
      V("-0.09", f"{SCR}/rho_primary/diff_boot90[0]", fmt="%.2f"), V("0.04", f"{SCR}/rho_primary/diff_boot90[1]", fmt="%.2f"),
      C("includes zero", lambda S: "includes zero" if g(S, f"{SCR}/rho_primary/diff_boot90")[0] <= 0 <= g(S, f"{SCR}/rho_primary/diff_boot90")[1]
        else "excludes zero", "screen/rho_primary/diff_boot90 spans zero", "outcome")),
    E("Appendix A", "which the share cannot tell apart, ρ was -0.64 against -0.60 over 31 configurations",
      V("-0.64", f"{SCR}/rho_all/D", fmt="%.2f"), V("-0.60", f"{SCR}/rho_all/share", fmt="%.2f"), V("31", f"{SCR}/rho_all/n", "protocol")),
    E("Appendix A", "A logistic regression on log D fitted to the 24 predicted the held-out configurations' success with a mean absolute error of 16.4 points, against 16.7 for the share",
      V("24", f"{SCR}/n_primary", "protocol"),
      V("16.4", f"{SCR}/predictions/P8/mae_D", fmt="%.1f"), V("16.7", f"{SCR}/predictions/P8/mae_share", fmt="%.1f")),
    E("Appendix A", "On the held-out states, the original checkpoint at s=1 under fresh noise succeeded on 92 and 20% with six and eight cached steps (96% uncached), and Optional-IDM at s=2 on 88% with seven (98% uncached)",
      K("1", "shift"), V("92", f"{SCR}/validation[0]/observed/pct"), V("20", f"{SCR}/validation[1]/observed/pct"),
      K("six", "prefix=6"), K("eight", "prefix 8"), V("96", f"{SCR}/validation_uncached/scr_uncond_s1_rn/pct"),
      K("2", "shift"), V("88", f"{SCR}/validation[2]/observed/pct"), K("seven", "prefix=7"),
      V("98", f"{SCR}/validation_uncached/scr_idm_s2_rn/pct")),
    E("Appendix A", "Under fixed noise at s=1 with eight cached steps, 27 and 28% of the two checkpoints' reuse calls inverted the gripper command on all ten executed actions, against at most 6% under fresh noise",
      K("1", "shift"), K("eight", "prefix 8"),
      V("27", f"{SCR}/cells/scr_idm_s1/grip_all/p8_tau8", scale=100), V("28", f"{SCR}/cells/scr_uncond_s1/grip_all/p8_tau8", scale=100),
      K("two", "checkpoints"), K("ten", "EVALUATION.replan_steps = 10"),
      C("6", lambda S: 100 * max(c["grip_all"]["p8_tau8"] for c in g(S, f"{SCR}/cells").values() if c["fresh_noise"]),
        "largest share of such calls in a fresh-noise cell at 8 cached steps", "outcome")),
    E("Appendix A", "At similar shares on different grids, success agreed below two-thirds of the path (92–96% at 54–64% for s=1, 2, 3 and 5) but not above it: caching 82% of the path at s=2 kept 70%, against 54% for 80% at s=1 and 90% for 75% at s=3",
      C("92", lambda S: min(sc for _, sc in _equal_share(S)), "lowest success of the four caches near 54-64%", "outcome"),
      C("96", lambda S: max(sc for _, sc in _equal_share(S)), "highest such success", "outcome"),
      C("54", lambda S: 100 * min(sh for sh, _ in _equal_share(S)), "smallest of their shares (s=2, 7 cached)", "formula"),
      C("64", lambda S: 100 * max(sh for sh, _ in _equal_share(S)), "largest of their shares (s=5, 9 cached)", "formula"),
      K("1", "shift"), K("2", "shift"), K("3", "shift"), K("5", "shift"),
      C("82", lambda S: 100 * g(S, f"{HO}/share_p9"), "Eq. 1 at s=2 after 9 of 10 steps", "formula"), K("2", "shift"),
      V("70", f"{HO}/p9/pct"),
      V("54", f"{SW.format('idm', 1.0, 8)}/success/pct"), V("80", f"{SW.format('idm', 1.0, 8)}/fraction", "formula", scale=100), K("1", "shift"),
      V("90", f"{SW.format('idm', 3.0, 9)}/success/pct"), V("75", f"{SW.format('idm', 3.0, 9)}/fraction", "formula", scale=100), K("3", "shift")),
    E("Appendix A", "whereas fixed noise reuses one latent for the whole run. A second fixed latent (seed 7) kept the 60% cache at 90%, near the 96% of seed 42",
      K("one", "seed= set: one fixed latent per run (fw_libero.py)", "definition"), K("7", "run seed 7 (heldout.sh @streams)", "protocol"),
      C("60", lambda S: 100 * share(10, 6, 1.0), "Eq. 1, first 6 of 10 steps at s=1", "formula"),
      V("90", f"{HS}/seed7/fixed/pct"), V("96", f"{HS}/seed42/fixed/pct"), K("42", "run seed 42 (default)", "protocol")),
    E("Table A6 caption", "on the 50 states of Fig. §. Cached: leading steps of ten served from the cache, refreshed every τ calls (τ=0: only when an episode starts)",
      V("50", f"{SW.format('idm', 1.0, 0)}/success/n", "protocol"), K("ten", "EVALUATION.num_inference_steps=10"), K("0", "whole:0")),
    E("Table A6 caption", "over the ten executed actions",
      K("ten", "EVALUATION.replan_steps = 10")),
    E("Table A6 caption", "on the other 50 states (initial states 5–9 of every task)",
      V("50", f"{SCR}/validation[0]/observed/n", "protocol"), K("5", "first_ep=5 (screen.sh @validate)"), K("9", "first_ep=5, 5 per task")),
    E("Fig. A1 caption", "for the 24 τ=8 configurations of Table ? (markers as in Fig. §; diamonds: s=2)",
      V("24", f"{SCR}/n_primary", "protocol"), K("8", "tau"), K("2", "shift")),
    E("Fig. A1 caption", "Stars: held-out configurations on the other 50 states. Bars: 95% Wilson intervals",
      V("50", f"{SCR}/validation[0]/observed/n", "protocol"), K("95", "make_figure.wilson95 (z = 1.96)")),
    E("Appendix A", "Over all 249 pairs of runs that share a configuration but not a machine (thirteen machines, eleven revisions), 7 of 11725 outcomes changed, on five initial states, and 117 lengths",
      V("249", "determinism_exhaustive/pairs", "hosts"),
      C("thirteen", lambda S: len(g(S, "determinism_exhaustive/hosts")), "determinism_exhaustive/hosts (count)", "hosts", word=True),
      C("eleven", lambda S: len(g(S, "determinism_exhaustive/harness_revisions")), "determinism_exhaustive/harness_revisions (count)", "hosts", word=True),
      V("7", "determinism_exhaustive/outcome_changes", "hosts"),
      V("11725", "determinism_exhaustive/paired_episodes", "hosts"),
      V("five", "determinism_exhaustive/states_changed", "hosts", word=True),
      V("117", "determinism_exhaustive/step_count_changes", "hosts")),
    E("Appendix A", "ran 21–41% slower on one machine and 1.95–2.36 times slower on two others",
      C("21", lambda S: 100 * (min(x["ratio"] for x in g(S, "latency_across_hosts")) - 1), "latency_across_hosts/*/ratio, smallest", "hosts"),
      C("41", lambda S: 100 * (max(x["ratio"] for x in g(S, "latency_across_hosts")) - 1), "the same, largest", "hosts"),
      C("1.95", lambda S: min(x["ratio"] for x in g(S, "followup/latency_vs_frontier_host")), "followup/latency_vs_frontier_host/*/ratio, smallest", "hosts", fmt="%.2f"),
      C("2.36", lambda S: max(x["ratio"] for x in g(S, "followup/latency_vs_frontier_host")), "the same, largest", "hosts", fmt="%.2f"),
      C("two", lambda S: len({x["host"] for x in g(S, "followup/latency_vs_frontier_host")}), "hosts in followup/latency_vs_frontier_host", "hosts", word=True)),
    E("Appendix A", "stops the collection if the replayed actions differ from uncached ones by 0.5% or more in relative L2 error",
      K("0.5", "MAX_IDENTITY_ERR = 5e-3 (probes/gate_cache.py); repro/run_experiment.sh runs the gate once per collection")),
]

# The tables and figures are generated whole; they are checked by regenerating
# them from the snapshot and comparing with the copies in repro/expected/.
GENERATED = [
    ("Table 1", "tables/frontier.tex"), ("Table A1", "tables/app_contrasts.tex"),
    ("Table A2", "tables/app_share.tex"), ("Table A3", "tables/app_regimes.tex"),
    ("Table A4", "tables/app_latency.tex"), ("Table A5", "tables/app_composed.tex"),
    ("Table A6", "tables/app_screen.tex"), ("Table A7", "tables/app_latent_delay.tex"),
    ("Table A8", "tables/app_per_task.tex"),
    ("Fig. 1, Fig. 2 and Fig. A1 (every plotted point)", "figure_data.json"),
]


# ---------------------------------------------------------------- checking
def _norm(printed):
    return printed.replace(",", "")


def evaluate(S):
    """[(key, entry, num, value or None, error or None)] for every number."""
    out = []
    for i, e in enumerate(ENTRIES):
        for j, n in enumerate(e.nums):
            try:
                out.append((f"{i}:{j}", e, n, n.value(S), None))
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as ex:
                out.append((f"{i}:{j}", e, n, None, f"{type(ex).__name__}: {ex}"))
    return out


def check(S, label, compare=None, runs=None, present=None):
    """Check every number against a snapshot.

    With a regenerated snapshot, `compare` (probes/compare_regen.py --json) says
    which regenerated runs changed an outcome against their own shipped copy, and
    `runs` (repro/paper_number_runs.json) which runs each number depends on. An
    outcome number that moved only where one of its runs flipped an episode is
    reported as moved, not as a failure: the machines that collected the paper
    flipped 7 of 11,720 paired outcomes among themselves. With `present` (the
    runs a partial regeneration collected), a number that also depends on runs
    it did not collect is reported as partial.
    """
    moved = {r: v for r, v in ((compare or {}).get("same_path") or {}).items() if v.get("outcome_changes")}
    deps = (runs or {}).get("numbers", {})
    bad, differ, missing, flipped, partial, n_checked = [], [], [], [], [], 0
    for key, e, n, v, err in evaluate(S):
        if n.get is None:
            continue
        n_checked += 1
        need = deps.get(key, {}).get("runs", [])
        if present is not None and need and not set(need) <= present:
            partial.append((e, n))
            continue
        if err:
            missing.append((e, n, err))
        elif _norm(v) != _norm(n.printed):
            hit = [r for r in deps.get(key, {}).get("runs", []) if r in moved]
            if n.kind in EXACT and hit:
                flipped.append((e, n, v, hit))
            else:
                (bad if n.kind in EXACT else differ).append((e, n, v))
    print(f"{label}: {n_checked} numbers read from the snapshot, "
          f"{sum(1 for e in ENTRIES for n in e.nums if n.get is None)} constants (protocol, cited, notation)")
    for e, n, v in bad:
        print(f"  MISMATCH  {e.where}: printed {n.printed!r}, snapshot gives {v!r}  [{n.kind}; {n.how}]")
    for e, n, v in differ:
        print(f"  differs   {e.where}: printed {n.printed!r}, this snapshot {v!r}  [{n.kind}: host-specific]")
    for e, n, v, hit in flipped:
        eps = sum(moved[r]["outcome_changes"] for r in hit)
        print(f"  moved     {e.where}: printed {n.printed!r}, this snapshot {v!r}  [{n.kind}: {eps} episode(s) of "
              f"{', '.join(hit)} changed outcome on this host]")
    for e, n, err in missing:
        print(f"  absent    {e.where}: {n.printed!r} [{n.kind}] -- {err}")
    ok = not bad
    print(f"  {'OK' if ok else 'FAILED'}: {len(bad)} unexplained mismatches in numbers that must reproduce, "
          f"{len(flipped)} moved by flipped episodes, {len(differ)} host-specific differences, "
          f"{len(missing)} not in this snapshot"
          + (f", {len(partial)} that also need runs this regeneration did not collect" if partial else ""))
    return ok


# ---------------------------------------------------------------- the reviewer's map
KIND_NOTE = {
    "outcome": "reproduces exactly on an RTX A6000",
    "latency": "same runs, host-specific value",
    "hosts": "tally over the shipped machines",
    "formula": "arithmetic (Eq. 1)",
    "protocol": "set by the experiment scripts",
    "cited": "cited or upstream config",
    "definition": "notation",
}
TIER_ORDER = ["frontier", "share", "followup", "regimes", "composed", "idle", "heldout", "screen", "latent_delay", "states500", "checklist", "hosts", "extra"]


def _runs_cell(runs, run_tiers):
    if not runs:
        return "--", "--"
    tiers = sorted({t for r in runs for t in run_tiers.get(r, [])}, key=TIER_ORDER.index)
    by_dir = {}
    for r in runs:
        d, n = r.rsplit("/", 1)
        by_dir.setdefault(d, []).append(n[:-5])
    if len(runs) <= 4:
        cell = "; ".join(f"`{d}/` " + ", ".join(ns) for d, ns in by_dir.items())
    else:
        cell = f"{len(runs)} runs in " + ", ".join(f"`{d}/`" for d in by_dir)
    return cell, ", ".join(tiers) or "--"


def write_markdown(S, path, runs_file):
    deps = json.loads(Path(runs_file).read_text())
    rt = deps["run_tiers"]
    lines = [
        "# Every number in the paper, and how to regenerate it",
        "",
        "Generated by `python3 probes/paper_numbers.py --markdown repro/PAPER_NUMBERS.md` from the number",
        "map in `probes/paper_numbers.py`, the snapshot `results/budget_shift/summary.json` and the run",
        "dependencies in `repro/paper_number_runs.json` (`probes/paper_number_runs.py`). Do not edit by hand.",
        "",
        "Every number in the paper's text and captions is listed below with the snapshot entry it is read",
        "from; the tables and figures are generated whole from the same snapshot. `python3 probes/paper_numbers.py`",
        "checks all of them against the snapshot on a CPU, and `bash repro/regenerate_paper.sh TIER` re-collects",
        "the runs behind them on a GPU host and checks them again (see the README).",
        "",
        "Kinds: **outcome** numbers (success, discordant pairs, statistics, the gate's replay error) come from",
        "deterministic rollouts and must reproduce on an RTX A6000; across the shipped machines 7 of 11,720",
        "paired outcomes changed, so a regenerated rate can move by a state in rare cases. **latency** numbers",
        "are re-measured by the same runs but belong to the machine. **hosts** numbers count the shipped",
        "machines and runs. **formula**, **protocol**, **cited** and **definition** numbers are arithmetic,",
        "design constants, citations or notation.",
        "",
        "## Regeneration tiers",
        "",
        "| tier | results directories | GPU-hours (one A6000) |",
        "|---|---|---|",
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_regeneration as CR
    plan = CR.read_plan()
    for t in TIER_ORDER:
        rows = [r for r in plan if r["tier"] == t]
        dirs = ", ".join(dict.fromkeys(f"`{r['dir']}/`" for r in rows))
        lines.append(f"| {t} | {dirs} | {sum(r['hours'] for r in rows):.1f} |")
    lines += ["", "`paper` runs every tier except `hosts` and `extra`: those regenerate runs whose only role is",
              "the cross-machine tally, or that no printed number uses.", "",
              "## Tables and figures", "",
              "| item | generated by | runs | tiers |", "|---|---|---|---|"]
    for name, rel in GENERATED:
        if rel.endswith(".tex"):
            runs = deps["tables"].get(Path(rel).name, [])
            gen = f"`probes/make_tables.py` -> `{rel}`"
        else:
            runs = sorted({r for v in deps["figures"].values() for r in v})
            gen = "`probes/make_figure.py` -> `figure_data.json`"
        cell, tiers = _runs_cell(runs, rt)
        lines.append(f"| {name} | {gen} | {cell} | {tiers} |")
    lines += ["", "## Numbers in the text", ""]
    where = None
    for i, e in enumerate(ENTRIES):
        if e.where != where:
            where = e.where
            lines += ["", f"### {where}", "", "| phrase | number | kind | from | runs | tier |", "|---|---|---|---|---|---|"]
        first = True
        for j, n in enumerate(e.nums):
            src = f"`{n.how}`" if isinstance(n.get, str) else n.how
            dep = deps["numbers"].get(f"{i}:{j}")
            cell, tiers = _runs_cell(dep["runs"], rt) if dep else ("--", "--")
            phrase = e.text.replace("|", "/") if first else ""
            lines.append(f"| {phrase} | **{n.printed}** | {n.kind} | {src.replace('|', '/')} | {cell} | {tiers} |")
            first = False
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}: {sum(len(e.nums) for e in ENTRIES)} numbers in {len(ENTRIES)} phrases")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--markdown", type=Path, help="write the reviewer-facing number map here")
    ap.add_argument("--runs", type=Path, default=ROOT / "repro" / "paper_number_runs.json",
                    help="run dependencies (probes/paper_number_runs.py)")
    ap.add_argument("--compare", type=Path,
                    help="probes/compare_regen.py --json output for the runs behind --summary")
    ap.add_argument("--results", type=Path,
                    help="the results tree behind --summary, when it holds only part of the collection")
    args = ap.parse_args(argv)
    S = json.loads(args.summary.read_text())
    if args.markdown:
        write_markdown(S, args.markdown, args.runs)
        return 0
    compare = json.loads(args.compare.read_text()) if args.compare and args.compare.is_file() else None
    runs = json.loads(args.runs.read_text()) if args.runs.is_file() else None
    present = None
    if args.results:
        present = {str(q.relative_to(args.results)) for q in args.results.rglob("*.json")}
    return 0 if check(S, str(args.summary), compare, runs, present) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())

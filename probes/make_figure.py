"""Render the paper's two figures from the analysis snapshot.

    python make_figure.py [--summary PATH] [--outdir DIR]

Reads results/budget_shift/summary.json (written by
probes/report_budget_shift.py in the reproduction repository) and writes
figures/frontier.pdf (Fig. 1) and figures/shift.pdf (Fig. 2), with PNG previews.
Every plotted value is copied into figure_data.json together with the
snapshot's SHA-256, so the figures can be checked against the numbers without
re-running anything.

Palette: categorical slots validated for colour-vision deficiency with the
dataviz validator (blue/orange for uncached/cached in Fig. 1; teal/violet for the
two checkpoints in Fig. 2; light surface). Identity never rests on colour alone:
marker shape, fill and direct labels carry it too.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE.parent / "results/budget_shift/summary.json"
if not DEFAULT_SUMMARY.is_file():
    for cand in [
        HERE / "results/budget_shift/summary.json",
        HERE.parents[1] / "release/cache-or-denoise-less/results/budget_shift/summary.json",
        HERE.parents[1] / "release/residual-attenuation/results/budget_shift/summary.json",
    ]:
        if cand.is_file():
            DEFAULT_SUMMARY = cand
            break

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FS, FS_SMALL = 6.5, 5.8

UNCACHED = ["uncached 1 step", "uncached 2 steps", "uncached 4 steps", "uncached 10 steps"]
CACHED = {  # label -> (marker, direct label)
    "10 steps, cache tau=2": ("s", r"$\tau$=2"),
    "10 steps, cache tau=4": ("s", r"$\tau$=4"),
    "10 steps, cache tau=8": ("s", r"$\tau$=8"),
    "10 steps, cache tau=0": ("s", "never"),
    "10 steps, attenuated cache tau=8": ("D", r"att. $\tau$=8"),
    "4 steps, cache tau=4": ("^", r"4 steps, $\tau$=4"),
}
YLIM_FRONTIER = (55, 101)
YLIM_SHARE = (20, 102)

# Direct labels, placed by hand in data coordinates (latency ms, success %) on
# the final data; a thin leader joins a label to its point when they are apart.
# The plotted values do not depend on these positions.
LABELS = {
    "idm": {"uncached 1 step": (83.5, 83), "uncached 2 steps": (84, 99.5), "uncached 4 steps": (113, 99.5),
            "uncached 10 steps": (188, 99.5), "4 steps, cache tau=4": (97, 86.5),
            "10 steps, cache tau=2": (150, 99.5), "10 steps, cache tau=4": (136, 88),
            "10 steps, cache tau=8": (121, 62), "10 steps, attenuated cache tau=8": (122, 80)},
    "native": {"uncached 1 step": (83.5, 87), "uncached 2 steps": (90, 100), "uncached 4 steps": (113, 99.5),
               "uncached 10 steps": (188, 99.5), "4 steps, cache tau=4": (84, 79),
               "10 steps, cache tau=0": (106, 86.5), "10 steps, cache tau=8": (124, 84),
               "10 steps, cache tau=4": (134, 79), "10 steps, cache tau=2": (152, 86)},
}


def label(ax, text, xy, xytext, color):
    """Direct label at data position xytext, with a thin leader when it sits apart from its point."""
    ha = "left"
    if xytext is not None and len(xytext) == 3:
        x, y, ha = xytext
        xytext = (x, y)
    elif xytext is None:
        xytext = (xy[0] + 3, xy[1] - 3)
    far = abs(xytext[0] - xy[0]) > 6 or abs(xytext[1] - xy[1]) > 3.5
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", fontsize=FS_SMALL, color=INK,
                ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.45, shrinkA=1.5, shrinkB=2.5) if far else None)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=FS, width=0.6, length=2.5)


def frontier_panel(ax, block, title, offsets, plotted):
    arms = block["arms"]
    xs, ys = [], []
    for lab in UNCACHED:
        if lab not in arms:
            continue
        a = arms[lab]
        x, y = a["latency"]["ep_mean"], a["success"]["pct"]
        lo, hi = a["wilson95"]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=4.2, color=BLUE, mec="white",
                    mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
        xs.append(x); ys.append(y)
        n = lab.split()[1]
        label(ax, f"{n} step" + ("" if n == "1" else "s"), (x, y), offsets.get(lab), BLUE)
        plotted.append({"figure": 1, "panel": title, "arm": lab, "latency_ms": x, "success_pct": y,
                        "wilson95": [lo, hi]})
    ax.plot(xs, ys, color=BLUE, linewidth=1.0, zorder=2)
    for lab, (marker, short) in CACHED.items():
        if lab not in arms:
            continue
        a = arms[lab]
        x, y = a["latency"]["ep_mean"], a["success"]["pct"]
        lo, hi = a["wilson95"]
        plotted.append({"figure": 1, "panel": title, "arm": lab, "latency_ms": x, "success_pct": y,
                        "wilson95": [lo, hi]})
        if y < YLIM_FRONTIER[0]:
            # below the shared axis: mark it at the floor with downward indicator and clear label
            ax.plot([x], [YLIM_FRONTIER[0]], marker="v", ms=4.0, color=ORANGE, mec="white", mew=0.8, zorder=4)
            ax.annotate(f"{short}: {y:.0f}%\n(off scale)", xy=(x, YLIM_FRONTIER[0]), xytext=(x - 30, YLIM_FRONTIER[0] + 5),
                        fontsize=FS_SMALL, color=INK, ha="left",
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, mutation_scale=6))
            continue
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, ms=4.2, color=ORANGE, mec="white",
                    mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
        label(ax, short, (x, y), offsets.get(lab), ORANGE)
    ax.set_title(title, fontsize=FS + 0.5, color=INK, loc="left", pad=3)
    ax.set_xlabel("mean latency per replanning call (ms)", fontsize=FS, color=INK, labelpad=1.5)
    ax.set_xlim(70, 215)
    style(ax)


def wilson95(k, n, z=1.959964):
    """95% Wilson score interval for k successes in n, in percent."""
    p_, d_ = k / n, 1 + z * z / n
    c_ = (p_ + z * z / (2 * n)) / d_
    h_ = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d_
    return 100 * (c_ - h_), 100 * (c_ + h_)


def share_panel(ax, S, plotted):
    from matplotlib.lines import Line2D
    # Bars are 95% Wilson intervals. Series that share an x position (every
    # uncached control sits at 0%) are drawn slightly apart so their bars stay
    # readable; the figure data records the true shares.
    dodge = {("idm", 1.0): -1.2, ("idm", 3.0): -0.4, ("idm", 5.0): 0.4, ("native", 1.0): 1.2, ("native", 5.0): 2.0}
    # (checkpoint, shift) -> colour, marker, filled, line style. Colour is the
    # checkpoint; the shift is carried by fill and line style, so the s=3 series
    # of the follow-up collection sits between s=1 (solid) and s=5 (dashed).
    series = {("idm", 1.0): (AQUA, "o", True, "-"), ("idm", 3.0): (AQUA, "^", True, "-."),
              ("idm", 5.0): (AQUA, "o", False, "--"),
              ("native", 1.0): (VIOLET, "s", True, "-"), ("native", 5.0): (VIOLET, "s", False, "--")}
    names = {"idm": "Optional-IDM", "native": "original"}
    handles = []
    for (ck, s), (col, mk, filled, ls) in series.items():
        pts = sorted((r["fraction"] * 100, r["success"]["pct"], r["success"]["k"], r["success"]["n"])
                     for r in S["shift_sweep"] if r["ckpt"] == ck and r["shift"] == s)
        if not pts:
            continue
        x = [p_[0] + dodge[(ck, s)] for p_ in pts]
        y = [p_[1] for p_ in pts]
        ci = [wilson95(p_[2], p_[3]) for p_ in pts]
        ax.errorbar(x, y, yerr=[[yi - lo for yi, (lo, _) in zip(y, ci)], [hi - yi for yi, (_, hi) in zip(y, ci)]],
                    fmt="none", ecolor=col, elinewidth=0.6, capsize=0, alpha=0.55, zorder=1)
        ax.plot(x, y, ls, color=col, linewidth=1.0, zorder=2)
        ax.plot(x, y, mk, ms=4.2, color=col, mfc=col if filled else "white", mec=col, mew=1.0, zorder=3)
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.0, marker=mk, ms=3.8,
                              mfc=col if filled else "white", mec=col, label=f"{names[ck]}, s={s:.0f}"))
        for (xi, yi, _, _), (lo, hi) in zip(pts, ci):
            plotted.append({"figure": 2, "series": f"{ck} s={s:g} fixed noise", "cached_share_pct": xi,
                            "success_pct": yi, "wilson95": [round(lo, 1), round(hi, 1)]})
    # fresh noise on every call: uncached ten steps at 0% and every cache arm
    fresh = {r["arm"]: r["fresh"] for r in S["noise"]["fixed_vs_fresh"]}
    sweep = S["noise"].get("fresh_noise_share_sweep", [])
    for ck, s, col, base in (("idm", 1.0, AQUA, "idm_s1_stock10"), ("idm", 3.0, AQUA, None),
                             ("idm", 5.0, AQUA, None), ("native", 5.0, VIOLET, "uncond_s5_stock10")):
        pts = [(0.0, fresh[base]["pct"], fresh[base]["k"], fresh[base]["n"])] if base in fresh else []
        pts += [(r["fraction"] * 100, r["success"]["pct"], r["success"]["k"], r["success"]["n"])
                for r in sweep if r["ckpt"] == ck and r["shift"] == s]
        pts.sort()
        if not pts:
            continue
        x = [p_[0] + dodge[(ck, s)] + 0.8 for p_ in pts]
        y = [p_[1] for p_ in pts]
        ci = [wilson95(p_[2], p_[3]) for p_ in pts]
        ax.errorbar(x, y, yerr=[[yi - lo for yi, (lo, _) in zip(y, ci)], [hi - yi for yi, (_, hi) in zip(y, ci)]],
                    fmt="none", ecolor=col, elinewidth=0.6, capsize=0, alpha=0.4, zorder=1)
        if len(pts) > 1:
            ax.plot(x, y, ":", color=col, linewidth=1.0, zorder=2)
        ax.plot(x, y, "x", ms=4.6, color=col, mew=1.2, zorder=4)
        for (xi, yi, _, _), (lo, hi) in zip(pts, ci):
            plotted.append({"figure": 2, "series": f"{ck} s={s:g} fresh noise", "cached_share_pct": xi,
                            "success_pct": yi, "wilson95": [round(lo, 1), round(hi, 1)]})
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.0, marker="x", ms=4.2, mew=1.1,
                          label="fresh noise per call"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, frameon=False,
              handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)
    ax.set_xlabel("share of the flow path cached (%)", fontsize=7.2, color=INK, labelpad=1.5)
    ax.set_ylabel("task success (%)", fontsize=7.2, color=INK, labelpad=1.5)
    ax.set_xlim(-3, 85)
    ax.set_ylim(*YLIM_SHARE)
    style(ax)
    ax.tick_params(colors=MUTED, labelsize=6.8, width=0.6, length=2.5)


def screen_panel(ax, S, plotted):
    """Closed-loop success against the screen's deviation D (Sec. 4, repro/experiments/screen.sh).

    Same encoding as share_panel: colour is the checkpoint, filled circles s=1,
    triangles s=3, open circles s=5, diamonds s=2, crosses fresh noise. The grey
    curve is the logistic regression on log D that made the held-out
    predictions; stars are the held-out configurations' observed success on the
    other 50 states.
    """
    from matplotlib.lines import Line2D
    sc = S["screen"]
    col = {"idm": AQUA, "uncond": VIOLET}
    mk = {1.0: ("o", True), 2.0: ("D", True), 3.0: ("^", True), 5.0: ("o", False)}
    for r in sc["primary"]:
        c = col[r["ckpt"]]; m_, filled = mk[r["shift"]]
        x, y = r["D"], r["success"]["pct"]
        lo, hi = wilson95(r["success"]["k"], r["success"]["n"])
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=0.6,
                    capsize=0, alpha=0.5, zorder=1)
        if r["fresh_noise"]:
            ax.plot([x], [y], "x", ms=4.6, color=c, mew=1.2, zorder=3)
        else:
            ax.plot([x], [y], m_, ms=3.6 if m_ == "D" else 4.2, color=c, mfc=c if filled else "white",
                    mec=c, mew=1.0, zorder=3)
        plotted.append({"figure": "A1", "series": f"{r['ckpt']} s={r['shift']:g} "
                        f"{'fresh' if r['fresh_noise'] else 'fixed'} noise", "arm": f"{r['cell']} p{r['m']}_tau{r['tau']}",
                        "cached_steps": r["m"],
                        "D": r["D"], "success_pct": y, "wilson95": [round(lo, 1), round(hi, 1)]})
    xs = [r["D"] for r in sc["primary"]] + [v["D"] for v in sc.get("validation", [])]
    # the fit exists only when every configuration of the prediction is present
    # (a snapshot regenerated from part of the runs may lack it)
    if "logistic" in sc:
        w = sc["logistic"]["D"]["w"]
        grid = np.geomspace(min(xs) / 1.3, max(xs) * 1.3, 200)
        ax.plot(grid, 100.0 / (1.0 + np.exp(-(w[0] + w[1] * np.log(grid)))), "-", color=MUTED, lw=0.8,
                alpha=0.8, zorder=0)
    for v in sc.get("validation", []):
        if "observed" not in v:
            continue
        lo, hi = wilson95(v["observed"]["k"], v["observed"]["n"])
        y = v["observed"]["pct"]
        ax.errorbar([v["D"]], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, elinewidth=0.6,
                    capsize=0, alpha=0.6, zorder=2)
        ax.plot([v["D"]], [y], "*", ms=7.0, color=col[v["cell"].split("_")[1]], mec=INK, mew=0.5, zorder=4)
        plotted.append({"figure": "A1", "series": "held-out", "arm": f"{v['cell']} p{v['m']}_tau{v['tau']}",
                        "cell": v["cell"], "cached_steps": v["m"],
                        "D": v["D"], "success_pct": y, "predicted_pct": v["pred_D"],
                        "wilson95": [round(lo, 1), round(hi, 1)]})
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator
    ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if min(xs) / 1.3 <= t <= max(xs) * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("screened deviation $D$ (log scale)", fontsize=7.2, color=INK, labelpad=1.5)
    ax.set_ylabel("task success (%)", fontsize=7.2, color=INK, labelpad=1.5)
    ax.set_ylim(0, 102)
    style(ax)
    ax.tick_params(colors=MUTED, labelsize=6.8, width=0.6, length=2.5)
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=0.8, label="logistic fit on log D"),
                       Line2D([], [], ls="none", marker="*", ms=6.5, color=MUTED, mec=INK, mew=0.5,
                              label="held-out, other 50 states")],
              loc="lower left", fontsize=6.8, frameon=False, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=HERE,
                    help="where figures/ and figure_data.json go (default: next to this script)")
    args = ap.parse_args()
    raw = args.summary.read_bytes()
    S = json.loads(raw)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    out = args.outdir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plotted = []
    meta = {"Creator": None, "Producer": None, "Author": None, "CreationDate": None}

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.28), sharey=True)
    # Fig. 1 plots the frontier host only. The composed two-step cache ran on
    # other machines, and latency does not transfer between machines, so it is
    # reported in the text against its own host's baselines instead.
    frontier_panel(axes[0], S["frontier"]["idm"], "(a) Optional-IDM checkpoint, s = 1", LABELS["idm"], plotted)
    frontier_panel(axes[1], S["frontier"]["native"], "(b) original checkpoint, s = 5", LABELS["native"], plotted)
    axes[0].set_ylabel("task success (%)", fontsize=FS, color=INK, labelpad=1.5)
    axes[0].set_ylim(*YLIM_FRONTIER)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(out / "frontier.pdf", metadata=meta)
    fig.savefig(out / "frontier.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.7, 1.7))
    share_panel(ax, S, plotted)
    fig.tight_layout(pad=0.25)
    fig.savefig(out / "shift.pdf", metadata=meta)
    fig.savefig(out / "shift.png", dpi=220)
    plt.close(fig)

    if S.get("screen", {}).get("primary"):
        fig, ax = plt.subplots(figsize=(2.7, 1.7))
        screen_panel(ax, S, plotted)
        fig.tight_layout(pad=0.25)
        fig.savefig(out / "screen.pdf", metadata=meta)
        fig.savefig(out / "screen.png", dpi=220)
        plt.close(fig)

    (args.outdir / "figure_data.json").write_text(json.dumps(
        {"summary": "results/budget_shift/summary.json", "summary_sha256": hashlib.sha256(raw).hexdigest(),
         "points": plotted}, indent=1) + "\n")
    extra = ", figures/screen.pdf" if S.get("screen", {}).get("primary") else ""
    print(f"wrote figures/frontier.pdf, figures/shift.pdf{extra} ({len(plotted)} points)")


if __name__ == "__main__":
    main()

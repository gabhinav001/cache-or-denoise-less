"""Write the paper's tables from the analysis snapshot.

    python make_tables.py [--summary PATH] [--outdir DIR]

Reads results/budget_shift/summary.json (probes/report_budget_shift.py in the
reproduction repository) and writes tables/frontier.tex, so every number in
Table 1 is copied from the snapshot rather than typed. All rows come from one
host; the script stops if the snapshot's frontier arms span several.
"""
from pathlib import Path
import argparse
import json

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

ROWS = [  # (label in summary, printed label, K, number of cached leading steps)
    ("uncached 1 step", "Uncached", 1, 0),
    ("uncached 2 steps", "Uncached", 2, 0),
    ("uncached 4 steps", "Uncached", 4, 0),
    ("uncached 10 steps", "Uncached", 10, 0),
    ("4 steps, cache tau=4", r"Cache, $\tau{=}4$", 4, 2),
    ("10 steps, cache tau=2", r"Cache, $\tau{=}2$", 10, 8),
    ("10 steps, cache tau=4", r"Cache, $\tau{=}4$", 10, 8),
    ("10 steps, cache tau=8", r"Cache, $\tau{=}8$", 10, 8),
    ("10 steps, cache tau=0", r"Cache, never refreshed", 10, 8),
    ("10 steps, attenuated cache tau=8", r"Attenuated cache, $\tau{=}8$", 10, 8),
]
# Every row comes from the one frontier host. Runs from other machines (the
# composed two-step cache, for instance) are reported in the text against their
# own host's baselines, never in this table: latency does not transfer between
# machines.


def share(K, m, s):
    """Percent of the flow path (sigma from 1 to 0) integrated by the first m of K steps."""
    if m == 0:
        return None
    u = 1 - m / K
    return 100 * (1 - s * u / (1 + (s - 1) * u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=HERE,
                    help="tables/ is written here (default: next to this script)")
    args = ap.parse_args()
    S = json.loads(args.summary.read_text())
    F = S["frontier"]
    hosts = {a["host"]["host_tag"] for block in F.values() for a in block["arms"].values()}
    if len(hosts) != 1:
        raise SystemExit(f"Table 1 must come from one host, found {sorted(hosts)}")
    num = lambda v: "--" if v is None else "%.0f" % v
    succ = lambda a: "--" if a is None else "%.0f" % a["success"]["pct"]
    mean = lambda a: "--" if a is None else "%.0f" % a["latency"]["ep_mean"]
    p95 = lambda a: "--" if a is None else "%.0f" % a["latency"]["cyc_p95"]
    lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{path cached (\%)} & \multicolumn{2}{c}{success (\%)}"
             r" & \multicolumn{2}{c}{mean latency (ms)} & \multicolumn{2}{c}{p95 latency (ms)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
             r"Configuration & $K$ & IDM & orig. & IDM & orig. & IDM & orig. & IDM & orig. \\",
             r"\midrule"]
    for key, label, K, m in ROWS:
        idm, nat = F["idm"]["arms"].get(key), F["native"]["arms"].get(key)
        # Optional-IDM runs at s=1 and the original checkpoint at s=5, so the same
        # cached steps cover different shares of the flow path.
        lines.append(f"{label} & {K} & {num(share(K, m, 1.0))} & {num(share(K, m, 5.0))} & "
                     f"{succ(idm)} & {succ(nat)} & {mean(idm)} & {mean(nat)} & {p95(idm)} & {p95(nat)} \\\\")
        if key == "uncached 10 steps":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = args.outdir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frontier.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/frontier.tex")
    write_appendix(S, out)


# ------------------------------------------------------------------ appendix
# The appendix tables are generated the same way as Table 1: every number is
# copied from the snapshot, never typed.

def _pct(v):
    return "--" if v is None else "%.0f" % v


def _p(v):
    if v is None:
        return "--"
    if v < 1e-4:
        return "$<10^{-4}$"
    if v < 0.01:
        return "%.4f" % v
    return "%.3f" % v if v < 0.1 else "%.2f" % v


def _iv(iv):
    return "--" if not iv else "$[%.1f, %.1f]$" % (iv[0], iv[1])


def _tab(spec, head, rows, mid=()):
    body = [r"\begin{tabular}{@{}%s@{}}" % spec, r"\toprule", head, r"\midrule"]
    for i, r in enumerate(rows):
        if i in mid:
            body.append(r"\midrule")
        body.append(r)
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


CONTRAST_NAMES = {
    "idm: uncached 2 steps vs uncached 10 steps": ("IDM", "2 vs.\\ 10 steps"),
    "idm: uncached 2 steps vs 10 steps, cache tau=8": ("IDM", "2 steps vs.\\ cache $\\tau{=}8$"),
    "idm: uncached 2 steps vs 10 steps, attenuated cache tau=8": ("IDM", "2 steps vs.\\ attenuated $\\tau{=}8$"),
    "idm: uncached 2 steps vs 10 steps, cache tau=2": ("IDM", "2 steps vs.\\ cache $\\tau{=}2$"),
    "idm: uncached 2 steps vs 4 steps, cache tau=4": ("IDM", "2 steps vs.\\ 4-step cache $\\tau{=}4$"),
    "idm: 10 steps, attenuated cache tau=8 vs 10 steps, cache tau=8": ("IDM", "attenuated vs.\\ plain $\\tau{=}8$"),
    "native: uncached 2 steps vs uncached 10 steps": ("orig.", "2 vs.\\ 10 steps"),
    "native: uncached 2 steps vs 10 steps, cache tau=8": ("orig.", "2 steps vs.\\ cache $\\tau{=}8$"),
    "native: uncached 2 steps vs 10 steps, cache tau=0": ("orig.", "2 steps vs.\\ never refreshed"),
    "native: uncached 2 steps vs 4 steps, cache tau=4": ("orig.", "2 steps vs.\\ 4-step cache $\\tau{=}4$"),
    "native: uncached 1 step vs uncached 10 steps": ("orig.", "1 vs.\\ 10 steps"),
}


def app_contrasts(S, out):
    # A1: the Holm family of Sec. 3
    rows = []
    for c in S["primary_contrasts"]:
        ck, name = CONTRAST_NAMES[c["label"]]
        rows.append(f"{ck} & {name} & {_pct(c['a_pct'])} & {_pct(c['b_pct'])} & {('%+.0f' % c['diff']) if c['diff'] else '0'} & "
                    f"{_iv(c['tango90'])} & {_p(c['p'])} & {_p(c['p_holm'])} \\\\")
    (out / "app_contrasts.tex").write_text(_tab(
        "llrrrrrr", r"Ckpt. & Contrast (A vs.\ B) & A (\%) & B (\%) & A$-$B & Tango 90\% & $p$ & Holm $p$ \\",
        rows, mid=(6,)))


def app_share(S, out):
    # A2: tau=8 cached-share sweep, fixed and fresh noise (50 states)
    fresh = {r["arm"]: r for r in S["noise"]["fixed_vs_fresh"]}
    rows = []
    names = {"idm": "IDM", "native": "orig."}
    for r in sorted(S["shift_sweep"], key=lambda r: (r["ckpt"] != "idm", r["shift"], r["prefix"])):
        stem = Path(r["source"]).stem
        f = fresh.get(stem)
        share = 100 * r["fraction"]
        sig = "--" if r["prefix"] == 0 else "%.2f" % (1 - r["fraction"])
        fr = "--" if f is None else _pct(f["fresh"]["pct"])
        pf = "--" if f is None else _p(f["fresh_minus_fixed"]["p"])
        rows.append(f"{names[r['ckpt']]} & {r['shift']:.0f} & {r['prefix']} & {share:.0f} & {sig} & "
                    f"{_pct(r['success']['pct'])} & {fr} & {pf} \\\\")
    (out / "app_share.tex").write_text(_tab(
        "lrrrrrrr", r"Ckpt. & $s$ & cached & share (\%) & $\sigma$ fresh & fixed (\%) & fresh (\%) & $p$ \\", rows))


def app_regimes(S, out):
    # A3: other regimes, two against ten steps
    cam = S["followup"]["camera"]["cam0015"]
    cr = cam["rates"]; cc = {c["label"].split(": ")[1]: c for c in cam["contrasts"]}
    nz = {r["arm"]: r for r in S["noise"]["fixed_vs_fresh"]}
    nc = {c["label"]: c for c in S["noise"]["contrasts_under_fresh_noise"]}
    lg = S["libero_long"]
    ss = S["earlier"]["statescale"]["full_vs_2"]
    rows = [
        f"LIBERO-Goal, all states & IDM & 500 & {ss['b_pct']:.1f} & {ss['a_pct']:.1f} & {_iv(ss['tango90'])} & -- & -- \\\\",
        f"Camera jitter 1.5\\,cm & IDM & 50 & {_pct(cr['idm_stock10']['pct'])} & {_pct(cr['idm_stock2']['pct'])} & "
        f"{_iv(cc['idm_stock2 vs idm_stock10']['tango90'])} & {_pct(cr['idm_stock1']['pct'])} & {_pct(cr['idm_tau8']['pct'])} \\\\",
        f"Camera jitter 1.5\\,cm & orig. & 50 & {_pct(cr['uncond_stock10']['pct'])} & {_pct(cr['uncond_stock2']['pct'])} & "
        f"{_iv(cc['uncond_stock2 vs uncond_stock10']['tango90'])} & -- & -- \\\\",
        f"Fresh noise per call & IDM & 50 & {_pct(nz['idm_s1_stock10']['fresh']['pct'])} & {_pct(nz['idm_s1_stock2']['fresh']['pct'])} & "
        f"{_iv(nc['idm_s1_stock2_rn vs idm_s1_stock10_rn']['tango90'])} & -- & {_pct(nz['idm_s1_whole_tau8']['fresh']['pct'])} \\\\",
        f"Fresh noise per call & orig. & 50 & {_pct(nz['uncond_s5_stock10']['fresh']['pct'])} & {_pct(nz['uncond_s5_stock2']['fresh']['pct'])} & "
        f"{_iv(nc['uncond_s5_stock2_rn vs uncond_s5_stock10_rn']['tango90'])} & -- & {_pct(nz['uncond_s5_whole_tau8']['fresh']['pct'])} \\\\",
        f"LIBERO-Long & IDM & 50 & {_pct(lg['idm']['stock10']['pct'])} & {_pct(lg['idm']['stock2']['pct'])} & "
        f"{_iv(lg['idm']['contrasts'][0]['tango90'])} & -- & {_pct(lg['idm']['whole_tau8']['pct'])} \\\\",
        f"LIBERO-Long & orig. & 50 & {_pct(lg['uncond']['stock10']['pct'])} & {_pct(lg['uncond']['stock2']['pct'])} & "
        f"{_iv(lg['uncond']['contrasts'][0]['tango90'])} & -- & {_pct(lg['uncond']['whole_tau8']['pct'])} \\\\",
    ]
    ji = ((S.get("idle_host") or {}).get("jitter") or {}).get("idm")
    if ji:   # the planned 250-state extension (idle_host.sh), after the 50-state row it contains
        rows.insert(2, f"Camera jitter 1.5\\,cm & IDM & {ji['contrast']['n']} & {ji['stock10']['pct']:.1f} & "
                       f"{ji['stock2']['pct']:.1f} & {_iv(ji['contrast']['tango90'])} & -- & -- \\\\")
    (out / "app_regimes.tex").write_text(_tab(
        "llrrrrrr", r"Condition & Ckpt. & states & 10 steps & 2 steps & 2$-$10, Tango 90\% & 1 step & cache $\tau{=}8$ \\", rows))


def app_latency(S, out):
    # A4: latency per call on each host
    F = S["frontier"]; lm = S["latency_model"]; fu = S["followup"]
    def ep(block, lab, key="latency"):
        a = F[block]["arms"].get(lab)
        return None if not a or not a.get(key) else a[key]["ep_mean"]
    r1 = fu["timing"]["rounds"]["1"]
    mc = S["matched_composed"]["arms"]
    rows = [
        f"Table~1 host, IDM & {_pct(ep('idm','uncached 1 step'))} & {_pct(ep('idm','uncached 2 steps'))} & {_pct(ep('idm','uncached 4 steps'))} & "
        f"{_pct(ep('idm','uncached 10 steps'))} & {_pct(ep('idm','10 steps, cache tau=8'))} & {lm['idm']['fixed_in_step_costs']:.1f} \\\\",
        f"Table~1 host, orig. & {_pct(ep('native','uncached 1 step'))} & {_pct(ep('native','uncached 2 steps'))} & {_pct(ep('native','uncached 4 steps'))} & "
        f"{_pct(ep('native','uncached 10 steps'))} & {_pct(ep('native','10 steps, cache tau=8'))} & {lm['native']['fixed_in_step_costs']:.1f} \\\\",
        f"Table~1 host, orig., CUDA graphs & {_pct(ep('native','uncached 1 step','latency_cudagraphs'))} & {_pct(ep('native','uncached 2 steps','latency_cudagraphs'))} & "
        f"{_pct(ep('native','uncached 4 steps','latency_cudagraphs'))} & {_pct(ep('native','uncached 10 steps','latency_cudagraphs'))} & -- & "
        f"{lm['native_cudagraphs']['fixed_in_step_costs']:.1f} \\\\",
        f"Follow-up host, IDM (round 1) & {_pct(r1['stock1']['ep_mean'])} & {_pct(r1['stock2']['ep_mean'])} & -- & {_pct(r1['stock10']['ep_mean'])} & "
        f"{_pct(r1['tau8']['ep_mean'])} & {fu['cost_two_point']['fixed_in_step_costs']:.1f} \\\\",
        f"Shared host, IDM & {_pct(mc['idm_stock_1step']['latency']['ep_mean'])} & {_pct(mc['idm_stock_2step']['latency']['ep_mean'])} & -- & -- & -- & -- \\\\",
        f"Shared host, orig. & {_pct(mc['uncond_stock_1step']['latency']['ep_mean'])} & {_pct(mc['uncond_stock_2step']['latency']['ep_mean'])} & -- & -- & -- & -- \\\\",
    ]
    idle = (S.get("idle_host") or {}).get("timing")
    if idle:   # the machine rented whole (idle_host.sh); mean over its interleaved rounds
        rows.append(f"Idle host, IDM ({idle['n_rounds']} rounds) & -- & {_pct(idle['stock2_ms_mean'])} & -- & -- & -- & -- \\\\")
    (out / "app_latency.tex").write_text(_tab(
        "lrrrrrr", r"Host and checkpoint & 1 step & 2 steps & 4 steps & 10 steps & cache $\tau{=}8$ & $F/a$ \\", rows))


def app_composed(S, out):
    # A5: the composed two-step cache
    fu = S["followup"]
    rr = fu["composed_reuse_over_refresh"]
    per_round = fu["timing"]["ratios"]["comp4/stock2"]["per_round"]
    mcc = S["matched_composed"]["contrasts"]
    vs2 = {"idm_whole_2step_p1_tau4": "%.2f" % (1 / mcc["idm_tau4_vs_2step"]["speedup"]),
           "uncond_whole_2step_p1_tau4": "%.2f" % (1 / mcc["native_tau4_vs_2step"]["speedup"])}
    # Success on the 100 states, from the run on each host; the follow-up rounds
    # are timing runs on three states per task.
    succ = {"quiet A6000": lambda a: S["composed2step"][a]["rate"]["pct"],
            "shared A6000": lambda a: S["matched_composed"]["arms"][a]["rate"]["pct"]}
    rows = []
    k = 0
    for x in rr:
        if x["host"] == "follow-up A6000":
            vs = "%.2f" % per_round[k]; k += 1
            name = f"follow-up A6000, IDM, round {k}"
            sc = "--"
        else:
            ck = "IDM" if x["arm"].startswith("idm") else "orig."
            name = f"{x['host']}, {ck}"
            vs = vs2.get(x["arm"], "--") if x["host"] == "shared A6000" else "--"
            sc = _pct(succ[x["host"]](x["arm"]))
        rows.append(f"{name} & {sc} & {x['reuse_ms']:.0f} & {x['refresh_ms']:.0f} & {x['reuse_over_refresh']:.3f} & "
                    f"{100 * x['tau4_mean_saving']:.0f} & {vs} \\\\")
    # The idle machine's rounds ran alone on a machine rented whole (idle_host.sh).
    for x in ((S.get("idle_host") or {}).get("timing") or {}).get("rounds", []):
        rows.append(f"idle A6000, IDM, round {x['round']} & -- & {x['reuse_ms']:.0f} & {x['refresh_ms']:.0f} & "
                    f"{x['reuse_over_refresh']:.3f} & {75 * (1 - x['reuse_over_refresh']):.0f} & {x['ratio']:.2f} \\\\")
    (out / "app_composed.tex").write_text(_tab(
        "lrrrrrr", r"Host & success (\%) & reuse (ms) & refresh (ms) & reuse/refresh & saving (\%) & vs.\ 2 steps \\", rows))


def app_screen(S, out):
    # A6: the cache screen (Sec. 4): every configuration with a closed-loop run on
    # the screened states, then the held-out configurations on the other 50 states
    sc = S["screen"]
    names = {"idm": "IDM", "uncond": "orig."}
    def row(r, succ, pred="--"):
        return (f"{names[r['ckpt']]} & {r['shift']:.0f} & {'fresh' if r['fresh_noise'] else 'fixed'} & "
                f"{r['m']} & {r['tau']} & {100 * r['share']:.0f} & {r['D']:.3f} & {succ} & {pred} \\\\")
    key = lambda r: (r["ckpt"] != "idm", r["fresh_noise"], r["shift"], r["m"], r["tau"])
    prim = [row(r, _pct(r["success"]["pct"])) for r in sorted(sc["primary"], key=key)]
    taus = [row(r, _pct(r["success"]["pct"])) for r in sorted(sc["tau_variants"], key=key)]
    held = []
    for v in sc["validation"]:
        cell = v["cell"].split("_")
        r = {"ckpt": cell[1], "shift": float(cell[2][1:]), "fresh_noise": v["cell"].endswith("_rn"),
             "m": v["m"], "tau": v["tau"], "share": v["share"], "D": v["D"]}
        held.append(row(r, _pct(v["observed"]["pct"]) if "observed" in v else "--", _pct(v["pred_D"])))
    rows = prim + taus + held
    (out / "app_screen.tex").write_text(_tab(
        "lrlrrrrrr", r"Ckpt. & $s$ & noise & cached & $\tau$ & share (\%) & $D$ & success (\%) & pred.\ (\%) \\",
        rows, mid=(len(prim), len(prim) + len(taus))))


def app_latent_delay(S, out):
    # A7: the episode-latent control and delayed execution (latent_delay.sh), as
    # paired contrasts on the same 50 states
    ld = S["latent_delay"]
    rows = []
    def row(a, b, c):
        rows.append(f"{a} & {b} & {_pct(c['a_pct'])} & {_pct(c['b_pct'])} & {('%+.0f' % c['diff']) if c['diff'] else '0'} & "
                    f"{_iv(c['tango90'])} & {_p(c['p'])} \\\\")
    for m in (6, 8):
        lt = ld["latent"][f"p{m}"]
        cache = f"IDM, {m} of 10 cached"
        row(f"{cache}, episode latent", "fresh noise", lt["episode_vs_fresh"])
        row(f"{cache}, episode latent", "fixed latent", lt["episode_vs_fixed"])
    dl = ld["delay"]
    row("IDM, 2 steps, delay 2", "10 steps, delay 5", dl["two_d2_vs_ten_d5"])
    row("IDM, 2 steps, delay 2", "2 steps, no delay", dl["two_d2_vs_two_d0"])
    row("IDM, 10 steps, delay 5", "10 steps, no delay", dl["ten_d5_vs_ten_d0"])
    row("IDM, 2 steps, delay 5", "2 steps, delay 2", dl["two_d5_vs_two_d2"])
    row("IDM, 10 steps, delay 5", "10 steps, delay 2", dl["ten_d5_vs_ten_d2"])
    row("IDM, 10 steps, delay 2", "2 steps, delay 2", dl["ten_d2_vs_two_d2"])
    cd = ld["cache_delay"]
    row(r"orig., cache $\tau{=}8$, delay 5/2", "2 steps, delay 2", cd["cache_d52_vs_two_d2"])
    row(r"orig., cache $\tau{=}8$, delay 5/2", "same, no delay", cd["cache_d52_vs_cache_sync"])
    row("orig., 2 steps, delay 2", "2 steps, no delay", cd["two_d2_vs_two_sync"])
    mid = [4, 10]
    gr = S.get("delay_graphs")
    if gr:   # delay_graphs.sh: the delays of CUDA-graph execution (two steps 2, ten steps 3)
        mid.append(len(rows))
        names = {"idm": "IDM", "orig": "orig."}
        for ck in ("idm", "orig"):
            if ck not in gr:
                continue
            row(f"{names[ck]}, 2 steps, delay 2", "10 steps, delay 3", gr[ck]["two_d2_vs_ten_d3"])
            row(f"{names[ck]}, 10 steps, delay 3", "10 steps, no delay", gr[ck]["ten_d3_vs_ten_d0"])
            if ck == "idm" and "ten_d4_vs_ten_d2" in gr[ck]:
                row("IDM, 10 steps, delay 4", "10 steps, delay 2", gr[ck]["ten_d4_vs_ten_d2"])
    (out / "app_latent_delay.tex").write_text(_tab(
        "llrrrrr", r"A & B & A (\%) & B (\%) & A$-$B & Tango 90\% & $p$ \\", rows, mid=tuple(mid)))


def app_per_task(S, out):
    # A8: the successes behind Table 1, task by task (descriptive)
    pt = S["per_task"]
    i, n = pt["idm"], pt["native"]
    labs = ("uncached 2 steps", "uncached 10 steps", "10 steps, cache tau=8")
    rows = []
    for j, task in enumerate(i["tasks"]):
        cells = [str(i["success"][lab][j]) for lab in labs] + [str(n["success"][lab][j]) for lab in labs]
        rows.append(f"{task} & {i['task_names'][j]} & " + " & ".join(cells) + " \\\\")
    tot = [str(sum(i["success"][lab])) for lab in labs] + [str(sum(n["success"][lab])) for lab in labs]
    rows.append(r"& total & " + " & ".join(tot) + " \\\\")
    head = (r"& & \multicolumn{3}{c}{IDM} & \multicolumn{3}{c}{orig.} \\ \cmidrule(lr){3-5} \cmidrule(l){6-8}" "\n"
            r"& Task & 2 steps & 10 steps & $\tau{=}8$ & 2 steps & 10 steps & $\tau{=}8$ \\")
    (out / "app_per_task.tex").write_text(_tab("rlrrrrrr", head, rows, mid=(len(rows) - 1,)))


def write_appendix(S, out):
    # A snapshot regenerated from part of the collection lacks some sections;
    # each table is written when everything it reads is present.
    done = []
    for fn in (app_contrasts, app_share, app_regimes, app_latency, app_composed, app_screen, app_latent_delay, app_per_task):
        try:
            fn(S, out)
            done.append(fn.__name__)
        except (KeyError, IndexError, TypeError) as e:
            print(f"skipped tables/{fn.__name__}.tex: this snapshot lacks {e}")
    print("wrote " + ", ".join(f"tables/{n}.tex" for n in done))

if __name__ == "__main__":
    main()

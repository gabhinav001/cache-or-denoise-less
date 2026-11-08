"""Does the write-up still agree with the artefacts it claims to report?

A paper and its data drift apart silently. A run gets re-collected, canonical
numbers move, and a sentence three sections away still quotes the old figure --
which is exactly how the superseded draft came to state, among other things, a
pi0.5 null that its own sweep contradicted.

This reads the .tex (or any text file) and checks every figure it is supposed to
contain against results/canonical_numbers.json and results/alpha_constraint.json.
It is deliberately literal: each entry names the value and a regex that must
match somewhere in the file. A number that moves in the data and not in the prose
fails here rather than in review.

usage: check_paper_numbers.py [path/to/main.tex]
"""
from __future__ import annotations

import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "results")
DEFAULT_TEX = os.path.join(HERE, "..", "..", "..", "paper", "main.tex")


_WORDS = {13: "Thirteen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty",
          21: "Twenty-one", 22: "Twenty-two", 23: "Twenty-three", 24: "Twenty-four"}


def _word(n):
    """Regex matching the spelled-out count, so prose cannot drift from the map."""
    w = _WORDS.get(n)
    return re.escape(w) if w else rf"{n}"


# How many experiments actually ship, counted from repro/experiments/.
_EXPDIR = os.path.join(HERE, "..", "repro", "experiments")
_EXPS = [f for f in os.listdir(_EXPDIR) if f.endswith(".sh")]
# An experiment marked PENDING is written and validated but not yet collected,
# so it ships without having been executed. Counting those as executed would let
# the page claim a run that never happened -- the page states both numbers, and
# this checks both.
_PENDING = [f for f in _EXPS
            if re.search(r"^PENDING=yes\b",
                         open(os.path.join(_EXPDIR, f), encoding="utf-8",
                              errors="replace").read(), re.M)]
NEXP = len(_EXPS)
NDONE = NEXP - len(_PENDING)


def checks(c, a):
    """(label, expected, pattern) triples, built from the artefacts themselves."""
    g, t8 = c["geometry"], c["tau8_attenuation"]
    t8r, ash = c["tau8_replication"], c["arm_shift"]
    pc = c["pi05_confirmation"]
    gb, sc = c["global_beta_curve"], c["scale"]
    ri, ft = c["refimpl"], c["frontier_timing"]
    sq, cst = c["stockquiet"], c["cachesteps"]
    fg = c["deployfrontier"]["fragility"]
    fs = c["deployfrontier"]["seeds"]
    ss = c["statescale"]
    ssr = ss["replication"]
    mx = c["matrix"]
    c7 = c["cache_age_taps7"]
    _b = c7["beta_by_age"]
    _g7 = round(_b["1"] ** 7, 4)
    # Both come from the artefact via canonical, never from arithmetic here:
    # the per-cell median is what the probe computes, the ratio of medians is
    # what a reader would derive from the printed rows, and they differ.
    _e7 = c7["geometric_error_pct_at_max_age"]
    _e7m = c7["geometric_error_pct_at_max_age_of_medians"]
    _spread = round(max(_b[str(k)] for k in (4, 5, 6, 7))
                    - min(_b[str(k)] for k in (4, 5, 6, 7)), 3)
    tl = c["tails"]
    dfr = c["deployfrontier"]["arms"]
    ca = c["cache_age_deployment_ckpt"]
    b7 = c["baselines"]["by_seed"]
    f4, f2 = c["tau4_frontier_set"], c["tau2_frontier_set"]
    fr, oc, rb = c["frontier"], c["oracle_ceiling"], c["robustness"]
    pi = c["pi05_beta_sweep"]["best"]
    fw, p5, dz = (x["held_out_rel_l2"] for x in a)
    out = [
        ("Fast-WAM cosine",   g["fw"]["cosine"],        r"0\.8035"),
        ("Fast-WAM angle",    g["fw"]["angle_deg"],     r"36\.5\^?\\?circ|36\.5"),
        ("pi0.5 angle",       g["pi05"]["angle_deg"],   r"41\.8"),
        ("DreamZero angle",   g["dz"]["angle_deg"],     r"42\.2"),
        ("pi0.5 cosine",      g["pi05"]["cosine"],      r"0\.7453"),
        ("DreamZero cosine",  g["dz"]["cosine"],        r"0\.7413"),
        ("observations",      398520,                   r"398,520"),
        # refimpl: the executed-horizon test. Its whole point is that the gap
        # WIDENS, so both legs are guarded.
        ("refimpl replan10 plain", ri["replan10"]["plain"], r"61\.5"),
        ("refimpl replan32 plain", ri["replan32"]["plain"], r"39\.5"),
        ("refimpl replan10 full",  ri["replan10"]["full"],  r"96\.0"),
        ("refimpl replan32 full",  ri["replan32"]["full"],  r"95\.0"),
        ("refimpl horizon moves",  ri["horizon_moves_plain_by"], r"22\.0"),
        ("c3ache published",       96.4,                    r"96\.4"),
        # frontier latency, quiet host
        ("latency 4 steps",   ft["arms"]["full@4"]["ms"],   r"203\.1"),
        ("latency 6 steps",   ft["arms"]["full@6"]["ms"],   r"255\.4"),
        ("latency full10",    ft["arms"]["full@10"]["ms"],  r"355\.9"),
        ("speedup 4 steps",   ft["arms"]["full@4"]["speedup_vs_full10"], r"1\.75"),
        # stockquiet: the instrumentation overhead, finally measured
        ("recording overhead ms", sq["order_averaged_overhead_ms"], r"10\.18|10\.2"),
        # TWO DIFFERENT PERCENTAGES, and they were confused once. 10.2 ms is 5.1%
        # OF STOCK (10.18/199.9). Restating a ratio against stock instead of full
        # shrinks it by 4.8% (1 - 199.9/210.1). Both are guarded so neither can
        # be substituted for the other again.
        ("recording overhead pct", sq["overhead_pct_of_stock"], r"5\.1\\?%"),
        ("stock-first delta",  sq["runs"]["q_sfirst"]["full_minus_stock_ms"], r"11\.56"),
        ("full-first delta",   sq["runs"]["q_ffirst"]["full_minus_stock_ms"], r"8\.80"),
        ("ratios inflated by", sq["ratios_shrink_by_pct"], r"4\.8"),
        ("4 steps vs stock",  1.67, r"1\.67"),
        ("tau8 vs stock", sq["speedup_vs_stock"]["c3ache_tau8_shrink_skip"], r"1\.36|1\.357"),
        ("tau8 vs full",  sq["speedup_vs_full"]["c3ache_tau8_shrink_skip"],  r"1\.43|1\.426"),
        # cachesteps: caching on a short schedule
        ("cs4 full",   cst["runs"]["cs4"]["full"],   r"95\.0"),
        ("cs4 cached", cst["runs"]["cs4"]["cached"], r"92\.0"),
        ("cs6 full",   cst["runs"]["cs6"]["full"],   r"96\.0"),
        ("cs6 cached", cst["runs"]["cs6"]["cached"], r"93\.0"),
        ("cs4 cells",  cst["runs"]["cs4"]["cells_cached"], r"\b60\b"),
        ("cs6 cells",  cst["runs"]["cs6"]["cells_cached"], r"\b120\b"),
        # deployfrontier: the clean same-host comparison
        ("df 2st ms",      dfr["2st"]["ms"],            r"165\.6"),
        ("df 2st success", dfr["2st"]["success_pct"],   r"93\.3"),
        ("df 4st+t4 ms",   dfr["4st+tau4"]["ms"],       r"187\.9"),
        ("df 4st+t4 sr",   dfr["4st+tau4"]["success_pct"], r"90\.0"),
        ("df 10st+t4 ms",  dfr["10st+tau4"]["ms"],      r"302\.6"),
        ("df 10st+t4 sr",  dfr["10st+tau4"]["success_pct"], r"96\.7"),
        ("df 10st+t4 refresh", dfr["10st+tau4"]["refresh_ms"], r"418\.5"),
        ("df 10st+t4 reuse",   dfr["10st+tau4"]["reuse_ms"],   r"259\.6"),
        ("df 10st+t4 p95", dfr["10st+tau4"]["p95"],     r"455\.9"),
        ("df 10st+t4 max", dfr["10st+tau4"]["max"],     r"505\.7"),
        # the two-seed short-schedule replication
        ("cs 4step seed7", cst["runs"]["cs4_seed7"]["delta"], r"-?3\.0"),
        ("cs 4step p seed7", cst["runs"]["cs4_seed7"]["mcnemar_p"], r"0\.250|0\.25\b"),
        # cache age on the deployment checkpoint -- the mechanism link
        ("cache-age beta k1", ca["beta_by_age"]["1"], r"0\.716"),
        ("cache-age beta k4", ca["beta_by_age"]["4"], r"0\.341"),
        # the illustrative inference budget: measured latency, not control failures
        ("illustrative budget", 500, r"500\\?,?\s*ms|\$500\$"),
        ("10st calls over budget", 6, r"6/367|\$6/367\$"),
        ("10st+t4 calls over budget", 3, r"3/366|\$3/366\$"),
        # paired confidence interval, not just a p-value, on the paired cost
        ("cs4 paired CI lo", cst["paired_ci"]["cs4"]["boot_ci95"][0], r"-8\.0|&minus;8\.0"),
        ("cs4 seed7 CI lo", cst["paired_ci"]["cs4_seed7"]["boot_ci95"][0], r"-7\.0|&minus;7\.0"),
        ("cs6 seed7 CI hi", cst["paired_ci"]["cs6_seed7"]["boot_ci95"][1], r"\+4\.0"),
        # the 30-episode ordering: what the count can and cannot support
        ("dominance count floor", fg["dominates_min"], r"between \$?4\$? and|to \$?4\$? of|4 of 11"),
        ("dominance count ceiling", fg["dominates_max"], r"and \$?10\$? of|\$?10\$? of the eleven|10 of 11"),
        ("smallest paired p", fg["smallest_paired_p"], r"0\.688"),
        ("episodes needed per arm", fg["episodes_needed_per_arm"], r"670"),
        ("factor more episodes", fg["factor_more_episodes"], r"22\\?\s*\\times|22&times;|22\u00d7"),
        # repeated seeds: the ordering is not a property of the configurations
        ("seeds collected", len(fs["seeds"]), r"three seeds|3 seeds"),
        ("dominance at seed 7", fs["dominance_by_seed"]["7"], r"\b5\b"),
        ("cached arms flipping sign", fs["n_sign_flips"], r"[Tt]hree of (?:the )?five|3 of 5"),
        ("observations per arm", fs["observations_per_arm"], r"\b90\b"),
        ("arm pairs tested", fs["pairs_tested"], r"\b55\b"),
        ("smallest state sign-test p", fs["smallest_state_signtest_p"], r"0\.25\b"),
        ("initial states", fs["initial_states"], r"\b30\b"),
        ("cells unanimous pct", fs["pct_state_arm_cells_unanimous"], r"87\\?%|87 ?\\%"),
        # the ranges the workshop paper quotes in place of per-seed values
        ("two-step ms range", fs["two_step_ms_range"][0],
         r"166--171|166.171|166\u2013171", ("workshop",)),
        ("fastest cache ms range", fs["fastest_cache_ms_range"][0],
         r"188--208|188.208|188\u2013208", ("workshop",)),
        # state-scale: the experiment that resolved the success question
        ("statescale states", ss["n_states"], r"\b500\b"),
        ("statescale 2st success", ss["arms"]["2st"]["success_pct"], r"96\.6"),
        ("statescale 10st cache success", ss["arms"]["10st+cache"]["success_pct"], r"93\.6"),
        ("statescale 2st ms", ss["arms"]["2st"]["ms"], r"123\.6"),
        ("statescale 10st cache ms", ss["arms"]["10st+cache"]["ms"], r"221\.8"),
        ("statescale sep diff", ss["pairs"]["2st-10st+cache"]["diff_pts"], r"3\.0"),
        ("statescale sep p", ss["pairs"]["2st-10st+cache"]["sign_p"], r"0\.020|0\.02\b"),
        ("statescale discordant", ss["pairs"]["2st-10st+cache"]["discordant"], r"\b37\b"),
        # the second policy seed: same sign, still excludes zero
        ("statescale seed7 diff", ssr["per_seed"]["7"]["diff_pts"], r"\+?6\.0"),
        ("statescale seed7 ci lo", ssr["per_seed"]["7"]["ci95"][0], r"\+?3\.4"),
        # the completed matrix: reference, plain reuse, correctly fitted global
        ("matrix full success", mx["arms"]["10st full"]["success_pct"], r"96\.4"),
        ("matrix plain success", mx["arms"]["10st plain"]["success_pct"], r"88\.4"),
        ("matrix global success", mx["arms"]["10st global"]["success_pct"], r"91\.4"),
        ("attenuation contribution", mx["pairs"]["attenuation_percell"]["diff"], r"\+?5\.2"),
        ("percell vs pooled global", mx["pairs"]["percell_vs_pooled_global"]["diff"], r"\+?2\.2"),
        # cache ages 5-7: the optimum plateaus where a geometric model collapses
        # The condensed paper rounds to 3dp and the full one keeps 4, so each
        # pattern admits both spellings of the same artefact value.
        ("taps7 beta age1", _b["1"], r"0\.7369|0\.737\b"),
        ("taps7 beta age2", _b["2"], r"0\.5222|0\.522\b"),
        ("taps7 beta age3", _b["3"], r"0\.4174|0\.417\b"),
        ("taps7 beta age4", _b["4"], r"0\.3919|0\.392\b"),
        ("age7 optimal beta", c7["beta_by_age"]["7"], r"0\.3706", "tex page"),
        ("age5 optimal beta", c7["beta_by_age"]["5"], r"0\.3703", "tex page"),
        # The geometric comparison is DERIVED here rather than typed into the
        # prose: a hand-computed error row shipped once with every entry wrong.
        ("geom beta at age7", _g7, r"0\.118"),
        ("geom error at age7", _e7, r"(59\.8|60)\\?%"),
        ("geom error of medians", _e7m, r"68\.2\\?%", "tex page"),
        ("plateau spread ages 4-7", _spread, r"0\.022"),
        # tail latency across every arm, and the formulation difference
        ("tail calls total", tl["total_calls"], r"4,?021"),
        ("tail 2st max", tl["arms"]["2st"]["max"], r"183\.6"),
        ("tail 10st over", tl["arms"]["10st"]["over_budget"], r"6\s*/\s*367|6/367"),
        ("pooled LS scalar", 0.7384, r"0\.7384|0\.738"),
        # The headline counts. These sat at 23,096 over 57 runs and "17 of 17
        # experiments" while the repository had grown to 28,736 over 77 and 18;
        # nothing guarded them, so they aged quietly for five experiments.
        # Derived, not typed -- these went stale twice as literals.
        ("closed-loop episodes", sc["closed_loop_episodes"],
         re.escape(f"{sc['closed_loop_episodes']:,}"), ("page",)),
        ("closed-loop runs",     sc["closed_loop_runs"],
         rf"{sc['closed_loop_runs']} runs", ("page",)),
        # Derived from the filesystem, not typed. This count went stale twice --
        # at 13 when there were 18, and at 18 when there were 21 -- because it
        # was a literal that nothing recomputed when an experiment was added.
        ("experiments executed", NDONE, rf"{NDONE} of {NEXP}", ("page",)),
        ("experiment count",     NEXP, _word(NEXP), ("page",)),
        # The five-seed re-collection is the reported tau=8 row.
        ("tau8 full",         t8r["mean_full"],         r"94\.8"),
        ("tau8 plain",        t8r["mean_cached"],       r"67\.6"),
        ("tau8 attenuated",   t8r["mean_attenuated"],   r"82\.8"),
        ("tau8 delta",        t8r["mean_delta"],        r"\+?15\.2"),
        ("tau8 CI low",       t8r["seed_level"]["ci95"][0], r"10\.5"),
        ("tau8 CI high",      t8r["seed_level"]["ci95"][1], r"19\.9"),
        ("tau8 sign p",       t8r["seed_level"]["sign_test_p"], r"0\.06(25)?\b"),
        ("tau8 pooled p",     t8r["pooled_p"],          r"2\.6e-11"),
        # The superseded nine-run figure, quoted in the paragraph that retires it.
        ("tau8 old delta",    t8["mean_delta"],         r"\+?16\.0"),
        ("tau8 old sign p",   t8["seed_level"]["sign_test_p"], r"0\.004"),
        # A reference arm that should not move.
        # The arm-shift table is page-only now: the paper carries the retraction
        # in prose, without the interval, because the comparison is confounded
        # by GPU and no longer supports a quoted effect size.
        ("arm shift verified", ash["verified_single"],  r"-4\.0", ("page",)),
        ("arm shift CI",      ash["with_filename_seeds"]["ci95"],
         r"\[-4\.5, -0\.7\]", ("page",)),
        ("arm shift gpu confound", ash["confounded_by_gpu"],
         r"A100|A6000", ("page",)),
        ("arm shift not arm order", ash["attributable_to_arm_order"],
         r"predicts\s+no change|isolates neither", ("page",)),
        # pi0.5 seed-level tests, which the pooled p must never stand in for.
        ("pi05 seed-level t p", pc["seed_level_t_p"], r"p=0\.026|p = 0\.026"),
        ("pi05 seed-level sign p", pc["seed_level_sign_p"], r"p=0\.25|p = 0\.25|0\.25"),
        ("tau4 delta",        f4["mean_delta"],         r"\+?4\.6"),
        ("tau4 CI low",       f4["seed_level"]["ci95"][0], r"0\.6"),
        ("tau4 CI high",      f4["seed_level"]["ci95"][1], r"8\.6"),
        ("tau4 plain",        f4["mean_cached"],        r"87\.8"),
        ("tau2 delta",        f2["mean_delta"],         r"-1\.2"),
        ("tau2 CI",           f2["seed_level"]["ci95"], r"\[-4\.2, \+?1\.8\]"),
        ("frontier delta",    fr["mean_delta"],         r"-2\.8"),
        ("oracle vs fitted",  oc["oracle_minus_fitted_pts"], r"0\.5 points"),
        ("oracle CI",         oc["ci95"],               r"\[-3\.5, \+?4\.5\]"),
        ("pi0.5 best beta",   pi["beta"],               r"0\.25"),
        ("pi0.5 sweep delta", pi["delta_vs_beta1"],     r"\+?8\.5"),
        ("pi0.5 sweep p",     pi["p"],                  r"0\.0015"),
        ("alpha gain FW",     1.2,                      r"1\.2/|\+1\.2%"),
        ("beta gain FW",      7.7,                      r"7\.7/|\+7\.7%"),
        # One global weight, per backbone (results page Result 1's second table).
        ("global b=1 FW",     gb["fw"]["beta_1"],       r"0\.660\b", ("page",)),
        ("global b=0.8 FW",   gb["fw"]["beta_0.8"],     r"0\.630\b", ("page",)),
        ("global b=1 pi",     gb["pi05"]["beta_1"],     r"0\.753\b", ("page",)),
        ("global b=0.8 pi",   gb["pi05"]["beta_0.8"],   r"0\.705\b", ("page",)),
        ("global b=1 dz",     gb["dz"]["beta_1"],       r"0\.705\b", ("page",)),
        ("global b=0.8 dz",   gb["dz"]["beta_0.8"],     r"0\.671\b", ("page",)),
        ("hold rel FW",       fw["hold"],               r"0\.6530"),
        ("beta rel FW",       fw["beta"],               r"0\.6025"),
        ("hold rel pi",       p5["hold"],               r"0\.7565"),
        ("beta rel pi",       p5["beta"],               r"0\.6754"),
        ("hold rel dz",       dz["hold"],               r"0\.7079"),
        ("beta rel dz",       dz["beta"],               r"0\.6429"),
    ]
    out += [
        ("latency tau2 speedup",  1.201, r"1\.201"),
        ("latency tau4 speedup",  1.361, r"1\.361"),
        ("latency tau8 speedup",  1.459, r"1\.46|1\.459"),
        ("latency full baseline", 470.4, r"470\.4"),
        ("dce substitute ms",     371.4, r"371\.4"),
        ("dce full ms",           502.6, r"502\.6"),
        ("time saved tau8",       31.5,  r"31\.5"),
        ("realised fraction",     0.44,  r"0\.42--0\.45|0\.44"),
        # The frontier's ms column now comes from results/frontier_timing/ (one
        # quiet host, shared across seeds), NOT from each baselines run's own
        # timing. Those per-run figures -- 254.1, 309.3, 457.7 -- were collected
        # while the host load swung from 63 to 1.8 and are no longer quoted.
        ("frontier steps4 ms",      ft["arms"]["full@4"]["ms"],  r"203\.1"),
        ("frontier steps4 success", 96.5,  r"96\.5"),
        ("frontier tau8 ms",  ft["arms"]["c3ache_tau8_shrink_skip@10"]["ms"], r"256\.8"),
        ("frontier tau8 success",   83.0,  r"83\.0"),
        ("frontier full ms",        ft["arms"]["full@10"]["ms"], r"355\.9"),
        # seed 7 replication
        ("seed7 steps4",  b7["7"]["arms"]["steps4_seed7:full_4steps"]["success_pct"], r"97\.5"),
        ("seed7 tau8",    b7["7"]["arms"]["cache_seed7:c3ache_tau8_shrink_skip"]["success_pct"], r"87\.5"),
        ("seed7 tau4",    b7["7"]["arms"]["cache_seed7:c3ache_tau4_shrink_skip"]["success_pct"], r"89\.5"),
        # The four-step comparison on BOTH seeds. Against tau=2 it is unresolved
        # on either, which is the qualification the deployment claim needs.
        ("4step vs tau8 seed42", 13.5, r"\+?13\.5"),
        ("4step vs tau8 seed7",  10.0, r"\+?10\.0"),
        ("4step vs tau2 seed7",   3.5, r"\+?3\.5"),
        ("global beta success",     91.5,  r"91\.5"),
        ("per-cell success",        94.5,  r"94\.5"),
        ("global vs per-cell p",    0.24,  r"0\.24"),
        # Derived from the artefacts, not typed. These three were hardcoded to
        # +7.57 / [+2.26, +12.87] -- the values canonical_numbers.py produced
        # while it averaged per-seed deltas that had already been rounded to one
        # decimal. The guard therefore certified the error it existed to catch.
        ("pi05 confirm mean",       pc["mean_delta"],
         r"\+?" + re.escape(f"{pc['mean_delta']:.2f}")),
        ("pi05 confirm CI",         pc["ci95"],
         r"\[\+?" + re.escape(f"{pc['ci95'][0]:.2f}") + r", \+?"
         + re.escape(f"{pc['ci95'][1]:.2f}") + r"\]"),
        ("pi05 confirm sd",         pc["sd"], re.escape(f"{pc['sd']:.2f}")),
        ("pi05 confirm t",          pc["t_crit"], re.escape(f"{pc['t_crit']:.3f}")),
        ("pi05 confirm better",     pc["pooled_better"], r"\b67/33\b"),
        ("pi05 confirm p",          0.00087, r"0\.00087"),
        ("pi05 seed7 delta",        6.0,   r"\+?6\.0"),
        ("pi05 seed23 delta",       10.0,  r"\+?10\.0"),
        ("pi05 seed42 delta",       6.7,   r"\+?6\.7"),
        ("pi05 pooled",             None,  r"67/33|67 better"),
    ]
    for nm, v in (("Fast-WAM", a[0]), ("pi0.5", a[1]), ("DreamZero", a[2])):
        out.append((f"{nm} two-tap coeff sum", v["two_tap_coeff_sum"]["median"],
                    re.escape(f"{v['two_tap_coeff_sum']['median']:.3f}")))
    for arm in rb["arms"]:
        if arm["perturbation"] == "bright:0.05":
            out.append(("brightness delta", arm["delta"], r"\+?13\.3"))
        if arm["perturbation"] == "cam:0.005":
            out.append(("camera delta", arm["delta"], r"\+?1\.7"))
    out.append(("dissociation CI", rb["dissociation"]["ci95"], r"\[-1\.7, \+?25\.0\]"))
    return out



def workshop_checks(c, a):
    """Current four-page claims, with patterns derived from their data sources.

    Composite table-row and interval patterns prevent an unrelated occurrence of
    the same number from certifying a corrupted result. Legacy full-paper/page
    contracts remain in checks(); deleted exploratory claims are not required in
    the condensed workshop paper.
    """
    import contextlib
    import io
    import random
    import statistics
    import math
    import report_matrix
    import report_statescale

    def number(value, digits=1, signed=False):
        text = f"{value:.{digits}f}"
        return (r"\+?" if signed and value >= 0 else "") + re.escape(text)

    def interval(values, digits=1):
        return (r"\[\s*" + number(values[0], digits, True) + r"\s*,\s*"
                + number(values[1], digits, True) + r"\s*\]")

    rows = []
    def add(label, value, pattern=None, digits=1):
        rows.append((label, value, pattern if pattern is not None
                     else r"(?<![\d.])" + number(value, digits) + r"(?![\d.])"))

    families = json.load(open(os.path.join(R, "residual", "families_summary.json")))
    by_label = {item["label"]: item for item in families}
    for label, key, latex_label in (("Fast-WAM", "fw", "Fast-WAM"),
                                  ("pi0.5", "pi05", r"\\pifive"),
                                  ("DreamZero", "dz", "DreamZero")):
        family = by_label[label]
        fitted = {row[0]: row for row in family["rows"]}
        def gain(name):
            row = fitted[name]
            return 100 * (1 - row[2] / row[3])
        values = [c["geometry"][key]["magnitude"], c["geometry"][key]["angle_deg"],
                  gain("shrink1 one coefficient set for all cells"), gain("shrink1 [tap1]")]
        pattern = (latex_label + r"\s*&\s*" + number(values[0], 4) + r"\s*&\s*"
                   + number(values[1]) + r"\^\\circ\s*&\s*" + number(values[2])
                   + r"\\?%\s*&\s*" + number(values[3]) + r"\\?%")
        add(f"geometry row {label}", values, pattern)
    add("replay episodes LIBERO", c["geometry"]["fw"]["episodes"], digits=0)
    add("replay episodes DreamZero", c["geometry"]["dz"]["episodes"], digits=0)

    for tau, key in ((2, "tau2_frontier_set"), (4, "tau4_frontier_set"), (8, "tau8_replication")):
        row = c[key]
        values = [row["mean_full"], row["mean_cached"], row["mean_attenuated"], row["mean_delta"]]
        pattern = (rf"\b{tau}\s*&\s*" + r"\s*&\s*".join(number(v) for v in values[:3])
                   + r"\s*&\s*" + number(values[3], signed=True) + r"\s*"
                   + interval(row["seed_level"]["ci95"]))
        add(f"tau{tau} complete success row", values + row["seed_level"]["ci95"], pattern)
    add("tau8 attenuation gain", c["tau8_replication"]["mean_delta"])
    add("tau8 sign test", c["tau8_replication"]["seed_level"]["sign_test_p"], digits=4)
    add("published reference success", 96.4)
    add("local horizon10 plain", c["refimpl"]["replan10"]["plain"])
    add("local horizon32 plain", c["refimpl"]["replan32"]["plain"])

    # The manuscript uses report_matrix.main's frozen draw order. The older
    # canonical summary uses another draw order and has [2.0,8.4] rather than
    # [2.2,8.4]; do not silently substitute one Monte Carlo interval for another.
    random.seed(20260919)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        report_matrix.main(os.path.join(R, "matrix"))
    matrix_text = buffer.getvalue()
    parsed = {}
    for name in ("attenuation's contribution (per-cell)", "per-cell vs POOLED-fit global"):
        match = re.search(re.escape(name) + r"\s+([+-]\d+\.\d+)%\s+\d+\s+\[\s*([+-]\d+\.\d+),\s*([+-]\d+\.\d+)\]", matrix_text)
        if not match:
            raise ValueError(f"Cannot parse current matrix report: {name}")
        parsed[name] = [float(x) for x in match.groups()]
    mx = c["matrix"]
    values = [mx["arms"][name]["success_pct"] for name in
              ("10st full", "10st plain", "10st global", "10st per-cell")]
    pattern = r"/".join(number(v) for v in values) + r"\\?%"
    add("matrix four success rates", values, pattern)
    for label, key in (("matrix attenuation", "attenuation's contribution (per-cell)"),
                       ("matrix percell versus pooled", "per-cell vs POOLED-fit global")):
        values = parsed[key]
        add(label + " difference", values[0])
        add(label + " interval", values[1:], interval(values[1:]))
    add("pooled scalar", 0.7384, digits=3)

    sensitivity = json.load(open(os.path.join(R, "schedule_sensitivity.json")))
    for collection in sensitivity["collections"]:
        record = collection["contrasts"]["excluding_pilot_470"][0]
        add(f"pilot-excluded seed{collection['seed']} delta", record["difference_pp_equal_task"])
        add(f"pilot-excluded seed{collection['seed']} interval", record["ci95_pp"], interval(record["ci95_pp"], 2))
    add("pilot-excluded states", 470, digits=0)

    ss = c["statescale"]
    add("schedule distinct states", ss["n_states"], digits=0)
    for arm in ("2st", "4st", "4st+cache", "10st+cache"):
        add(f"seed42 {arm} latency", ss["arms"][arm]["ms"])
    for arm in ("2st", "10st+cache"):
        add(f"seed42 {arm} success", ss["arms"][arm]["success_pct"])
    for label, key in (("seed42 twostep cache", "2st-10st+cache"),
                       ("seed42 fourstep caching cost", "4st-4st+cache")):
        add(label + " difference", ss["pairs"][key]["diff_pts"])
        add(label + " interval", ss["pairs"][key]["ci95"], interval(ss["pairs"][key]["ci95"]))
    sr = ss["replication"]["per_seed"]["7"]
    add("seed7 twostep success", sr["arms"]["2st"])
    add("seed7 tenstep cache success", sr["arms"]["10st+cache"])
    add("seed7 twostep cache difference", sr["diff_pts"])
    add("seed7 twostep cache interval", sr["ci95"], interval(sr["ci95"]))
    random.seed(20260919)
    seed7 = report_statescale.summary(os.path.join(R, "statescale"), seed=7)
    cost = seed7["pairs"]["4st-4st+cache"]
    add("seed7 fourstep caching cost", cost["diff_pts"])
    add("seed7 fourstep caching interval", cost["ci95"], interval(cost["ci95"]))

    add("recording overhead ms", c["stockquiet"]["order_averaged_overhead_ms"])
    add("recording overhead percent", c["stockquiet"]["overhead_pct_of_stock"])
    # canonical.deployfrontier historically merged seed variants. Read the
    # exact seed-42 pilot instead; its tail convention is nearest rank, whereas
    # canonical.tails uses interpolated percentiles.
    pilot = json.load(open(os.path.join(R, "deployfrontier", "d10.json")))
    if pilot["meta"]["seed"] != 42:
        raise ValueError("Expected seed-42 pilot artifact")
    cycle_pairs = [(ms, refresh) for row in pilot["rows"]
                   if row["config"] == "c3ache_tau4_shrink_skip" and not row.get("warmup")
                   for ms, refresh in zip(row["cycle_ms"], row["cycle_refresh"], strict=True)]
    times = sorted(ms for ms, _ in cycle_pairs)
    tail = {"refresh_ms": statistics.mean(ms for ms, refresh in cycle_pairs if refresh),
            "reuse_ms": statistics.mean(ms for ms, refresh in cycle_pairs if not refresh),
            "p95": times[math.ceil(0.95 * len(times)) - 1]}
    for field in ("refresh_ms", "reuse_ms", "p95"):
        add(f"pilot cache {field}", tail[field])
    calls, two_step_times = 0, []
    for steps in (2, 3, 4, 6, 8, 10):
        run = json.load(open(os.path.join(R, "deployfrontier", f"d{steps}.json")))
        if run["meta"]["seed"] != 42:
            raise ValueError("Expected seed-42 pilot artifact")
        for row in run["rows"]:
            if not row.get("warmup"):
                calls += len(row["cycle_ms"])
                if steps == 2 and row["config"] == "stock":
                    two_step_times.extend(row["cycle_ms"])
    add("pilot two-step maximum", max(two_step_times))
    add("pilot total calls", calls, re.escape(f"{calls:,}"))
    return rows


def main():
    tex = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEX
    if not os.path.exists(tex):
        print(f"  SKIP  no write-up at {tex}"); return 0
    raw = open(tex, encoding="utf-8", errors="replace").read()
    # Match against plain text, not LaTeX: "$7.7$/$10.7$" and "7.7/10.7" should
    # both satisfy the same pattern, so the checks stay readable and survive
    # ordinary re-typesetting. Strips math delimiters, thin spaces, padding
    # macros and line breaks that fall inside a quantity.
    src = re.sub(r"[$]|\\[,;! ]|\\hphantom\{[^{}]*\}", "", raw)
    src = re.sub(r"\s*\n\s*", " ", src)
    # Same checks must pass over the LaTeX source and over the HTML results page,
    # so normalise the ways the two spell one number: LaTeX groups digits as
    # 398{,}520 and writes a hyphen-minus, HTML writes 398,520 and U+2212.
    src = src.replace("{,}", ",").replace("\u2212", "-").replace("\u2013", "-")
    # Scientific notation: LaTeX writes 2.6\times10^{-11}, HTML writes 2.6x10(-11)
    # in superscript digits. Fold both to "2.6e-11" so one pattern covers them.
    sup = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")
    src = re.sub(r"(\d(?:\.\d+)?)\s*\\?times\s*10\^?\{?(-?\d+)\}?",
                 lambda m: f"{m.group(1)}e{m.group(2)}", src)
    src = re.sub(r"(\d(?:\.\d+)?)\s*\u00d7\s*10([\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u207b]+)",
                 lambda m: f"{m.group(1)}e{m.group(2).replace(chr(0x207b), '-').translate(sup)}", src)
    # Strip HTML tags two ways and accept a match in EITHER. Deleting a tag
    # outright fuses adjacent table cells -- "+7.56" and "67/33" become
    # "+7.5667/33", which silently destroys a word boundary and fails a pattern
    # that is actually present. Replacing it with a space instead splits any
    # figure a tag runs through. Neither alone is safe; both together are.
    variants = [re.sub(r"<[^>]{0,200}>", sep, src) for sep in ("", " ")]
    variants = [re.sub(r"\s+", " ", v) for v in variants]
    src = variants[0]
    c = json.load(open(os.path.join(R, "canonical_numbers.json")))
    ap = os.path.join(R, "alpha_constraint.json")
    if not os.path.exists(ap):
        print("  alpha_constraint.json missing -- run probes/alpha_constraint.py"); return 1
    a = json.load(open(ap))

    # A check may name the documents it applies to. The results page carries
    # tables the 7-page paper has no room for, so requiring every figure in both
    # would force either padding the paper or deleting the guard.
    # THREE DOCUMENTS, DIFFERENT CONTRACTS. The full paper and the results page
    # must carry every figure their scope admits. The workshop paper is a 4-page
    # condensation and legitimately omits most of them -- but everything it DOES
    # state must still match the artefacts, and a figure that drifts must fail
    # rather than read as "condensed out". So it carries an explicit expected
    # set, probes/workshop_figures.json, generated from what it states today:
    # dropping a figure is then a deliberate edit to that file, and changing one
    # to a wrong value fails as a missing figure.
    doc = "page"
    if tex.endswith(".tex"):
        doc = "workshop" if "workshop" in os.path.normpath(tex).split(os.sep) else "tex"
    expect = None
    if doc == "workshop":
        wf = os.path.join(HERE, "workshop_figures.json")
        if os.path.exists(wf):
            expect = set(json.load(open(wf)))
    available = workshop_checks(c, a) if doc == "workshop" else checks(c, a)
    if expect is not None:
        names = {r[0] for r in available}
        if expect != names:
            print("  FAIL: workshop_figures.json differs from the declared current claim contract")
            print("  unknown:", sorted(expect - names), "undeclared:", sorted(names - expect))
            return 1
    rows = [r if len(r) == 4 else (*r, None) for r in available]
    rows = [r for r in rows if r[3] is None or doc in r[3] or (doc == "workshop" and "tex" in r[3])]
    if expect is not None:
        rows = [r for r in rows if r[0] in expect]
    bad = []
    for label, value, pat, _ in rows:
        if not any(re.search(pat, v) for v in variants):
            bad.append((label, value, pat))
    n = len(rows)
    print(f"\n  {os.path.relpath(tex)}: {n - len(bad)} of {n} figures found")
    if bad:
        print("\n  MISSING -- the artefacts say this, the write-up does not:\n")
        for label, value, pat in bad:
            print(f"    {label:28} {value!s:22} /{pat}/")
        print("\n  Either the write-up is stale or the check is. Fix whichever is wrong;")
        print("  do not delete the row to make this pass.")
        return 1
    print("  every checked figure in the write-up matches the committed artefacts\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The frontier a deployment would actually see, plus refresh/reuse tails.

usage: report_deployfrontier.py <dir>

Uncached arms here are `stock` -- full compute with residual recording off --
so these are not speedups against our own instrumentation. Cached arms keep
recording on refresh cycles, which for them is not instrumentation but the cost
of building the cache.
"""
import glob, json, os, statistics as st, sys


def pct(v, q):
    if not v: return float("nan")
    s = sorted(v); i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no deployfrontier artefacts in {d} -- not yet collected.")
        return 0

    arms, loads = [], []
    for p in files:
        j = json.load(open(p)); m = j.get("meta") or {}
        steps = m.get("num_inference_steps")
        pv = m.get("provenance") or {}
        la = (pv.get("loadavg_1_5_15") or [0])[0]; nc = pv.get("cpu_count") or 1
        loads.append((la, nc))
        by, tim, cyc = {}, {}, {}
        for r in j["rows"]:
            by.setdefault(r["config"], []).append(bool(r["success"]))
            if "ms_per_cycle" in r and not r.get("warmup"):
                tim.setdefault(r["config"], []).append(r["ms_per_cycle"])
                for ms, ref in zip(r.get("cycle_ms", []), r.get("cycle_refresh", [])):
                    cyc.setdefault(r["config"], {"refresh": [], "reuse": []})[
                        "refresh" if ref else "reuse"].append(ms)
        for c, v in tim.items():
            lbl = (f"{steps} steps" if c == "stock" else
                   f"{steps} steps + {c.replace('c3ache_','').replace('_shrink_skip','')}")
            arms.append({"arm": lbl, "cached": c != "stock", "steps": steps,
                         "ms": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
                         "n": len(v),
                         "sr": 100.0 * sum(by[c]) / len(by[c]), "eps": len(by[c]),
                         "cyc": cyc.get(c, {"refresh": [], "reuse": []})})

    hi = max(la / nc for la, nc in loads) if loads else 0
    print(f"\n  host load at collection: {hi:.2f} per core"
          f"{'  -- CONTENDED, timing not trustworthy' if hi > 0.5 else ''}")

    print("\n  LATENCY AGAINST SUCCESS  (uncached arms carry NO recording)\n")
    print(f"    {'arm':26} {'ms/cycle':>9} {'sd':>6} {'success':>8} {'n':>4}   verdict")
    for a in sorted(arms, key=lambda a: a["ms"]):
        dom = [o for o in arms if o["arm"] != a["arm"] and o["ms"] < a["ms"]
               and o["sr"] >= a["sr"]]
        v = f"dominated by {dom[0]['arm']}" if dom else "ON THE FRONTIER"
        print(f"    {a['arm']:26} {a['ms']:9.1f} {a['sd']:6.1f} {a['sr']:7.1f}% "
              f"{a['eps']:4}   {v}")

    front = [a for a in arms
             if not [o for o in arms if o["arm"] != a["arm"] and o["ms"] < a["ms"]
                     and o["sr"] >= a["sr"]]]
    con = [a for a in front if a["cached"]]
    print(f"\n    {len(front)} of {len(arms)} on the frontier; {len(con)} cached"
          f"{': ' + ', '.join(a['arm'] for a in con) if con else ''}.")
    if not con:
        print("    No cached arm survives against clean uncached baselines.")
    else:
        print("    A cached arm survives even against uninstrumented baselines,")
        print("    which is a stronger statement than the old frontier could make.")

    print("\n  REFRESH AGAINST REUSE, and the tail\n")
    print(f"    {'arm':26} {'refresh':>9} {'reuse':>9} {'p50':>7} {'p95':>7} {'max':>7}")
    for a in sorted(arms, key=lambda a: a["ms"]):
        R, U = a["cyc"]["refresh"], a["cyc"]["reuse"]
        allc = R + U
        if not allc: continue
        print(f"    {a['arm']:26} {(st.mean(R) if R else float('nan')):9.1f} "
              f"{(st.mean(U) if U else float('nan')):9.1f} "
              f"{pct(allc,0.50):7.1f} {pct(allc,0.95):7.1f} {pct(allc,1.0):7.1f}")
    # AN ILLUSTRATIVE INFERENCE BUDGET, without which a tail latency is just a
    # number. LIBERO runs at 20 Hz and the policy
    # replans every `replan` environment steps, so one chunk prediction has
    # replan/20 seconds. Read from the artefact rather than assumed.
    #
    # This is deliberately NOT called a control deadline met or missed. These are
    # measured inference times; the simulator never tests a controller's response
    # to a late action, so an over-budget call here is not an observed failure.
    _replan = None
    for p_ in files:
        _replan = (json.load(open(p_)).get("meta") or {}).get("replan") or _replan
    if _replan:
        deadline = 1000.0 * _replan / 20.0
        print(f"\n  AGAINST AN ILLUSTRATIVE {deadline:.0f} ms INFERENCE BUDGET")
        print(f"    ({_replan} actions executed per inference at 20 Hz)\n")
        print(f"    {'arm':16} {'max ms':>8} {'over':>6} {'% over':>8}  {'calls':>7}")
        late = []
        for a in sorted(arms, key=lambda a: -(max(a["cyc"]["refresh"] + a["cyc"]["reuse"])
                                              if (a["cyc"]["refresh"] + a["cyc"]["reuse"]) else 0)):
            allc = a["cyc"]["refresh"] + a["cyc"]["reuse"]
            if not allc: continue
            over = sum(1 for x in allc if x > deadline)
            flag = "  <<< OVER" if over else ""
            if over: late.append(a["arm"])
            print(f"    {a['arm']:16} {max(allc):8.1f} {over:6d} "
                  f"{100.0*over/len(allc):7.1f}% {len(allc):8}{flag}")
        if late:
            print(f"\n    {len(late)} arm(s) have calls over the illustrative budget: "
                  f"{', '.join(late)}.")
            print("    Their MEANS are all comfortably inside it -- which is the point")
            print("    of recording per cycle: the amortised mean hides these calls.")
            print("    Stated carefully: these are measured inference times, not")
            print("    observed control failures. The simulator does not test a")
            print("    controller's response to a late action, so this bounds when a")
            print("    budget WOULD be at risk rather than showing anything went wrong.")
        else:
            print("\n    Every call is inside the budget.")

    print("\n    An uncached arm has no reuse cycles, so its refresh column is its")
    print("    only column. For a cache the two differ by construction, and the")
    print("    p95 and max are what a deadline-bound loop would actually meet:")
    print("    a mean hides the refresh spike that a periodic cache guarantees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

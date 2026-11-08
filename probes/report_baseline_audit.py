"""Validate and report fresh-process baseline-audit collections on the CPU.

    python probes/report_baseline_audit.py RESULTS_ROOT --json NEW_REPORT.json
    python probes/report_baseline_audit.py RESULTS_ROOT --source-manifest MANIFEST \
        --figure NEW_FIGURE.pdf

RESULTS_ROOT contains original/{smoke,pilot,full}/seed42 (or --variant optional).
Every phase and seed is reported separately. A smoke or pilot cannot become a
500-state result, and an incomplete or mismatched collection is rejected. The
stock-repeat check concerns recorded outcomes and lengths, not tensor identity.
The same-input gate supplies a separate numerical identity check. Neither test
establishes equivalence to an unspecified published checkpoint or protocol.

Intervals are deterministic, pointwise bootstrap intervals over paired initial
states in one fixed benchmark collection and policy seed. They do not measure
variation across policy seeds or robot tasks outside this collection. Latency is
the mean of episode-level cycle means, excluding the first episode of each task
and arm. Cycle-pooled refresh/reuse summaries are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

PHASES = {"smoke": (1, 2), "pilot": (10, 5), "full": (10, 50)}
ARMS = {"00_stock_a": "stock", "01_whole_tau8": "whole_tau8",
        "02_block_tau8": "c3ache_tau8", "03_stock_b": "stock"}
LABELS = {"00_stock_a": "Stock, 10 steps", "01_whole_tau8": "Whole-stack, tau=8",
          "02_block_tau8": "Blockwise, tau=8", "03_stock_b": "Stock repeat"}
RUNTIME_FIELDS = ("source_bundle_sha256", "checkpoint_sha256", "checkpoint_state_coverage", "dataset_stats_sha256",
                  "dataset_stats", "sigma_shift_override", "effective_action_sigma_shift",
                  "action_infer_mode", "compile_action_infer", "model_class", "mixed_precision")
PROVENANCE_FIELDS = ("commit", "host_tag", "gpu", "python", "torch", "mujoco", "ckpt")
BOOTSTRAP_SEED = 20260921
REPEAT_FIELDS = ("success", "steps", "cycles", "init_state", "refresh_cycles", "reuse_cycles")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_number(value):
    return (type(value) in (int, float) and math.isfinite(value) and value > 0)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def bootstrap_ci(values, *, label, samples=10000, scale=1.0):
    """Paired differences enter as one vector; never resample arms separately."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            "Bootstrap requires a nonempty finite vector")
    seed = int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 256):
        indices = rng.integers(0, len(values), size=(min(256, samples - start), len(values)))
        means.extend(values[indices].mean(axis=1))
    return [float(x * scale) for x in np.quantile(means, [0.025, 0.975], method="linear")]


def validate_runtime(runtime, variant):
    require(isinstance(runtime, dict), "Missing runtime metadata")
    for field in RUNTIME_FIELDS:
        require(field in runtime and runtime[field] is not None, f"Missing runtime.{field}")
    for field in ("source_bundle_sha256", "checkpoint_sha256", "dataset_stats_sha256"):
        require(is_sha(runtime[field]), f"Invalid runtime.{field}")
    require(type(runtime["compile_action_infer"]) is bool, "Compiler mode must be boolean")
    require(runtime["mixed_precision"] == "bf16", "Audit requires recorded BF16 mode")
    require(runtime["action_infer_mode"] == "first_frame", "Audit requires first-frame inference")
    coverage = runtime["checkpoint_state_coverage"]
    require(isinstance(coverage, dict) and coverage.get("verified") is True
            and coverage.get("action_keys", 0) > 0 and coverage.get("video_keys", 0) > 0,
            "Checkpoint must cover both trained experts")
    shift = 5.0 if variant == "original" else 1.0
    require(runtime["effective_action_sigma_shift"] == shift and runtime["sigma_shift_override"] == shift,
            f"Wrong explicit/effective scheduler shift for {variant}")
    expected_stats = f"libero_{'uncond' if variant == 'original' else 'optional_idm'}_2cam224_dataset_stats.json"
    require(runtime["dataset_stats"] == expected_stats, f"Wrong dataset stats for {variant}")


def identity(meta):
    return {"runtime": {k: meta["runtime"][k] for k in RUNTIME_FIELDS},
            "provenance": {k: meta["provenance"][k] for k in PROVENANCE_FIELDS},
            "initial_latent": meta["initial_latent"]}


def validate_arm(data, tag, phase, variant, seed):
    nt, ne = PHASES[phase]
    meta, rows = data.get("meta", {}), data.get("rows", [])
    require(meta.get("suite") == "libero_goal" and meta.get("perturb") == "none", "Wrong suite or perturbation")
    require(meta.get("seed") == seed and meta.get("configs") == [ARMS[tag]], "Wrong seed or arm metadata")
    require(meta.get("n_tasks") == nt and meta.get("n_episodes") == ne,
            f"{phase} requires {nt} tasks x {ne} initial states")
    for field, expected in (("num_inference_steps", 10), ("replan", 10), ("no_reseed", 0),
                            ("timing", 1), ("warmup_episodes", 1), ("skip_cells", 240)):
        require(meta.get(field) == expected, f"Wrong or missing {field}")
    require(isinstance(meta.get("initial_latent"), str) and bool(meta["initial_latent"]), "Missing initial latent digest")
    validate_runtime(meta.get("runtime"), variant)
    provenance = meta.get("provenance", {})
    for field in PROVENANCE_FIELDS:
        require(provenance.get(field) is not None, f"Missing provenance.{field}")
    expected_ckpt = f"libero_{'uncond' if variant == 'original' else 'optional_idm'}_2cam224.pt"
    require(provenance["ckpt"] == expected_ckpt, "Checkpoint basename disagrees with variant")
    require(isinstance(rows, list) and len(rows) == nt * ne, f"Incomplete {tag}: need {nt * ne} rows")
    keyed = {}
    for row in rows:
        require(type(row.get("task")) is int and type(row.get("episode")) is int, "Invalid task/episode index")
        key = row["task"], row["episode"]
        require(key not in keyed, f"Duplicate task-state {key} in {tag}")
        require(row.get("config") == ARMS[tag] and row.get("suite") == "libero_goal", "Wrong row arm or suite")
        require(type(row.get("success")) is bool, "Success must be a recorded boolean")
        require(isinstance(row.get("init_state"), str) and bool(row["init_state"]), "Missing initial-state digest")
        for field in ("cycles", "steps", "refresh_cycles", "reuse_cycles", "denoise_calls"):
            require(type(row.get(field)) is int and row[field] >= 0, f"Invalid {field} in {tag}/{key}")
        n = row["cycles"]
        require(n > 0 and row["denoise_calls"] == n, "Missing/unequal cycle timing coverage")
        refresh = n if ARMS[tag] == "stock" else (n + 7) // 8
        require(row["refresh_cycles"] == refresh and row["reuse_cycles"] == n - refresh,
                f"Wrong refresh/reuse schedule in {tag}/{key}")
        require(type(row.get("warmup")) is bool and row["warmup"] == (key[1] == 0), "Incorrect warmup flag")
        require(positive_number(row.get("ms_per_cycle")), "Missing finite episode latency")
        cycle_ms, cycle_refresh = row.get("cycle_ms", []), row.get("cycle_refresh", [])
        require(len(cycle_ms) == n and len(cycle_refresh) == n, "Incomplete cycle timing arrays")
        require(all(positive_number(v) for v in cycle_ms), "Invalid cycle latency")
        require(all(type(v) is bool for v in cycle_refresh), "Cycle refresh flags must be boolean")
        expected_flags = [True if ARMS[tag] == "stock" else k % 8 == 0 for k in range(n)]
        require(cycle_refresh == expected_flags, "Recorded cycle refresh sequence disagrees with schedule")
        require(abs(sum(cycle_ms) / n - row["ms_per_cycle"]) <= 0.0011,
                "Episode mean disagrees with rounded per-cycle times")
        if ARMS[tag] == "whole_tau8":
            require(row.get("whole_cache_hits") == (n - refresh) * 8 and row.get("whole_cache_misses") == 0,
                    "Whole-stack observed cache hits/misses do not match selected reuse steps")
        keyed[key] = row
    require(set(keyed) == {(t, e) for t in range(nt) for e in range(ne)}, "Incomplete expected task-state grid")
    require(len({r["init_state"] for r in keyed.values()}) == nt * ne, "Repeated initial states within arm")
    return meta, keyed


def cycle_summary(rows):
    result = {"episodes": len(rows), "cycles": sum(r["cycles"] for r in rows),
              "refresh_cycles": sum(r["refresh_cycles"] for r in rows),
              "reuse_cycles": sum(r["reuse_cycles"] for r in rows)}
    for flag, label in ((True, "refresh"), (False, "reuse")):
        values = [ms for row in rows for ms, refresh in zip(row["cycle_ms"], row["cycle_refresh"]) if refresh == flag]
        result[f"{label}_cycle_mean_ms"] = float(np.mean(values)) if values else None
        result[f"{label}_cycle_p95_nearest_rank_ms"] = sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None
    return result


def read_collection(directory, phase, variant, *, samples=10000):
    match = re.fullmatch(r"seed(\d+)(?:[_-].*)?", directory.name)
    require(match is not None, f"Invalid seed directory {directory.name}")
    seed = int(match.group(1))
    tags = list(ARMS) if phase != "full" else list(ARMS)[:3]
    collection = f"{variant}/{phase}/{directory.name}"
    metas, arms, sources = {}, {}, []
    for tag in tags:
        path = directory / f"{tag}.json"
        data = read_json(path)
        try:
            metas[tag], arms[tag] = validate_arm(data, tag, phase, variant, seed)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        sources.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    base_meta, base_arm = metas[tags[0]], arms[tags[0]]
    base_identity = identity(base_meta)
    for tag in tags[1:]:
        require(identity(metas[tag]) == base_identity, f"Checkpoint/source/runtime/host/latent pairing differs in {tag}")
        require(all(arms[tag][key]["init_state"] == row["init_state"] for key, row in base_arm.items()),
                f"Initial states do not pair in {tag}")
    gate_path = directory / "gate.json"
    gate = read_json(gate_path)
    require(gate.get("schema_version") == 3 and gate.get("pass") is True, "Missing or failed schema-3 identity gate")
    require(all(gate.get("runtime", {}).get(k) == base_meta["runtime"][k] for k in RUNTIME_FIELDS),
            "Identity gate checkpoint/source/runtime differs from rollouts")
    require(gate.get("gpu") == base_meta["provenance"]["gpu"] and gate.get("ckpt") == base_meta["provenance"]["ckpt"],
            "Identity gate GPU/checkpoint differs from rollouts")
    for key in ("identity_err", "repeat_full_err", "skip_identity_err", "whole_identity_err"):
        value = gate.get(key)
        require(type(value) in (float, int) and math.isfinite(value) and 0 <= value < 0.005, f"Gate failed {key}")
    for name, count in (("old_block", 240), ("fresh_block", 240), ("old_stack", 8), ("fresh_stack", 8)):
        check = gate.get("cache_integrity", {}).get(name, {})
        require(check.get("entries") == count and type(check.get("checks")) is int
                and check["checks"] > 0 and check.get("unchanged") is True
                and check.get("finite") is True, f"Gate lacks verified cache integrity: {name}")
    require(gate.get("cells") == 240 and gate.get("whole_identity_counts") == {"hits": 8, "misses": 0},
            "Gate lacks selected-cell and observed whole-stack coverage")
    sources.append({"file": gate_path.name, "sha256": sha256(gate_path), "bytes": gate_path.stat().st_size})
    report = {"collection": collection, "variant": variant, "phase": phase, "seed": seed,
              "valid": True, "complete_500_state_comparison": phase == "full",
              "tasks": PHASES[phase][0], "states_per_task": PHASES[phase][1],
              "n_paired_states": len(base_arm), "identity": base_identity,
              "provenance_by_arm": {tag: meta["provenance"] for tag, meta in metas.items()},
              "sources": sources, "gate": gate, "arms": {}, "paired_contrasts": []}
    keys = sorted(base_arm)
    timed_keys = [key for key in keys if not base_arm[key]["warmup"]]
    for tag in tags:
        rows = [arms[tag][key] for key in keys]
        timed = [arms[tag][key] for key in timed_keys]
        success = [int(row["success"]) for row in rows]
        latency = [row["ms_per_cycle"] for row in timed]
        report["arms"][tag] = {
            "config": ARMS[tag], "label": LABELS[tag], "successes": sum(success), "episodes": len(rows),
            "success_pct": 100 * sum(success) / len(rows),
            "success_ci95_pct": bootstrap_ci(success, label=f"{collection}/{tag}/success", samples=samples, scale=100),
            "latency_episode_mean_ms": float(np.mean(latency)), "latency_episodes": len(timed),
            "latency_ci95_ms": bootstrap_ci(latency, label=f"{collection}/{tag}/latency", samples=samples),
            "all_episodes_cycles": cycle_summary(rows), "nonwarmup_cycles": cycle_summary(timed)}
    for left, right in (("01_whole_tau8", "00_stock_a"), ("02_block_tau8", "00_stock_a"),
                        ("01_whole_tau8", "02_block_tau8")):
        success_delta = [int(arms[left][k]["success"]) - int(arms[right][k]["success"]) for k in keys]
        latency_delta = [arms[left][k]["ms_per_cycle"] - arms[right][k]["ms_per_cycle"] for k in timed_keys]
        report["paired_contrasts"].append({
            "left": left, "right": right, "n_paired_states": len(keys),
            "success_difference_pp": float(np.mean(success_delta) * 100),
            "success_ci95_pp": bootstrap_ci(success_delta, label=f"{collection}/{left}-{right}/success", samples=samples, scale=100),
            "left_only_successes": success_delta.count(1), "right_only_successes": success_delta.count(-1),
            "latency_paired_episodes": len(timed_keys), "latency_difference_ms": float(np.mean(latency_delta)),
            "latency_ci95_ms": bootstrap_ci(latency_delta, label=f"{collection}/{left}-{right}/latency", samples=samples)})
    if phase != "full":
        mismatches = [{"task": key[0], "episode": key[1],
                       "fields": [f for f in REPEAT_FIELDS if arms["00_stock_a"][key][f] != arms["03_stock_b"][key][f]]}
                      for key in keys]
        mismatches = [row for row in mismatches if row["fields"]]
        report["stock_repeat"] = {"pass": not mismatches, "compared_fields": list(REPEAT_FIELDS),
                                  "matched_states": len(keys) - len(mismatches), "mismatches": mismatches,
                                  "scope": "Recorded episode outcomes and lengths; not action-tensor equality."}
        if mismatches:
            report["valid"] = False
            report["error"] = "Fresh-process repeated stock outcomes/lengths disagree; scaling gate failed."
    return report


def validate_source_manifest(path):
    manifest = read_json(path)
    digest = manifest.get("source_bundle_sha256")
    require(is_sha(digest), "Source manifest has no valid source_bundle_sha256")
    archive = Path(path).with_name("source_bundle.tar.gz")
    require(archive.is_file(), f"Source archive missing beside manifest: {archive}")
    require(sha256(archive) == digest, "Source archive hash disagrees with manifest")
    return {"manifest_file": Path(path).name, "manifest_sha256": sha256(path),
            "source_bundle_file": archive.name, "source_bundle_sha256": digest,
            "archive_bytes": archive.stat().st_size, "manifest": manifest}


def analyze(root, *, variant="original", samples=10000, source_manifest=None):
    require(samples >= 100, "Use at least 100 bootstrap samples")
    root = Path(root)
    reports, errors = [], []
    source = validate_source_manifest(source_manifest) if source_manifest else None
    for phase in PHASES:
        for directory in sorted((root / variant / phase).glob("seed*")):
            if not directory.is_dir():
                continue
            try:
                report = read_collection(directory, phase, variant, samples=samples)
                if source:
                    require(report["identity"]["runtime"]["source_bundle_sha256"] == source["source_bundle_sha256"],
                            "Recorded runtime source hash differs from verified source archive")
                reports.append(report)
            except ValueError as error:
                errors.append({"collection": f"{variant}/{phase}/{directory.name}", "error": str(error)})
    for report in reports:
        if report["phase"] != "full":
            continue
        pilots = [p for p in reports if p["phase"] == "pilot" and p["seed"] == report["seed"] and p["valid"]]
        matching = [p for p in pilots if p["identity"] == report["identity"]]
        report["passed_pilot_with_same_identity"] = len(matching) == 1
        if len(matching) != 1:
            report["valid"] = False
            report["error"] = "Full collection lacks exactly one passed pilot with identical checkpoint/source/runtime/host/latent."
    for report in reports:
        if not report["valid"]:
            errors.append({"collection": report["collection"], "error": report["error"]})
    require(reports or errors, f"No {variant}/{{smoke,pilot,full}}/seed* collections under {root}")
    return {"schema_version": 1, "analysis_script_sha256": sha256(__file__),
            "bootstrap": {"method": "paired initial-state percentile bootstrap", "samples": samples,
                          "seed": BOOTSTRAP_SEED, "numpy_version": np.__version__,
                          "confidence": 0.95, "quantile_method": "linear",
                          "scope": "Pointwise; conditional on each fixed benchmark collection and policy seed. No cross-seed uncertainty or published-system equivalence test."},
            "latency_estimand": "Equal-weight mean of episode cycle means; exclude first episode per task/arm. Simulator/reset excluded. Refresh/reuse means are separately cycle-pooled.",
            "source_archive": source,
            "source_scope": "Recorded bundle identity only." if source is None else "Recorded identity matched to independently hashed local archive.",
            "collections": reports, "errors": errors, "valid": not errors}


def text_report(data):
    lines = ["Fresh-process baseline audit; phases and seeds remain separate."]
    for report in data["collections"]:
        scope = "COMPLETE 500-STATE COLLECTION" if report["complete_500_state_comparison"] else "SMOKE / PILOT DIAGNOSTIC"
        lines.append(f"\n{report['collection']}: {scope}; valid={report['valid']}")
        lines.append(f"  {report['tasks']} tasks x {report['states_per_task']} states; {report['n_paired_states']} paired states per arm")
        for tag, arm in report["arms"].items():
            lo, hi = arm["success_ci95_pct"]
            lines.append(f"  {arm['label']:<24} {arm['successes']}/{arm['episodes']} = {arm['success_pct']:.1f}% [{lo:.1f}, {hi:.1f}]; "
                         f"{arm['latency_episode_mean_ms']:.1f} ms ({arm['latency_episodes']} nonwarmup episode means)")
        for comparison in report["paired_contrasts"]:
            lo, hi = comparison["success_ci95_pp"]
            lines.append(f"  {LABELS[comparison['left']]} minus {LABELS[comparison['right']]}: "
                         f"{comparison['success_difference_pp']:+.1f} pp [{lo:+.1f}, {hi:+.1f}], "
                         f"latency {comparison['latency_difference_ms']:+.1f} ms")
        if "stock_repeat" in report:
            repeat = report["stock_repeat"]
            lines.append(f"  Fresh-process stock repeat: pass={repeat['pass']}; {repeat['matched_states']}/{report['n_paired_states']} states match all recorded outcome/length fields")
        gate = report["gate"]
        lines.append(f"  Same-input gate: full repeat={gate['repeat_full_err']:.3g}; block identity={gate['identity_err']:.3g}; "
                     f"skip identity={gate['skip_identity_err']:.3g}; whole identity={gate['whole_identity_err']:.3g}")
        for tag, arm in report["arms"].items():
            counts = arm["nonwarmup_cycles"]
            lines.append(f"  {LABELS[tag]} nonwarmup cycles: {counts['refresh_cycles']} refresh, {counts['reuse_cycles']} reuse")
    for error in data["errors"]:
        lines.append(f"\nINVALID {error['collection']}: {error['error']}")
    lines.extend(["\nIntervals are pointwise and conditional on the fixed benchmark collection and policy seed.",
                  "A matched local baseline does not establish published checkpoint/protocol equivalence.", data["source_scope"]])
    return "\n".join(lines)


def make_figure(data, path):
    complete = [r for r in data["collections"] if r["phase"] == "full" and r["valid"]]
    require(bool(complete), "Figure requires a validated full 500-state collection and matching passed pilot")
    require(not data["errors"], "Resolve invalid collections before creating a figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(complete), figsize=(5.0 * len(complete), 3.8), squeeze=False)
    for ax, report in zip(axes[0], complete):
        for tag, color, marker in zip(list(ARMS)[:3], ("#333333", "#246fa8", "#bd6c20"), ("o", "s", "^")):
            arm = report["arms"][tag]
            x, y = arm["latency_episode_mean_ms"], arm["success_pct"]
            lx, hx = arm["latency_ci95_ms"]
            ly, hy = arm["success_ci95_pct"]
            ax.errorbar(x, y, xerr=[[max(0, x - lx)], [max(0, hx - x)]], yerr=[[max(0, y - ly)], [max(0, hy - y)]],
                        fmt=marker, color=color, markersize=7, capsize=3, label=arm["label"])
        ax.set(title=f"{report['variant']}, seed {report['seed']}\n500 paired states; 490 timed episode means/arm",
               xlabel="Mean prediction-cycle latency (ms)", ylabel="Task success (%)")
        ax.set_ylim(max(0, min(a["success_ci95_pct"][0] for a in report["arms"].values()) - 5), 101)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.012, "Pointwise 95% initial-state bootstrap intervals; one policy seed per panel. No cross-collection timing comparisons.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, metadata={"Creator": "report_baseline_audit.py"})
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--variant", choices=("original", "optional"), default="original")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    outputs = [p for p in (args.json_path, args.figure) if p is not None]
    try:
        require(len({p.resolve() for p in outputs}) == len(outputs), "Output paths must be distinct")
        for path in outputs:
            require(not path.exists(), f"Refusing to overwrite existing output: {path}")
            require(path.parent.is_dir(), f"Output parent directory does not exist: {path.parent}")
        data = analyze(args.root, variant=args.variant, samples=args.samples, source_manifest=args.source_manifest)
        print(text_report(data))
        if args.figure:
            make_figure(data, args.figure)
        if args.json_path:
            args.json_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        return 0 if data["valid"] else 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

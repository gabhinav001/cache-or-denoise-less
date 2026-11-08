"""Report fresh-process tau=8 plain vs attenuated caching results.

Verifies:
  - Complete 10-task x 10-episode (100 state) coverage without duplicates
  - Exact state pairing and matching initial-state hashes between arms
  - Process isolation: single-arm execution per file (singleton meta.configs)
  - Matching checkpoint hash, sigma shift, model class, and runtime settings
  - Paired success rates, discordant counts, and exact two-sided binomial p-values
  - State-clustered within-task bootstrap interval over the 100 shared states
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import sys
from pathlib import Path

EXPECTED_TASKS = 10
EXPECTED_EPISODES = 10
EXPECTED_N = EXPECTED_TASKS * EXPECTED_EPISODES
EXPECTED_CHECKPOINT_SHA256 = "26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4"
BOOTSTRAP_SEED = 20260921


def binom_test(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test without external dependencies."""
    if n == 0:
        return 1.0
    p_k = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total_p = 0.0
    for i in range(n + 1):
        p_i = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if p_i <= p_k + 1e-12:
            total_p += p_i
    return min(1.0, total_p)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_arm_file(path: str, expected_config: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    require("meta" in data, f"{path}: missing meta")
    require("rows" in data, f"{path}: missing rows")
    meta = data["meta"]
    rows = data["rows"]

    # Verify single-arm execution (process isolation)
    configs = meta.get("configs")
    require(isinstance(configs, list) and len(configs) == 1,
            f"{path}: expected singleton configs list, got {configs}")
    require(configs[0] == expected_config,
            f"{path}: expected config {expected_config}, got {configs[0]}")

    # Verify runtime metadata
    runtime = meta.get("runtime", {})
    require(isinstance(runtime, dict), f"{path}: missing runtime metadata")
    ckpt_sha = runtime.get("checkpoint_sha256")
    require(is_sha(ckpt_sha), f"{path}: invalid checkpoint_sha256 {ckpt_sha}")
    require(ckpt_sha == EXPECTED_CHECKPOINT_SHA256,
            f"{path}: unexpected checkpoint_sha256 {ckpt_sha}")
    require(runtime.get("effective_action_sigma_shift") == 1.0,
            f"{path}: expected effective_action_sigma_shift == 1.0")
    require(runtime.get("action_infer_mode") == "first_frame",
            f"{path}: expected action_infer_mode == 'first_frame'")
    require(runtime.get("mixed_precision") == "bf16",
            f"{path}: expected mixed_precision == 'bf16'")

    # Verify complete 10x10 coverage without duplicates
    require(len(rows) == EXPECTED_N,
            f"{path}: expected {EXPECTED_N} rows, got {len(rows)}")
    seen_keys = set()
    row_map = {}
    for r in rows:
        key = (r.get("task"), r.get("episode"))
        require(key[0] is not None and 0 <= key[0] < EXPECTED_TASKS,
                f"{path}: invalid task in row {r}")
        require(key[1] is not None and 0 <= key[1] < EXPECTED_EPISODES,
                f"{path}: invalid episode in row {r}")
        require(key not in seen_keys, f"{path}: duplicate episode {key}")
        seen_keys.add(key)
        init_state = r.get("init_state")
        require(bool(init_state), f"{path}: missing init_state in row {key}")
        row_map[key] = r

    require(len(seen_keys) == EXPECTED_N,
            f"{path}: incomplete coverage ({len(seen_keys)}/{EXPECTED_N})")
    return {"meta": meta, "rows": rows, "row_map": row_map}


def analyze_dir(outdir: str) -> dict:
    plain_files = sorted(glob.glob(os.path.join(outdir, "*_plain.json")))
    atten_files = sorted(glob.glob(os.path.join(outdir, "*_attenuated.json")))

    if not plain_files:
        raise ValueError(f"No *_plain.json files found in {outdir}")

    # Check for orphaned attenuated files
    plain_prefixes = {os.path.basename(f).replace("_plain.json", "") for f in plain_files}
    atten_prefixes = {os.path.basename(f).replace("_attenuated.json", "") for f in atten_files}
    orphans = (plain_prefixes ^ atten_prefixes)
    require(not orphans, f"Mismatched arm files in {outdir}: unpaired seeds {orphans}")

    print(f"tau=8 Fresh-Process Replication Analysis: {outdir}\n")
    print(f"{'Seed':<8} {'Plain':<10} {'Attenuated':<12} {'Diff (pts)':<12} {'Wins/Losses':<14} {'p-value':<10}")
    print("-" * 68)

    seed_results = {}
    state_diffs_by_seed = {}
    init_state_hashes_first = None

    for pf in plain_files:
        base = os.path.basename(pf)
        seed_prefix = base.replace("_plain.json", "")
        af = os.path.join(outdir, f"{seed_prefix}_attenuated.json")

        plain_arm = validate_arm_file(pf, "c3ache_tau8")
        atten_arm = validate_arm_file(af, "c3ache_tau8_shrink")

        # Verify cross-arm pairing within seed
        pmeta = plain_arm["meta"]
        ameta = atten_arm["meta"]
        require(pmeta.get("seed") == ameta.get("seed"),
                f"{seed_prefix}: seed mismatch between plain and attenuated")
        require(pmeta.get("initial_latent") == ameta.get("initial_latent"),
                f"{seed_prefix}: initial_latent mismatch between arms")
        require(pmeta.get("runtime", {}).get("dataset_stats_sha256") == ameta.get("runtime", {}).get("dataset_stats_sha256"),
                f"{seed_prefix}: dataset_stats_sha256 mismatch between arms")
        require(pmeta.get("provenance", {}).get("commit") == ameta.get("provenance", {}).get("commit"),
                f"{seed_prefix}: commit mismatch between arms")
        require(pmeta.get("provenance", {}).get("host_tag") == ameta.get("provenance", {}).get("host_tag"),
                f"{seed_prefix}: host_tag mismatch between arms")

        # State hashes pairing check
        prows = plain_arm["row_map"]
        arows = atten_arm["row_map"]
        curr_hashes = {}
        for key in sorted(prows.keys()):
            p_hash = prows[key]["init_state"]
            a_hash = arows[key]["init_state"]
            require(p_hash == a_hash, f"{seed_prefix} {key}: state hash mismatch between arms!")
            curr_hashes[key] = p_hash

        # Verify that all seeds share the identical 100 state hashes
        if init_state_hashes_first is None:
            init_state_hashes_first = curr_hashes
        else:
            require(curr_hashes == init_state_hashes_first,
                    f"{seed_prefix}: benchmark initial states differ from first evaluated seed!")

        plain_succ = sum(1 for r in prows.values() if r["success"])
        atten_succ = sum(1 for r in arows.values() if r["success"])
        better = sum(1 for k in prows if arows[k]["success"] and not prows[k]["success"])
        worse = sum(1 for k in prows if prows[k]["success"] and not arows[k]["success"])

        diff_pts = (atten_succ - plain_succ) / EXPECTED_N * 100.0
        discordant = better + worse
        pval = binom_test(min(better, worse), discordant)

        state_diffs_by_seed[seed_prefix] = {
            k: (1 if arows[k]["success"] else 0) - (1 if prows[k]["success"] else 0)
            for k in prows
        }

        seed_results[seed_prefix] = {
            "plain_succ": plain_succ,
            "atten_succ": atten_succ,
            "diff_pts": diff_pts,
            "atten_wins": better,
            "plain_wins": worse,
            "discordant": discordant,
            "pval": pval,
        }

        print(f"{seed_prefix:<8} {plain_succ:5.1f}% ({plain_succ}/{EXPECTED_N})  {atten_succ:5.1f}% ({atten_succ}/{EXPECTED_N})  {diff_pts:+6.1f}       {better}/{worse:<10}  {pval:.4f}")

    # Cross-seed combined analysis
    summary = {"seeds": seed_results}
    if len(seed_results) > 1:
        seeds = list(seed_results.keys())
        mean_diff = sum(res["diff_pts"] for res in seed_results.values()) / len(seed_results)
        
        # State-level average difference across seeds for each of the 100 shared states
        all_keys = sorted(init_state_hashes_first.keys())
        by_task = {t: [] for t in range(EXPECTED_TASKS)}
        for k in all_keys:
            avg_state_d = sum(state_diffs_by_seed[s][k] for s in seeds) / len(seeds)
            by_task[k[0]].append(avg_state_d)

        # Within-task stratified bootstrap over the 100 state clusters
        rng = random.Random(BOOTSTRAP_SEED)
        boot_means = []
        for _ in range(100000):
            resamp = []
            for t in range(EXPECTED_TASKS):
                td = by_task[t]
                n_t = len(td)
                resamp.extend([td[rng.randrange(n_t)] for _ in range(n_t)])
            boot_means.append(sum(resamp) / len(resamp) * 100.0)

        boot_means.sort()
        ci_low = boot_means[int(0.025 * len(boot_means))]
        ci_high = boot_means[int(0.975 * len(boot_means))]

        summary["mean_diff_pts"] = mean_diff
        summary["state_clustered_bootstrap_95ci"] = [round(ci_low, 1), round(ci_high, 1)]

        print("-" * 68)
        print(f"Mean difference: {mean_diff:+.1f} points across {len(seed_results)} seeds on {EXPECTED_N} shared states")
        print(f"State-clustered within-task bootstrap 95% CI: [{ci_low:.1f}, {ci_high:.1f}] points")
        print(f"(Note: Seeds reuse the same {EXPECTED_N} benchmark states; unadjusted discordant pooling across seeds would ignore state reuse)")
    print()
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) < 1:
        print("Usage: report_tau8_fresh.py <outdir>")
        return 1
    try:
        analyze_dir(argv[0])
        return 0
    except Exception as e:
        print(f"Verification error in report_tau8_fresh: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

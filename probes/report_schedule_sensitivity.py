"""CPU-only sensitivity analysis of the historical four-schedule collections.

No new rollouts. Removes the pilot's actual state hashes, reports every task,
and drops each task in turn. The bootstrap resamples paired states WITHIN task
and weights tasks equally. This is a post-hoc sensitivity check, not independent
confirmation or uncertainty over new tasks. Never pool the two collections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONFIGS = {2: ["stock"], 4: ["stock", "c3ache_tau4_shrink_skip"],
           10: ["c3ache_tau4_shrink_skip"]}
PAIRS = [("2st", "10st+cache"), ("4st", "4st+cache")]
IDENTITY = ("commit", "host_tag", "gpu", "ckpt", "ckpt_sha256_16", "torch", "mujoco")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path, seed, steps, tasks, episodes, configs):
    data = json.loads(path.read_text())
    meta = data["meta"]
    expected = dict(seed=seed, num_inference_steps=steps, suite="libero_goal",
                    n_tasks=tasks, n_episodes=episodes, replan=10, configs=configs,
                    timing=1, warmup_episodes=1)
    require(all(meta.get(k) == v for k, v in expected.items()), f"Protocol mismatch: {path.name}")
    require(len(data["rows"]) == tasks * episodes * len(configs), f"Unexpected row count: {path.name}")
    arms = {}
    for config in configs:
        rows = [r for r in data["rows"] if r["config"] == config]
        keyed = {(r["task"], r["episode"]): r for r in rows}
        require(len(keyed) == len(rows) == tasks * episodes, f"Duplicate/missing states: {path.name}")
        require(set(keyed) == {(t, e) for t in range(tasks) for e in range(episodes)},
                f"Incomplete task grid: {path.name}")
        hashes = [r.get("init_state") for r in rows]
        require(all(isinstance(h, str) and h for h in hashes) and len(set(hashes)) == len(rows),
                f"Missing/repeated state hashes: {path.name}")
        for (task, ep), row in keyed.items():
            require(type(row["success"]) is bool, f"Non-boolean outcome: {path.name}")
            require(row.get("warmup") is (ep == 0), f"Incorrect warmup: {path.name}")
            require(row.get("suite") == "libero_goal", f"Wrong row suite: {path.name}")
            ms = row.get("ms_per_cycle")
            require(type(ms) in (float, int) and math.isfinite(ms) and ms > 0,
                    f"Invalid timing: {path.name}")
        arm = f"{steps}st" + ("" if config == "stock" else "+cache")
        if config != "stock" and len([c for c in configs if c != "stock"]) > 1:
            arm += "/" + config
        arms[arm] = keyed
    return meta, arms


def bootstrap(groups, label, samples):
    # Label-local RNG: adding/reordering another comparison cannot change a CI.
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    means = np.zeros(samples)
    for values in groups:
        values = np.asarray(values, dtype=float)
        require(len(values) > 0, "Empty task stratum")
        means += values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return (100 * np.quantile(means / len(groups), [0.025, 0.975], method="linear")).tolist()


def contrast(arms, left, right, keys, label, samples):
    tasks = sorted({t for t, _ in keys})
    groups, per_task = [], []
    for task in tasks:
        selected = [k for k in keys if k[0] == task]
        delta = [int(arms[left][k]["success"]) - int(arms[right][k]["success"]) for k in selected]
        groups.append(delta)
        per_task.append({"task": task, "states": len(selected),
                         "difference_pp": 100 * float(np.mean(delta)),
                         "better": delta.count(1), "worse": delta.count(-1)})
    delta = float(np.mean([p["difference_pp"] for p in per_task]))
    leave_one = [float(np.mean([p["difference_pp"] for p in per_task if p["task"] != t])) for t in tasks]
    return {"left": left, "right": right, "states": len(keys),
            "difference_pp_equal_task": delta, "ci95_pp": bootstrap(groups, label, samples),
            "per_task": per_task, "leave_one_task_out_range_pp": [min(leave_one), max(leave_one)],
            "tasks_positive_zero_negative": [sum(p["difference_pp"] > 0 for p in per_task),
                                              sum(p["difference_pp"] == 0 for p in per_task),
                                              sum(p["difference_pp"] < 0 for p in per_task)]}


def analyze(root, samples=20000):
    require(samples >= 100, "At least 100 resamples required")
    root = Path(root)
    sources, pilot_hashes, pilot_pairing = [], None, None
    # Validate the exclusion set against ALL eleven arms at all three pilot seeds.
    for seed in (42, 7, 11):
        for steps in (2, 3, 4, 6, 8, 10):
            suffix = "" if seed == 42 else f"_seed{seed}"
            path = root / "deployfrontier" / f"d{steps}{suffix}.json"
            configs = ["stock"]
            if steps in (4, 6):
                configs += ["c3ache_tau4_shrink_skip"]
            if steps == 10:
                configs += [f"c3ache_tau{tau}_shrink_skip" for tau in (2, 4, 8)]
            _, pilot = read_run(path, seed, steps, 3, 10, configs)
            states = {key: row["init_state"] for key, row in pilot[f"{steps}st"].items()}
            for rows in pilot.values():
                require({k: r["init_state"] for k, r in rows.items()} == states, "Pilot pairing differs")
            if pilot_pairing is None:
                pilot_pairing, pilot_hashes = states, set(states.values())
            require(states == pilot_pairing, "Pilot exclusion sets differ")
            sources.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
    collections, shared = [], None
    for seed in (42, 7):
        arms, base_identity = {}, None
        for steps, configs in CONFIGS.items():
            suffix = "" if seed == 42 else f"_seed{seed}"
            path = root / "statescale" / f"s{steps}{suffix}.json"
            meta, current = read_run(path, seed, steps, 10, 50, configs)
            identity = {k: meta["provenance"].get(k) for k in IDENTITY}
            identity["initial_latent"] = meta.get("initial_latent")
            require(all(identity.values()), f"Missing collection identity: {path.name}")
            if base_identity is None:
                base_identity = identity
            require(identity == base_identity, f"Unmatched host/code/checkpoint/latent: {path.name}")
            arms.update(current)
            sources.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
        for arm, rows in arms.items():
            states = {k: r["init_state"] for k, r in rows.items()}
            if shared is None:
                shared = states
            require(states == shared, f"Unmatched state identities: seed {seed}/{arm}")
        require(all(shared.get(k) == h for k, h in pilot_pairing.items()), "Pilot absent from large collection")
        retained = sorted(k for k, h in shared.items() if h not in pilot_hashes)
        require(len(retained) == 470, "Expected exactly 470 states after pilot exclusion")
        collection = {"seed": seed, "identity": base_identity, "arms": {}, "contrasts": {}}
        for arm, rows in arms.items():
            timed = [r["ms_per_cycle"] for r in rows.values() if not r["warmup"]]
            collection["arms"][arm] = {"successes": sum(r["success"] for r in rows.values()),
                                        "states": len(rows), "timed_episodes": len(timed),
                                        "episode_mean_ms": float(np.mean(timed))}
        for scope, keys in (("all_500", sorted(shared)), ("excluding_pilot_470", retained)):
            collection["contrasts"][scope] = [contrast(arms, a, b, keys, f"schedule-v1/{seed}/{scope}/{a}-{b}", samples)
                                                 for a, b in PAIRS]
        collections.append(collection)
    return {"schema_version": 1, "analysis_sha256": digest(__file__), "sources": sources,
            "method": {"resamples": samples, "rng": "PCG64; SHA256(label) first 8 bytes, big-endian",
                       "numpy": np.__version__, "ci": "pointwise 95% paired within-task percentile bootstrap; linear quantiles",
                       "estimand": "equal-weight mean of the ten task success differences",
                       "scope": "Post-hoc sensitivity analysis of historical data. Not a new experiment, prespecified holdout, task-generalization test, or published-cache reproduction.",
                       "exclusion": "30 state hashes shared by every pilot arm/seed; remaining 470 states weighted equally within each task"},
            "collections": collections}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()
    data = analyze(args.root, args.samples)
    if args.output:
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    for c in data["collections"]:
        for scope, contrasts in c["contrasts"].items():
            for p in contrasts:
                print(f"seed {c['seed']}, {scope}, {p['left']} - {p['right']}: "
                      f"{p['difference_pp_equal_task']:+.2f} pp; CI {p['ci95_pp']}; "
                      f"task signs {p['tasks_positive_zero_negative']}; leave-task-out {p['leave_one_task_out_range_pp']}")
    print(data["method"]["scope"])


if __name__ == "__main__":
    main()

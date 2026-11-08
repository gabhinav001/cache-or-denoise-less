# Cache or Denoise Less? Residual Reuse in a Flow-Matching World Action Model

[![Paper](https://img.shields.io/badge/Paper-CoRL%202026%20EEAI-blue)](paper.pdf)
[![Code](https://img.shields.io/badge/Repository-cache--or--denoise--less-green)](https://github.com/gabhinav001/cache-or-denoise-less)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the evaluation harness, raw experimental artifacts, and analysis scripts for the paper:
> **Cache or Denoise Less? Residual Reuse in a Flow-Matching World Action Model**  
> *CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI (EEAI)*  
> **Authors:** Abhinav Gupta ([@gabhinav001](https://github.com/gabhinav001)) and Nissan Pow ([@npow](https://github.com/npow/))  
> **Repository:** [https://github.com/gabhinav001/cache-or-denoise-less](https://github.com/gabhinav001/cache-or-denoise-less)

---

## Executive Summary & TL;DR

Cross-cycle residual caching (such as the $C^3\text{ache}$ recipe) speeds up flow-matching robot policies by computing early denoising steps in a periodic "refresh" cycle and reusing those transformer residuals on subsequent "reuse" cycles. This work asks: **Does residual caching actually beat simply taking fewer denoising steps?**

Key findings evaluated on LIBERO with **Fast-WAM** (a 6B-parameter world action model):
1. **2 Uncached Steps Beat Caching:** On 500 initial states, 2 uncached denoising steps achieve equivalent success to the full 10-step baseline (within 2 points: 96.6% vs 96.4%) while cutting per-call latency from **202 ms to 92 ms** on an RTX A6000.
2. **Action Caches Cannot Beat 2 Steps:** A cache keeping 2 fresh steps still has to encode camera images, run the 5B video transformer, and denoise the remaining steps (~70% of total call time). Reusing 8 steps takes ~100 ms (plus refresh overhead), making it strictly slower than running 2 uncached steps.
3. **Time Shifts Move the Cache Boundary:** Fast-WAM's newer sampler shift ($s=1$) crowds steps differently than its legacy shift ($s=5$). Caching 8 of 10 steps covers **80%** of the flow path at $s=1$ compared to **44%** at $s=5$, causing policy success to collapse from 95% down to 63%.
4. **Noise Latent Drift Destroys Reused Residuals:** Deployed robot policies resample fresh noise latents between calls. Changing the latent across cycles causes stored residuals to drift by 85–97%, resulting in a 14–40 point success drop for caches covering $\ge 60\%$ of the flow path.

---

## Quickstart Guide

### 1. Fast CPU Verification (< 1 minute)
You do not need a GPU or model weights to verify the paper's claims, rerun the statistical tests, or reproduce all figures and tables. Everything rebuilds from raw shipped run logs:

```bash
# Set up a clean environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r repro/requirements-cpu.txt   # numpy, matplotlib, CPU torch

# Run the complete test suite & verification pipeline
PY=python3 bash repro/verify_workshop.sh

# Rebuild analysis snapshot from raw run logs
python3 probes/report_budget_shift.py

# Verify every printed number in the paper against the data
python3 probes/paper_numbers.py

# Re-generate paper figures and LaTeX tables locally
python3 probes/make_figure.py
python3 probes/make_tables.py
```

### 2. Minimal Attenuation Demo (NumPy only)
To inspect the least-squares residual attenuation and staleness weighting mechanism in a minimal 15-line demo:
```bash
python3 examples/online_beta_minimal.py
```

### 3. GPU Rollouts & Closed-Loop Reproduction
To re-run closed-loop policy rollouts on an NVIDIA GPU host (requires $\ge 48$ GB VRAM, CUDA 12.8, MuJoCo 3.3.2):

```bash
# Provision environment, LIBERO benchmarks, and download Fast-WAM checkpoints
bash setup/onstart_libero_shrink.sh

# Quick smoke test (~1.7 hrs on RTX A6000; 11 arms on 1 state per task)
bash repro/regenerate_paper.sh smoke

# Run specific evaluation tiers
bash repro/regenerate_paper.sh frontier     # 100-state frontier comparison
bash repro/regenerate_paper.sh share        # Cached flow-path share & shift sweep
bash repro/regenerate_paper.sh latent_delay # Per-episode latent & asynchronous delay

# Dry-run commands without launching GPU jobs
DRY_RUN=1 bash repro/regenerate_paper.sh paper
```

---

## Key Experimental Results

### Success vs. Replanning Latency (100 LIBERO-Goal States)
Evaluated on an RTX A6000 under PyTorch default compile mode:

| Policy Configuration | Denoising Steps ($K$) | Mean Call Latency | Success Rate (Optional-IDM, $s=1$) | Success Rate (Original, $s=5$) |
|---|:---:|:---:|:---:|:---:|
| **Uncached (10-step)** | 10 | 202 ms | 95.0% | 95.0% |
| **Uncached (2-step)** | 2 | **92–94 ms** | **95.0%** | **96.0%** |
| **Uncached (1-step)** | 1 | 81 ms | 88.0% | 93.0% |
| **$C^3\text{ache}$ ($\tau=8$, 8 cached)** | 2 fresh / 8 cached | 100 ms (+ refreshes) | 63.0% (collapsed) | 94.0% |
| **Composed Cache ($\tau=4$, 1 cached)**| 1 fresh / 1 cached | 2–10% below 2 steps (not resolved) | 94.0% | 95.0% |

*Takeaway:* 2 uncached steps cut latency by >50% with zero loss in task success, whereas caching 8 steps severely degrades policy performance under the current $s=1$ time schedule.

---

## Repository Architecture

```
├── README.md                 # Project overview and reproduction guide
├── paper.pdf                 # Camera-ready paper; the supplementary results are its Appendix A
├── probes/                   # Core evaluation harness and analysis tools
│   ├── fw_libero.py          # Fast-WAM rollout loop & policy interface in LIBERO
│   ├── gate_cache.py         # Bitwise cache identity gate run before collection
│   ├── make_figure.py        # Rebuilds Figures 1, 2, and A1
│   ├── make_tables.py        # Rebuilds LaTeX Tables 1 and A1-A8
│   ├── paper_numbers.py      # Map verifying every printed number against raw logs
│   ├── report_budget_shift.py# Compiles raw run files into results/budget_shift/summary.json
│   └── test_*.py             # Unit tests for cache semantics, math, and intervals
├── repro/                    # Reproduction pipeline & experiment manifests
│   ├── verify_workshop.sh    # One-click CPU verification script
│   ├── regenerate_paper.sh   # Master GPU driver for rerunning experiment tiers
│   ├── regeneration_plan.txt # Complete manifest of experiment tiers & GPU hour budgets
│   ├── PAPER_NUMBERS.md      # Audit log mapping every paper statistic to raw source runs
│   ├── MANIFEST.md           # Integrity checksums and metadata for all raw run files
│   └── experiments/*.sh      # Specific runner scripts for each study section
├── results/                  # Raw execution traces and metric snapshots
│   ├── budget_shift/         # Primary frontier, shift, noise, and latency logs
│   │   └── summary.json      # Canonical JSON analysis snapshot
│   ├── composed2step/        # Composed 2-step caching evaluations
│   └── statescale/           # 500-state large-sample validation runs
├── setup/                    # Environment provisioning recipes
│   ├── HOST_REQUIREMENTS.md  # Hardware & OS driver specifications
│   └── onstart_libero_shrink.sh # Container bootstrap script
└── examples/
    └── online_beta_minimal.py# Minimal NumPy implementation of online residual attenuation
```

---

## Checkpoint & Experiment Arm Conventions

### Checkpoints Used
| Name in Repo | Checkpoint File | Native Time Shift | Description |
|---|---|---|---|
| `uncond` ("original") | `libero_uncond_2cam224.pt` | $s = 5$ | Original Fast-WAM release (trained with shift 5) |
| `idm` ("Optional-IDM") | `libero_optional_idm_2cam224.pt` | $s = 1$ | Current upstream default (runs in first-frame mode) |

### Experiment Arm Syntax
Evaluation runs are launched via parameterized arm strings:
```
name:checkpoint:steps:shift:episodes_per_task:config[@options]
```
*Example:* `idm_s1_p8_tau8:idm:10:1.0:10:whole:8@prefix=8` runs Optional-IDM at shift $s=1.0$ with 10 episodes per task, caching the first 8 steps across a refresh period of $\tau=8$.

#### Configuration Options (`config`)
- `stock`: Standard uncached denoising schedule.
- `whole:TAU`: Whole-stack residual cache refreshed every `TAU` calls (`0` = fill once per episode).
- `whole:TAU:a`: Whole-stack cache with least-squares residual attenuation pre-aggregated at refresh.
- `c3ache:TAU[:sx]`: Per-block hooks (original $C^3\text{ache}$ implementation).

#### Modifiers (`@options`)
- `prefix=N`: Cache steps $0 \dots N-1$ (defaults to all but the last 2 steps).
- `noise`: Resample fresh initial Gaussian noise on each replanning call (deployed condition).
- `cudagraphs`: Use upstream PyTorch `reduce-overhead` CUDA graphs (uncached only).
- `perturb=cam:0.015`: Apply 1.5 cm camera jitter to evaluate robustness to sensory noise.

---

## Practitioner Guide: Traps to Avoid

When reproducing these experiments or developing custom inference caches for diffusion / flow-matching policies:

1. **Avoid Shared-Process Benchmarking:**
   *Running multiple configurations sequentially within the same Python process alters PyTorch's kernel compilation and CUDA caching state.* This caused 17–19 out of 20 evaluation trajectories to silently diverge (`results/rngdrift/`). Always run each configuration in a fresh, isolated Python process.
2. **CUDA Graphs Silently Corrupt Residual Caches:**
   PyTorch's `reduce-overhead` mode reuses fixed memory buffers across steps. In residual caching, this silently overwrites stored tensors with downstream activations. Run `probes/gate_cache.py` to confirm bitwise identity before collecting cached rollouts.
3. **Beware Fixed-Seed Latents:**
   Fast-WAM upstream reseeds noise generation identically on every replanning call. This artificial stability masks residual drift. Always test caches with fresh per-call noise (`@noise`) to reflect realistic deployment.
4. **Pin MuJoCo to 3.3.2:**
   Running on MuJoCo 2.3.7 instead of 3.3.2 results in a silent ~22-point drop in policy success rates due to contact solver differences. See [setup/HOST_REQUIREMENTS.md](setup/HOST_REQUIREMENTS.md) for pinned versions.

---

## Provenance of Shipped Results

Every result reported in the manuscript is traceable to raw data logs shipped in `results/`:

| Paper Section & Result | Raw Run Directory | Host Environment |
|---|---|---|
| **Table 1 & Fig. 1:** 100-State Frontier Sweep | `results/budget_shift/frontier100_a6000/` | Dedicated RTX A6000 |
| **Table 1:** CUDA Graph & Execution Latencies | `results/budget_shift/latency_a6000/` | Dedicated RTX A6000 |
| **Sec. 3:** 500-State Equivalence Validation | `results/statescale/` | RTX A6000 |
| **Sec. 3:** Composed Cache vs. 2-Step Baseline | `results/composed2step/`, `results/matched_composed/` | Quiet & Shared A6000 |
| **Fig. 2 & Sec. 4:** Shift ($s=1,2,3,5$) & Flow Share | `results/budget_shift/shift_a6000/`, `heldout_a6000/` | RTX A6000 |
| **Sec. 4:** Fresh Noise & Latent Control | `results/budget_shift/noise_a6000/`, `latent_delay_a6000/`| RTX A6000 |
| **Sec. 3:** Emulated Asynchronous Delays | `results/budget_shift/latent_delay_a6000/`, `delay_graphs_a6000/` | RTX A6000 |

Detailed hash manifests and audit mappings for every individual file are documented in [repro/AUDIT_MAP.md](repro/AUDIT_MAP.md) and [repro/MANIFEST.md](repro/MANIFEST.md).

---

## Citation

If you use this codebase, harness, or empirical findings in your research, please cite:

```bibtex
@inproceedings{gupta2026cache,
  title={Cache or Denoise Less? Residual Reuse in a Flow-Matching World Action Model},
  author={Gupta, Abhinav and Pow, Nissan},
  booktitle={CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI (EEAI)},
  year={2026}
}
```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

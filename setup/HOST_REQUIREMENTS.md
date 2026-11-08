# What the reproduction needs from a machine

Obtaining a machine is out of scope for this repository — rent it, borrow it, or
use your own. These are the properties it must have, and every one of them was
learned by having a run fail without it.

## Hardware

| requirement | why |
| --- | --- |
| **≥ 40 GB VRAM** | the model is 6 B parameters (5.00 B video expert + 1.02 B action expert). A 24 GB card runs out of memory during Gram accumulation. |
| **≥ 120 GB disk** | ~12 GB checkpoint, ~14 GB Wan common files, plus the image and an inductor cache. A working host settles at ~29 GB used, but hosts advertise disk they do not all deliver. |
| **driver >= 570** | the image is pinned to torch 2.7.1+cu128, and CUDA 12.8 needs a 570-series driver. A host running 555 advertised `CUDA Version: 12.8` in `nvidia-smi`, reported the right GPU and the right VRAM, and then failed every torch CUDA call with error 803. Filter offers on `cuda_max_good>=12.8`; the setup script now proves the GPU works before downloading anything. |
| **a fast link** | setup pulls ~26 GB. At 20 MB/s that is ~20 minutes; at 2 MB/s it is over three hours. |

## Which setup script for which experiment

There is no single script that prepares every experiment, and running the wrong
one leaves a host that passes a casual glance and fails once an experiment starts
loading a model. Pick by what you intend to run:

| experiments | setup script | provides |
| --- | --- | --- |
| every run behind the workshop paper (`repro/regenerate_paper.sh`): `budget_shift` `followup` `composed2step` `matched_composed` `matched50` `matched100` `statescale` `matrix` `rngdrift` `tau8_fresh` | `onstart_libero_shrink.sh` (fetches both checkpoints by default) | LIBERO + mujoco 3.3.2 + EGL, the pinned Fast-WAM source, `libero_optional_idm_2cam224.pt` and `libero_uncond_2cam224.pt` |
| `matched_frontier` `attenuation` `frontier` `staleness` `ceiling_beta` `curve` `generalize` `mujoco` `latency` `baselines` `tau8` `tau8gran` `published` `armorder` `refimpl` | `onstart_libero_shrink.sh` | LIBERO + mujoco 3.3.2 + EGL, and `libero_optional_idm_2cam224.pt` |
| `mechanism` `fidelity` `placement` `ceiling` `stackdiff` `stackbeta` | `onstart_fw_families.sh` | `libero_uncond_2cam224.pt`, `libero_optional_idm_2cam224.pt` (needed by `mechanism --seeds idm/idm7`, which replay the deployment checkpoint) and the recorded episodes under `/workspace/libero` |
| `pi05` `pi05_confirm` | `onstart_pi05_libero.sh` | LeRobot and the π₀.₅ checkpoint |
| `baseline_audit` | `BASELINE_VARIANT=original bash setup/onstart_libero_shrink.sh` (or `BASELINE_VARIANT=optional`) | original/optional checkpoint, paired statistics, pinned FastWAM/LIBERO environment |
| `weights` | none — no GPU | reads the committed Gram matrices |

For `baseline_audit`, provenance tracking records the exact source code hash. You can optionally archive and export:
```bash
tar -czf source_bundle.tar.gz probes repro
export WAM_SOURCE_BUNDLE_SHA256=$(sha256sum source_bundle.tar.gz | cut -d' ' -f1)
```
If unset, `baseline_audit` automatically hashes `inputs/source_bundle.tar.gz` if present, or computes a dynamic SHA256 digest over executed probe sources.

Each experiment declares the checkpoint it loads, and `preflight` verifies that
file specifically rather than assuming one. The four experiments on the
unconditional checkpoint also read recorded video; if yours is not at
`/workspace/libero`, set `VIDEO_ROOT` and preflight will check that instead.

Run `weights` first whatever else you intend. It needs no GPU and no checkpoint,
takes seconds, and refits the substitution coefficients from the recorded Gram
matrices — the check that distinguishes a closed-form least-squares fit from a
sweep, which is the claim most worth doubting.

## Software

`setup/onstart_libero_shrink.sh` installs all of this and is safe to run directly
on a fresh Ubuntu/CUDA container. A completed instance can be stopped and restarted:
the setup script checks its retained environment and exits without reinstalling it.
An interrupted partial setup may still need repair; retain the setup log.
The initial checkpoint downloader is pinned to a version with HTTP resume support.
Three further details matter:

- **mujoco 3.3.2, installed last.** The LIBERO data was rendered with 3.3.2;
  2.3.7 costs about **22 points** of full-compute success on the same machine
  with the same seed. Several dependencies will silently downgrade it, so it is
  installed after everything else.
- **An EGL vendor ICD at `/usr/share/glvnd/egl_vendor.d/`.** libglvnd reads only
  that directory. Without an ICD there, mujoco's EGL backend fails at import with
  `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`, or later
  with `EGL_NOT_INITIALIZED`. Some container runtimes ship one; many do not.
- **`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`.** FastWAM resolves the Wan2.2 common
  files through DiffSynth, which defaults to modelscope — measured at 304 kB/s
  from a western datacentre, about twelve hours for the 13 GB pair. The identical
  files come from HuggingFace at ~50 MB/s.

## Checking a candidate machine

```bash
./repro/run_experiment.sh preflight
```

Verifies CUDA can actually allocate (a host whose driver cannot initialise this
CUDA build still reports a GPU from `device_count()`), that EGL initialises, that
mujoco is 3.3.2, that LIBERO/robosuite/fastwam import, and that the checkpoint is
present. It exits non-zero and names the problem rather than failing later inside
an experiment.

## Then

```bash
bash repro/regenerate_paper.sh smoke        # the paper's configurations, checked state by state (~1.7 h)
bash repro/regenerate_paper.sh paper        # every run behind a printed number (~54 GPU-hours)
```

The workshop paper's outcomes were reproduced across thirteen machines
(7 of 11,725 paired outcomes changed); use an A6000 to compare outcomes
directly. Tier `idle` times the composed cache and must run with nothing else
on the machine. Or run a single experiment:

```bash
./repro/run_experiment.sh frontier --seeds "42 7 11"
```

Copy the JSON artefacts back and analyse them anywhere, with no GPU:

```bash
python3 probes/report_variance.py closedloop <dir>
```

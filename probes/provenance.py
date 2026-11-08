"""What produced this artefact.

A result file recorded its configuration but nothing about the code or the
machine that wrote it, and that omission cost real work. Two seeds in a
directory of "re-runs under the fixed harness" turned out to be the pre-fix
computation: the driver ships the repo to a rented box at launch, so a machine
already running when a fix lands keeps producing old data, and a commit
afterwards does not reach it. Nothing in the artefacts said so. Establishing it
took comparing step counts episode by episode against an archived copy, and the
wrong conclusion had already been published.

The commit answers "which code", the GPU answers "which machine" -- same-seed
runs reproduce exactly on identical hardware and differ by a few points across
different hardware, so a repeat that disagrees is only interpretable if you know
which of the two it was. mujoco's version is here because a point release of it
moved closed-loop success by more than any effect this study reports.

Every field degrades to None rather than raising: this is metadata, and no
artefact should be lost because a box has no git or no nvidia-smi.
"""
from __future__ import annotations

import hashlib, os, socket, subprocess, sys, time


def _run(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _gpu():
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if name: return name.splitlines()[0].strip()
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _host_tag():
    """A stable, non-identifying handle for "which machine".

    The hostname itself must never reach an artefact: these files ship with a
    double-blind submission, and a personal machine is commonly named after its
    owner. Eight hex characters are enough to tell two rented boxes apart, which
    is the only question the field exists to answer.
    """
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def file_digest(path, nbytes=None):
    """SHA-256 of a file, first 16 hex characters, or None.

    Answers "which checkpoint" the way the commit answers "which code". A run
    against a different checkpoint of the same name is a different experiment
    and nothing in an artefact used to say so. `nbytes` hashes only a prefix,
    for files large enough that a full read would cost real time -- a 12 GB
    checkpoint is read at 64 MB, which is ample to separate two of them.
    """
    if not path or not os.path.exists(path): return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            if nbytes:
                h.update(fh.read(nbytes))
            else:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def tensor_digest(t):
    """SHA-256 of a tensor's bytes, first 16 characters, or None.

    For recording the initial latent and each episode's initial state.
    Without them, two runs that disagree cannot be told apart from two runs
    that were never given the same starting point.
    """
    try:
        import numpy as _np
        a = t.detach().cpu().numpy() if hasattr(t, "detach") else _np.asarray(t)
        return hashlib.sha256(_np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    except Exception:
        return None


def provenance(ckpt=None):
    """Code, machine and library versions, for a result file's meta block."""
    here = os.path.dirname(os.path.abspath(__file__))
    # WAM_COMMIT first: a probe runs from a COPY of itself, installed next to
    # FastWAM's configs, and the driver rsyncs the repository with --exclude
    # .git/ -- so there is no checkout on the box to read and `git rev-parse`
    # returned None for every artefact collected remotely. The five tau=8 seeds
    # were collected that way and carry commit: None. The driver now passes the
    # commit it shipped, which is the only place that knows it.
    commit = os.environ.get("WAM_COMMIT") or _run(["git", "rev-parse", "HEAD"], cwd=here)
    if not commit:
        for mpath in ("/repo/inputs/source_manifest.json",
                      "/workspace/source_manifest.json",
                      os.path.join(here, "../inputs/source_manifest.json")):
            if os.path.isfile(mpath):
                try:
                    with open(mpath) as mf:
                        commit = json.load(mf).get("release_commit")
                    if commit:
                        break
                except Exception:
                    pass
    # A probe is copied into the FastWAM tree to run (hydra resolves its config
    # path relative to the file), and that copy is outside any checkout, so the
    # commit has to come from wherever the probe was installed from.
    dirty = _run(["git", "status", "--porcelain"], cwd=here)
    return {"commit": commit,
            "dirty": bool(dirty) if dirty is not None else None,
            # HOST LOAD AT COLLECTION. A rented box is shared, and this one ran
            # at a load average of 63 on 36 CPUs for part of an afternoon and 1.8
            # a few hours later. The full-compute arm read 5722 ms/cycle on one
            # task and 360 on another -- a 16x spread on identical work. Any
            # timing number is a property of the load it was taken under, so the
            # load is recorded beside it and a reviewer can discard what was
            # collected on a busy machine instead of averaging it in.
            "loadavg_1_5_15": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "cpu_count": os.cpu_count(),
            "ckpt": os.path.basename(ckpt) if ckpt else None,
            "ckpt_sha256_16": file_digest(ckpt, nbytes=64 << 20) if ckpt else None,
            "gpu": _gpu(),
            "host_tag": _host_tag(),
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "mujoco": _version("mujoco"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

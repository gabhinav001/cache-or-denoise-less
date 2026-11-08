#!/bin/bash
exec > >(tee -a /workspace/setup.log) 2>&1
set -x
nvidia-smi
mkdir -p /workspace && cd /workspace
# HF's Xet/CAS backend intermittently 401s on anonymous downloads; the classic
# route is slower but does not fail closed mid-setup
export HF_HUB_DISABLE_XET=1
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/workspace/hf
pip install -q -U huggingface_hub
# checkpoint BEFORE pip install -e . (FastWAM pins hub 0.29.2, removing the `hf` CLI)
time hf download yuanty/fastwam libero_uncond_2cam224.pt --local-dir /workspace/ckpt
[ -s /workspace/ckpt/libero_uncond_2cam224.pt ] || { touch /workspace/SETUP_FAILED; exit 1; }
python -c "
import urllib.request,tarfile,io,os,shutil
SHA='7faa71108368fbb3b6885649f112af607427a2d4'
d=urllib.request.urlopen(f'https://codeload.github.com/yuantianyuan01/FastWAM/tar.gz/{SHA}').read()
tarfile.open(fileobj=io.BytesIO(d)).extractall('/src')
n=[x for x in os.listdir('/src') if x.startswith('FastWAM')][0]
shutil.move(os.path.join('/src',n),'/w'); print('FastWAM at /w')
"
cd /w && sed -i "/deepspeed/d" pyproject.toml
pip install -q -e . --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86 imageio imageio-ffmpeg
# videos for both suites, via the python API (the `hf` CLI is gone after the install above)
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
for suite in ["libero_goal_no_noops_lerobot","libero_spatial_no_noops_lerobot"]:
    D=f"lerobot_v30/{suite}/videos"; out=f"/workspace/libero/{suite}"
    os.makedirs(out, exist_ok=True)
    for src,dst in [("observation.images.image","main"),("observation.images.wrist_image","wrist")]:
        f=f"{out}/{dst}.mp4"
        if not os.path.exists(f) or os.path.getsize(f)<1e6:
            shutil.copy(hf_hub_download("yuanty/LIBERO-fastwam", f"{D}/{src}/chunk-000/file-000.mp4",
                                        repo_type="dataset"), f)
        print(suite[7:14], dst, round(os.path.getsize(f)/1e6,1),"MB", flush=True)
PY
python -c "import cv2, imageio.v3, fastwam; print('IMPORTS OK')"
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

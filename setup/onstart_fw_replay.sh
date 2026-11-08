#!/bin/bash
exec > >(tee -a /workspace/setup.log) 2>&1
set -x
nvidia-smi
mkdir -p /workspace/libero && cd /workspace
export HF_HOME=/workspace/hf
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
pip install -q -U huggingface_hub
# checkpoint BEFORE pip install -e . (FastWAM pins hub 0.29.2, which removes the `hf` CLI)
time hf download yuanty/fastwam libero_uncond_2cam224.pt --local-dir /workspace/ckpt
[ -s /workspace/ckpt/libero_uncond_2cam224.pt ] || { touch /workspace/SETUP_FAILED; exit 1; }
# real in-domain LIBERO frames: primary + wrist from the released LeRobot v3.0 videos
D=lerobot_v30/libero_goal_no_noops_lerobot/videos
time hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir /workspace/lb \
  --include "$D/observation.images.image/chunk-000/file-000.mp4" \
            "$D/observation.images.wrist_image/chunk-000/file-000.mp4"
cp /workspace/lb/$D/observation.images.image/chunk-000/file-000.mp4       /workspace/libero/main.mp4
cp /workspace/lb/$D/observation.images.wrist_image/chunk-000/file-000.mp4 /workspace/libero/wrist.mp4
ls -la /workspace/libero/
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
python -c "import cv2, imageio.v3, fastwam; print('IMPORTS OK')"
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

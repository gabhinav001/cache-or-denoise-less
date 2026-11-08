#!/bin/bash
exec > >(tee -a /workspace/setup.log) 2>&1
set -x
nvidia-smi
mkdir -p /workspace && cd /workspace
export HF_HOME=/workspace/hf
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
pip install -q -U huggingface_hub
echo "=== downloading DreamZero checkpoint (excl. tensorrt) ==="
time hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir /workspace/ckpt --exclude "tensorrt/*"
[ -s /workspace/ckpt/config.json ] || { echo FATAL_CKPT; touch /workspace/SETUP_FAILED; exit 1; }
du -sh /workspace/ckpt
echo "=== pre-fetching Wan2.1-I2V-14B-480P base ==="
time hf download Wan-AI/Wan2.1-I2V-14B-480P --repo-type model
python -c "
import urllib.request,tarfile,io,os,shutil
d=urllib.request.urlopen('https://codeload.github.com/dreamzero0/dreamzero/tar.gz/refs/heads/main').read()
tarfile.open(fileobj=io.BytesIO(d)).extractall('/src')
n=[x for x in os.listdir('/src') if x.startswith('dreamzero')][0]
shutil.move(os.path.join('/src',n),'/dz'); print('repo at /dz')
"
cd /dz
pip install -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install -q -e . 2>&1 | tail -3
pip uninstall -y -q deepspeed || true
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86
python -c "import cv2, transformers.modeling_utils, flash_attn, imageio; print('IMPORTS OK')"
ls -la /dz/debug_image/
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

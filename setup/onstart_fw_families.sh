#!/bin/bash
exec > >(tee -a /workspace/setup.log) 2>&1
set -x
nvidia-smi
mkdir -p /workspace /w/probes && cd /workspace
# FAIL BEFORE THE 24 GB, NOT AFTER IT. nvidia-smi answering is not evidence that
# torch can use the GPU: a host with driver 555 advertised "CUDA Version: 12.8"
# and reported an A6000 with 46 GB, then failed every torch CUDA call with
# error 803 (unsupported driver/runtime combination) -- CUDA 12.8 needs a 570+
# driver and this image is pinned to cu128. Preflight caught it, but only after
# the checkpoints and video had been downloaded and paid for. One import here
# costs a second and turns a bad host into a 30-second loss.
python - <<'CUDACHK' || { echo "GPU UNUSABLE -- see above"; touch /workspace/SETUP_FAILED; exit 1; }
import sys, torch
try:
    a = torch.zeros(8, device="cuda"); a += 1; torch.cuda.synchronize()
except Exception as e:
    print("torch cannot use this GPU: %s" % e)
    print("driver/runtime mismatch. This image needs CUDA %s; check nvidia-smi above."
          % torch.version.cuda)
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
gb = total / 1e9
print("CUDA OK: %s, %.0f GB, torch %s / cu%s"
      % (torch.cuda.get_device_name(0), gb, torch.__version__, torch.version.cuda))
if gb < 38:
    print("only %.0f GB VRAM; this study needs >= 40" % gb); sys.exit(1)
CUDACHK
# HF's Xet/CAS backend intermittently 401s on anonymous downloads; the classic
# route is slower but does not fail closed mid-setup
export HF_HUB_DISABLE_XET=1
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/workspace/hf
pip install -q -U huggingface_hub
# everything needing the `hf` CLI must precede `pip install -e .`, which pins
# huggingface-hub to 0.29.2 and removes it
time hf download yuanty/fastwam libero_uncond_2cam224.pt --local-dir /workspace/ckpt
[ -s /workspace/ckpt/libero_uncond_2cam224.pt ] || { touch /workspace/SETUP_FAILED; exit 1; }
# The deployment checkpoint too: the idm/idm7 collections replay on the weights
# the closed-loop arms actually run, and without this a reviewer following the
# documented "CKPT=libero_optional_idm_2cam224.pt ... --seeds idm" invocation
# fails at load time with nothing explaining why.
time hf download yuanty/fastwam libero_optional_idm_2cam224.pt      libero_optional_idm_2cam224_dataset_stats.json --local-dir /workspace/ckpt
[ -s /workspace/ckpt/libero_optional_idm_2cam224.pt ] || { touch /workspace/SETUP_FAILED; exit 1; }
for S in libero_goal libero_spatial libero_object libero_10; do
  D=lerobot_v30/${S}_no_noops_lerobot/videos
  # one file per invocation: `--include A B` silently keeps only one of them
  for CAM in observation.images.image observation.images.wrist_image; do
    hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir /workspace/lb \
      "$D/$CAM/chunk-000/file-000.mp4"
  done
  O=/workspace/libero/${S}_no_noops_lerobot; mkdir -p $O
  cp /workspace/lb/$D/observation.images.image/chunk-000/file-000.mp4       $O/main.mp4
  cp /workspace/lb/$D/observation.images.wrist_image/chunk-000/file-000.mp4 $O/wrist.mp4
done
du -sh /workspace/libero/*
python -c "
import urllib.request,tarfile,io,os,shutil
SHA='7faa71108368fbb3b6885649f112af607427a2d4'
d=urllib.request.urlopen(f'https://codeload.github.com/yuantianyuan01/FastWAM/tar.gz/{SHA}').read()
tarfile.open(fileobj=io.BytesIO(d)).extractall('/src')
n=[x for x in os.listdir('/src') if x.startswith('FastWAM')][0]
for f in os.listdir(os.path.join('/src',n)):
    shutil.move(os.path.join('/src',n,f), os.path.join('/w',f))
print('FastWAM at /w')
"
cd /w && sed -i "/deepspeed/d" pyproject.toml
pip install -q -e . --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86 imageio imageio-ffmpeg
python -c "import cv2, imageio.v3, fastwam, numpy; print('IMPORTS OK')" || { touch /workspace/SETUP_FAILED; exit 1; }
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

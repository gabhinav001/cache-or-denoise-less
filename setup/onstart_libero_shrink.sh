#!/bin/bash
# Closed-loop LIBERO with and without per-cell attenuation of the cached residual.
# Follows vast/LIBERO_RECIPE.md exactly, including the two lessons that cost the
# most: mujoco must be 3.3.2 (the version the data was rendered with; 2.3.7 costs
# 21 points of success) and EGL needs the NVIDIA vendor ICD or it silently falls
# back to software rendering at 16x the cost per step.
exec > >(tee -a /workspace/setup.log) 2>&1
set -ex
set -o pipefail
trap 'touch /workspace/SETUP_FAILED' ERR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi
# device_count() can report a GPU on a host whose driver cannot actually
# initialise this CUDA build, so allocate something before spending 20 minutes
# on setup
python -c "
import torch; a=torch.zeros(8, device='cuda'); a+=1; torch.cuda.synchronize()
print('cuda ok:', torch.cuda.get_device_name(0), a.sum().item())
" || { echo FATAL_CUDA; touch /workspace/SETUP_FAILED; exit 1; }
CHECKPOINT_STEMS=()
case "${BASELINE_VARIANT:-both}" in
  original) CHECKPOINT_STEMS=("libero_uncond_2cam224") ;;
  optional) CHECKPOINT_STEMS=("libero_optional_idm_2cam224") ;;
  both|all) CHECKPOINT_STEMS=("libero_optional_idm_2cam224" "libero_uncond_2cam224") ;;
  *) echo "Unknown BASELINE_VARIANT" >&2; exit 2 ;;
esac
# A stopped instance retains its completed environment. Do not reinstall or
# attempt to clone into existing directories when the same instance restarts.
all_ckpts_present=true
for STEM in "${CHECKPOINT_STEMS[@]}"; do
  [ -s "/workspace/ckpt/${STEM}.pt" ] || all_ckpts_present=false
done
if [ -f /workspace/SETUP_DONE ] && [ "$all_ckpts_present" = true ]; then
  python -c "import mujoco, robosuite, libero.libero, fastwam; assert mujoco.__version__ == '3.3.2'"
  rm -f /workspace/SETUP_FAILED
  echo "Retained environment is ready; start an evaluation explicitly."
  exit 0
fi
# HF's Xet/CAS backend intermittently 401s on anonymous downloads; the classic
# route is slower but does not fail closed mid-setup
export HF_HUB_DISABLE_XET=1
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/workspace/hf DEBIAN_FRONTEND=noninteractive
# dpkg can arrive interrupted in a base image, and then apt-get install fails
# while the script carries on: the missing library only surfaces 200 lines later
# as ImportError: libGL.so.1, which reads like a broken host rather than a failed
# install. Recover the state first, then verify the packages are actually there.
dpkg --configure -a || true
apt-get update -qq
apt-get install -y -qq git cmake build-essential libgl1 \
    libglib2.0-0 libosmesa6-dev libegl1 patchelf || {
  echo "FATAL: system package install failed" >&2; touch /workspace/SETUP_FAILED; exit 1; }
for lib in libGL.so.1 libEGL.so.1; do
  ldconfig -p | grep "$lib" >/dev/null || {
    echo "FATAL: $lib missing after apt-get install" >&2
    touch /workspace/SETUP_FAILED; exit 1; }
done
mkdir -p /workspace && cd /workspace
# Keep resumable HTTP downloads: newer hub releases use process-unique temporary
# files and discard them on failure. FastWAM installs its own older hub later.
pip install -q huggingface_hub==0.34.4
# The checkpoint is fetched anonymously, on purpose: every weight this study uses
# is public, and a rented third-party machine is not somewhere to put a token.
# The cost is HuggingFace's anonymous rate limit -- a 12 GB file draws HTTP 429
# under load, and a single attempt then fails the whole setup after the box has
# already been paid for. Retry with backoff; hf resumes a partial download rather
# than restarting it, so a retry is cheap.
fetch_ckpt() {   # <repo> <file>
  local i
  for i in 1 2 3 4 5 6; do
    if time hf download "$1" "$2" --local-dir /workspace/ckpt; then return 0; fi
    echo "  download of $2 failed (attempt $i/6); sleeping $((i*60))s" >&2
    sleep $((i * 60))
  done
  return 1
}
for STEM in "${CHECKPOINT_STEMS[@]}"; do
  fetch_ckpt yuanty/fastwam "${STEM}.pt" \
    || { echo "FATAL: checkpoint $STEM unavailable after 6 attempts" >&2; touch /workspace/SETUP_FAILED; exit 1; }
  fetch_ckpt yuanty/fastwam "${STEM}_dataset_stats.json" \
    || { touch /workspace/SETUP_FAILED; exit 1; }
  [ -s "/workspace/ckpt/${STEM}.pt" ] || { touch /workspace/SETUP_FAILED; exit 1; }
done

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /libero
git -C /libero checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
cd /libero && pip install -q -e .          # package only; requirements.txt would downgrade numpy/transformers/hydra
export CMAKE_POLICY_VERSION_MINIMUM=3.5    # cmake 4 refuses egl_probe's declared minimum without this
pip install -q egl_probe
pip install --timeout 120 --resume-retries 8 "mujoco==2.3.7" "robosuite==1.4.1" "bddl==1.0.1" future easydict thop matplotlib "gym==0.25.2" cloudpickle
echo "N" | python -c "import libero.libero"    # writes ~/.libero/config.yaml instead of prompting

python -c "
import urllib.request,tarfile,io,os,shutil
SHA='7faa71108368fbb3b6885649f112af607427a2d4'
d=urllib.request.urlopen(f'https://codeload.github.com/yuantianyuan01/FastWAM/tar.gz/{SHA}').read()
tarfile.open(fileobj=io.BytesIO(d)).extractall('/src')
n=[x for x in os.listdir('/src') if x.startswith('FastWAM')][0]
os.makedirs('/w', exist_ok=True)
for f in os.listdir(os.path.join('/src',n)): shutil.move(os.path.join('/src',n,f), os.path.join('/w',f))
print('FastWAM at /w')
"
cd /w && sed -i "/deepspeed/d" pyproject.toml
pip install -q -e . --extra-index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip uninstall -y -q opencv-python opencv-python-headless || true
pip install -q opencv-python-headless==4.11.0.86 imageio imageio-ffmpeg
pip install -q "mujoco==3.3.2"             # LAST: must match the version the data was rendered with
mkdir -p /w/probes
# EGL needs an ICD where libglvnd looks, not in /tmp. Without it mujoco's EGL
# backend fails at import with a NoneType eglQueryString, and on hosts that do
# ship one this step is a harmless no-op.
mkdir -p /usr/share/glvnd/egl_vendor.d
if [ ! -s /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
fi
python -c "
import os; os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from OpenGL import EGL; print('EGL ok:', bool(EGL.eglQueryString))
" || { echo 'FATAL: EGL cannot initialise on this host'; touch /workspace/SETUP_FAILED; exit 1; }
# FastWAM fetches these lazily at first model load, which turns a transient
# broken pipe into a failed experiment instead of a failed setup. Fetch them now,
# with retries, one file per invocation (`--include A B` silently keeps only one).
WAN=/w/checkpoints/Wan-AI/Wan2.2-TI2V-5B
mkdir -p "$WAN"
# NOT the `hf` CLI: FastWAM's `pip install -e .` pins huggingface-hub to 0.29.2
# and removes the CLI entirely, so any `hf download` after that point fails with
# "hf: command not found" -- and a retry loop around a missing command retries
# instantly and silently, which is how an entire setup completed with the text
# encoder absent. The python API is always present because the library is.
# hf_transfer is a rust downloader with its own resume; the 10.6 GB encoder
# breaks mid-transfer often enough that plain requests fails repeatedly.
pip install -q hf_transfer || true
python - <<'PYDL'
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
DEST = "/w/checkpoints/Wan-AI/Wan2.2-TI2V-5B"
FILES = ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
         "config.json", "configuration.json",
         "google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model"]
for f in FILES:
    for attempt in range(12):          # each attempt resumes where the last broke
        try:
            hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", f, local_dir=DEST)
            print("got", f, flush=True); break
        except Exception as e:
            print(f"  {f} attempt {attempt+1}: {type(e).__name__}", flush=True)
            time.sleep(5)
    else:
        print("FAILED", f, flush=True)
        if f.endswith(".pth"): sys.exit(1)
PYDL
[ $? -eq 0 ] || { echo "FATAL: Wan common files incomplete"; touch /workspace/SETUP_FAILED; exit 1; }
du -sh "$WAN" 2>/dev/null

[ -s "$WAN/models_t5_umt5-xxl-enc-bf16.pth" ] || {
  echo "FATAL: text encoder never downloaded"; touch /workspace/SETUP_FAILED; exit 1; }
du -sh "$WAN" 2>/dev/null

python -c "import mujoco, robosuite, libero.libero, fastwam; print('mujoco', mujoco.__version__)" || { touch /workspace/SETUP_FAILED; exit 1; }
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

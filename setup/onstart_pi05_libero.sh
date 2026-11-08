#!/bin/bash
exec > >(tee -a /workspace/setup.log) 2>&1
set -x
nvidia-smi
# device_count() can report a GPU on a host whose driver cannot initialise this
# CUDA build; allocate before spending 20 minutes on setup
python -c "
import torch; a=torch.zeros(8, device='cuda'); a+=1; torch.cuda.synchronize()
print('cuda ok:', torch.cuda.get_device_name(0), a.sum().item())
" || { echo FATAL_CUDA; touch /workspace/SETUP_FAILED; exit 1; }
# HF's Xet/CAS backend intermittently 401s on anonymous downloads; the classic
# route is slower but does not fail closed mid-setup
export HF_HUB_DISABLE_XET=1
# DiffSynth pulls the Wan common files from modelscope by default, which
# sustains ~300 kB/s from western datacentres. Same files on HF, 60x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DEBIAN_FRONTEND=noninteractive HF_HOME=/workspace/hf
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
  ldconfig -p | grep -q "$lib" || {
    echo "FATAL: $lib missing after apt-get install" >&2
    touch /workspace/SETUP_FAILED; exit 1; }
done
mkdir -p /workspace
# lerobot requires python >= 3.12; the image ships 3.11, so uv fetches its own
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv --clear --python 3.12 /workspace/pienv || { touch /workspace/SETUP_FAILED; exit 1; }
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # cmake 4 refuses egl_probe's declared minimum
VIRTUAL_ENV=/workspace/pienv uv pip install \
  "lerobot[pi,libero,dataset,video] @ git+https://github.com/huggingface/lerobot.git" \
  2>&1 | tail -4 || { touch /workspace/SETUP_FAILED; exit 1; }
# LIBERO prompts on first import; answer once so the config file is written
echo "N" | /workspace/pienv/bin/python -c "import libero.libero" || true
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /tmp/10_nvidia.json
/workspace/pienv/bin/python -c "
import torch, mujoco, robosuite
from lerobot.policies.pi05 import PI05Policy
from lerobot.envs.libero import LiberoEnv
print('imports ok | torch', torch.__version__, '| mujoco', mujoco.__version__,
      '| cuda', torch.cuda.is_available())
" || { touch /workspace/SETUP_FAILED; exit 1; }
# EGL needs an ICD where libglvnd looks, not in /tmp. Without it mujoco's EGL
# backend fails at import with a NoneType eglQueryString, and on hosts that do
# ship one this step is a harmless no-op.
mkdir -p /usr/share/glvnd/egl_vendor.d
if [ ! -s /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
fi
# The venv interpreter, not the system one. Everything this setup installs lives
# in /workspace/pienv, so a bare `python` here has no PyOpenGL and the check fails
# with ModuleNotFoundError on a host whose EGL is perfectly fine -- after the
# whole setup has already been paid for. That is what it did.
/workspace/pienv/bin/python -c "
import os; os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from OpenGL import EGL; print('EGL ok:', bool(EGL.eglQueryString))
" || { echo 'FATAL: EGL cannot initialise on this host'; touch /workspace/SETUP_FAILED; exit 1; }
echo "=== SETUP COMPLETE ==="
touch /workspace/SETUP_DONE

# LIBERO + FastWAM in one environment — verified recipe

Base image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

## System
apt-get install -y git cmake build-essential libgl1 libglib2.0-0 \
                   libosmesa6-dev libegl1 patchelf

## LIBERO — install the PACKAGE ONLY, never requirements.txt
# requirements.txt hard-pins numpy==1.22.4, transformers==4.21.1, hydra==1.2.0,
# which would break FastWAM (needs numpy 2.x, transformers 4.49, hydra 1.3.2).
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /libero
cd /libero && pip install -e .          # NOT -r requirements.txt

## Simulator stack — versions matter
export CMAKE_POLICY_VERSION_MINIMUM=3.5      # REQUIRED with cmake >= 4
pip install egl_probe
pip install "mujoco==2.3.7" "robosuite==1.4.1" "bddl==1.0.1" \
            future easydict thop matplotlib "gym==0.25.2" cloudpickle
# mujoco 2.3.7 only lets this line resolve; 3.3.2 replaces it below (CRITICAL).
# egl_probe needs cmake; with cmake 4.x its CMakeLists declares a minimum below
# 3.5, which cmake 4 refuses -- CMAKE_POLICY_VERSION_MINIMUM=3.5 is the fix.
# Without it the whole pip line aborts and robosuite is silently absent.

## LIBERO prompts interactively on first import; answer once to write the config
echo "N" | python -c "import libero.libero"     # -> ~/.libero/config.yaml

## CRITICAL: mujoco version must match the data version
# FastWAM's LIBERO data is rendered with mujoco 3.3.2. Evaluating under 2.3.7
# (what LIBERO's requirements resolve to) drops success 97.5% -> 76.2%.
# Documented ONLY in README_zh.md: install LIBERO normally, then LAST:
pip install "mujoco==3.3.2"
# NOTE: mujoco 3.12 does NOT work (robosuite 1.4.1 asserts on joint type); 3.3.2 does.

## Rendering: EGL is 16x faster than OSMesa (27 ms/step vs 431 ms/step)
# The container ships libEGL_nvidia.so.0 but often only the MESA vendor ICD,
# so EGL silently falls back to software. Provide the NVIDIA ICD. /usr/share
# may be read-only, so prefer the env-var route:
cat > /tmp/10_nvidia.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/10_nvidia.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

## torch 2.7 defaults weights_only=True; LIBERO init states are pickled numpy
#   _ld = torch.load
#   torch.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})

## FastWAM (after LIBERO, so its pins win)
#   tarball at SHA 7faa71108368fbb3b6885649f112af607427a2d4
#   sed -i "/deepspeed/d" pyproject.toml ; pip install -e .
#   pip uninstall -y opencv-python; pip install opencv-python-headless==4.11.0.86

## Also needed for eval (not just the .pt):
#   yuanty/fastwam : libero_uncond_2cam224_dataset_stats.json

## Verified: numpy 2.2.6 | mujoco 3.3.2 | robosuite 1.4.1 | fastwam OK | libero OK
## Timing: EGL 27 ms/step -> ~8 s per 300-step episode

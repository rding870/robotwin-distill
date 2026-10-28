# Source before any work:  source /root/robotwin-distill/env.sh
export PROJ=/root/robotwin-distill
export UV_CACHE_DIR=/root/.cache/uv                  # /workspace (geesefs FUSE) can't host uv's cache
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}   # large weights live on the big mount
export ROBOTWIN_ROOT=$PROJ/RoboTwin
export PYTHONPATH=$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}
export VK_ICD_FILENAMES=$PROJ/configs/vulkan/nvidia_icd_egl.json  # system GLX ICD fails headless (NVIDIA_DRIVER_CAPABILITIES=compute,utility)
export TORCH_CUDA_ARCH_LIST="8.6"                    # RTX A6000
source $PROJ/.venv/bin/activate

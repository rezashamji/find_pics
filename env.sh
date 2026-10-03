# source env.sh  -- forces every cache/env/model download to live INSIDE find_pics/.
# Nothing this project writes may land outside this folder (shared lab storage rule).
export FP_ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
export HF_HOME=$FP_ROOT/.cache/huggingface
# Read the existing HF login token in place; the token is never copied into find_pics/.
export HF_TOKEN_PATH=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/.cache/huggingface/token
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_CACHE_DIR=$FP_ROOT/.cache/uv
export UV_PYTHON_INSTALL_DIR=$FP_ROOT/.cache/uv-python
export UV_TOOL_DIR=$FP_ROOT/.cache/uv-tools
export PIP_CACHE_DIR=$FP_ROOT/.cache/pip
export TORCH_HOME=$FP_ROOT/.cache/torch
export XDG_CACHE_HOME=$FP_ROOT/.cache/xdg
export VLLM_CACHE_ROOT=$FP_ROOT/.cache/vllm
export TRITON_CACHE_DIR=$FP_ROOT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$FP_ROOT/.cache/inductor
export CUDA_CACHE_PATH=$FP_ROOT/.cache/nv
export MPLCONFIGDIR=$FP_ROOT/.cache/mpl
export INSIGHTFACE_HOME=$FP_ROOT/models/insightface
export TMPDIR=$FP_ROOT/.cache/tmp
export PATH=$FP_ROOT/tools/bin:$PATH
mkdir -p "$TMPDIR" "$FP_ROOT/tools/bin"
# activate main env if it exists
if [ -f "$FP_ROOT/envs/fp/bin/activate" ]; then source "$FP_ROOT/envs/fp/bin/activate"; fi
# Keep library caches inside find_pics (vLLM/flashinfer/rust otherwise write to ~)
export FLASHINFER_WORKSPACE_BASE=$FP_ROOT/.cache
export FLASHINFER_CACHE_DIR=$FP_ROOT/.cache/flashinfer
export VLLM_CONFIG_ROOT=$FP_ROOT/.cache/vllm-config
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export RUSTUP_HOME=$FP_ROOT/.cache/rustup
export CARGO_HOME=$FP_ROOT/.cache/cargo
# GitHub CLI (installed in envs/gh); its login token lives in data/private/gh (chmod 700, never committed)
export GH_CONFIG_DIR="$FP_ROOT/data/private/gh"
export PATH="$FP_ROOT/envs/gh/gh_2.102.0_linux_amd64/bin:$PATH"

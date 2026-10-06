#!/bin/bash
# Builds the openvla env on a fresh machine: sibling repos, conda env from
# requirements-lock.txt, LIBERO config, checkpoint, and flash-attn on GPU.
# Idempotent. Prints each step's wall time and saves the summary under outputs/.
# Run with bash, then `source setup_env.sh`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(dirname "$REPO_DIR")"
LOCK_FILE="$REPO_DIR/requirements-lock.txt"

ENV_NAME=openvla
PYTHON_VERSION=3.10.20
CHECKPOINT=openvla/openvla-7b-finetuned-libero-10
FLASH_ATTN_VERSION=2.5.5
OPENVLA_REV=c8f03f48af692657d3060c19588038c7220e9af9
LIBERO_REV=8f1084e3132a39270c3a13ebe37270a43ece2a01

CONDA_BIN="${CONDA_EXE:-}"
[ -x "$CONDA_BIN" ] || CONDA_BIN="$HOME/miniforge3/bin/conda"
[ -x "$CONDA_BIN" ] || CONDA_BIN="$HOME/miniconda3/bin/conda"
if [ ! -x "$CONDA_BIN" ]; then
  echo "ERROR: conda not found; install Miniforge first (https://github.com/conda-forge/miniforge#install)." >&2
  exit 1
fi

TIMINGS=()
PROVISION_T0=$SECONDS
timed() {
  local name=$1 t0=$SECONDS
  shift
  echo "== $name"
  "$@"
  TIMINGS+=("$(printf '%6ds  %s' $((SECONDS - t0)) "$name")")
}

has_gpu() { command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; }

env_prefix() { "$CONDA_BIN" env list | awk -v n="$ENV_NAME" '$1 == n { print $NF }'; }

check_gpu() {
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  # torch 2.2.0+cu121 ships kernels up to sm_90; Blackwell fails at the first CUDA op.
  if [ "${cap%%.*}" -ge 10 ]; then
    echo "ERROR: GPU compute capability $cap is unsupported by torch 2.2.0+cu121; pick a pre-Blackwell GPU." >&2
    exit 1
  fi
  echo "   compute capability $cap"
}

install_build_tools() {
  # evdev (via robosuite's pynput) ships only an sdist and compiles against kernel headers.
  if dpkg -s build-essential linux-libc-dev >/dev/null 2>&1; then
    echo "   already installed"
  else
    apt-get update && apt-get install -y build-essential linux-libc-dev
  fi
}

clone_at() {
  local url=$1 dir=$2 rev=$3
  [ -d "$dir/.git" ] || git clone "$url" "$dir"
  git -C "$dir" checkout --quiet "$rev"
}

clone_siblings() {
  clone_at https://github.com/openvla/openvla.git "$WORK_ROOT/openvla" "$OPENVLA_REV"
  clone_at https://github.com/Lifelong-Robot-Learning/LIBERO.git "$WORK_ROOT/LIBERO" "$LIBERO_REV"
  # Missing upstream; without it find_packages() never registers `libero`.
  touch "$WORK_ROOT/LIBERO/libero/__init__.py"
}

create_env() {
  if [ -n "$(env_prefix)" ]; then
    echo "   $ENV_NAME already exists"
  else
    "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
}

install_deps() {
  # The lock is a complete freeze; resolving it fails on opencv's declared numpy>=2,
  # a false conflict (README Troubleshooting).
  pip install --no-deps -r "$LOCK_FILE"
  pip install --no-deps -e "$WORK_ROOT/openvla" -e "$WORK_ROOT/LIBERO"
}

seed_libero_config() {
  if [ -f "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" ]; then
    echo "   config already present"
  else
    # First import prompts for a dataset path; N writes the defaults.
    echo N | python -c "import libero.libero"
  fi
}

prefetch_checkpoint() {
  HF_HUB_ENABLE_HF_TRANSFER=1 python -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('$CHECKPOINT'))"
}

install_flash_attn() {
  if python -c "import flash_attn" 2>/dev/null; then
    echo "   flash-attn already installed"
  else
    # MAX_JOBS bounds memory if pip falls back to compiling from source.
    MAX_JOBS="${MAX_JOBS:-4}" pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
  fi
}

if [ ! -f "$LOCK_FILE" ]; then
  echo "ERROR: $LOCK_FILE not found." >&2
  exit 1
fi

has_gpu && timed "gpu check" check_gpu
timed "build tools" install_build_tools
timed "clone siblings" clone_siblings
timed "create env" create_env

ENV_PREFIX="$(env_prefix)"
export PATH="$ENV_PREFIX/bin:$PATH"
export PIP_CONSTRAINT="$REPO_DIR/constraints.txt"
# Same local cache paths as setup_env.sh, overriding RunPod images' /workspace defaults.
export HF_HOME="$HOME/.cache/huggingface"
export PIP_CACHE_DIR="$HOME/.cache/pip"

timed "install deps" install_deps
timed "seed LIBERO config" seed_libero_config
timed "prefetch checkpoint" prefetch_checkpoint
has_gpu && timed "flash-attn" install_flash_attn

mkdir -p "$REPO_DIR/outputs"
summary="$REPO_DIR/outputs/provision_$(date +%Y%m%d_%H%M%S).log"
{
  echo "host: ${RUNPOD_POD_ID:-$(hostname)}"
  has_gpu && echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  printf '%s\n' "${TIMINGS[@]}"
  printf '%6ds  total\n' $((SECONDS - PROVISION_T0))
  du -sh "$ENV_PREFIX" "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null
  df -h "$HOME" | tail -1
} | tee "$summary"
echo "Summary written to $summary"
echo "Next: source setup_env.sh && conda activate $ENV_NAME"

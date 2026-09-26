#!/bin/bash
# =============================================================================
# setup_env.sh — Physical AI Spine Project (OpenVLA on LIBERO)
# Per-shell pod setup, run after provision.sh: MuJoCo rendering libs, the
# openvla env's vars, and the VS Code tunnel. Shell and git config are machine
# setup and not handled here.
# =============================================================================

# Derived from this file's location so a clone under any directory name works.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_DIR

# Same search order as the machine's shell rc.
CONDA_BIN="${CONDA_EXE:-}"
[ -x "$CONDA_BIN" ] || CONDA_BIN="$HOME/miniforge3/bin/conda"
[ -x "$CONDA_BIN" ] || CONDA_BIN="$HOME/miniconda3/bin/conda"

# --- MuJoCo headless rendering (GPU-aware backend + libs) ---
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    export MUJOCO_GL=egl
    if ! dpkg -s libegl1 &> /dev/null; then
        apt-get update && apt-get install -y libegl1 libegl-dev
        ldconfig
    fi
else
    export MUJOCO_GL=osmesa
    if ! dpkg -s libosmesa6 &> /dev/null; then
        apt-get update && apt-get install -y libosmesa6 libosmesa6-dev libglfw3 libglew2.2 libgl1
        ldconfig
    fi
fi
echo "MUJOCO_GL set to: $MUJOCO_GL"

# --- opencv-python system libs (the non-headless wheel links glib, libSM and libGL) ---
# Checked by soname: Ubuntu 24.04 renamed the glib package to libglib2.0-0t64.
for _lib in libgthread-2.0.so.0 libSM.so.6 libGL.so.1; do
    if ! ldconfig -p | grep -q "$_lib"; then
        apt-get update && apt-get install -y libglib2.0-0 libsm6 libice6 libgl1
        break
    fi
done
unset _lib

# --- Pip numpy constraint (prevents unpinned deps like bddl from silently upgrading numpy) ---
export PIP_CONSTRAINT="$REPO_DIR/constraints.txt"

export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Caches on local disk; RunPod images point them at /workspace, which may be object storage ---
# transformers chmods its remote-code module cache, which object storage rejects.
export HF_HOME="$HOME/.cache/huggingface"
export PIP_CACHE_DIR="$HOME/.cache/pip"

# --- Persist the above as openvla env vars, applied on `conda activate openvla` ---
if [ -x "$CONDA_BIN" ] && "$CONDA_BIN" env list | grep -q '^openvla '; then
    "$CONDA_BIN" env config vars set -n openvla \
        MUJOCO_GL="$MUJOCO_GL" \
        PIP_CONSTRAINT="$PIP_CONSTRAINT" \
        HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
        HF_HOME="$HF_HOME" \
        PIP_CACHE_DIR="$PIP_CACHE_DIR" > /dev/null
    echo "openvla env vars set."
else
    echo "NOTE: conda env 'openvla' not found; run provision.sh, then re-source this file."
fi

# --- VS Code Tunnel (spawn only — login is manual, do NOT auto-answer prompts) ---
VSCODE_CLI="$HOME/.local/bin/code"
if [ ! -x "$VSCODE_CLI" ]; then
    mkdir -p "$(dirname "$VSCODE_CLI")"
    curl -fsSL 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /tmp/vscode_cli.tar.gz \
        && tar -xzf /tmp/vscode_cli.tar.gz -C "$(dirname "$VSCODE_CLI")"
    rm -f /tmp/vscode_cli.tar.gz
fi
if [ -x "$VSCODE_CLI" ]; then
    if tmux has-session -t vscode-tunnel 2>/dev/null; then
        echo "VS Code tunnel session already running/waiting."
    else
        tmux new-session -d -s vscode-tunnel "\"$VSCODE_CLI\" tunnel --accept-server-license-terms"
        echo "VS Code tunnel session spawned."
    fi
    echo ""
    echo "  >>> ACTION REQUIRED: tmux attach -t vscode-tunnel"
    echo "  >>> Select GitHub when prompted, then complete the device-code login."
else
    echo "WARNING: could not install the VS Code CLI at $VSCODE_CLI; skipping tunnel."
fi

echo ""
echo "Setup complete. Verifying..."
command -v conda > /dev/null && conda --version || echo "WARNING: conda not on PATH (machine setup incomplete)"
[ -x "$CONDA_BIN" ] && "$CONDA_BIN" env config vars list -n openvla 2>/dev/null

echo ""
echo "Next steps:"
echo "  1. tmux attach -t vscode-tunnel   (select GitHub, log in)"
echo "  2. conda activate openvla"

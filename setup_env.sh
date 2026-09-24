#!/bin/bash
# =============================================================================
# setup_env.sh — Physical AI Spine Project (OpenVLA on LIBERO)
# Repo-level pod setup after a RunPod container restart/redeploy: LIBERO config
# path, MuJoCo rendering libs, the openvla env's vars, and the VS Code
# tunnel. Shell and git config are machine setup and not handled here.
# =============================================================================

# Derived from this file's location so a clone under any directory name works.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_DIR
CONDA_BIN=/workspace/miniconda3/bin/conda

# --- LIBERO config on persistent volume ---
export LIBERO_CONFIG_PATH=/workspace/.libero
mkdir -p "$LIBERO_CONFIG_PATH"

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

# --- Pip numpy constraint (prevents unpinned deps like bddl from silently upgrading numpy) ---
export PIP_CONSTRAINT="$REPO_DIR/constraints.txt"

# --- Persist the above as openvla env vars, applied on `conda activate openvla` ---
if [ -x "$CONDA_BIN" ] && "$CONDA_BIN" env list | grep -q '^openvla '; then
    "$CONDA_BIN" env config vars set -n openvla \
        LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
        MUJOCO_GL="$MUJOCO_GL" \
        PIP_CONSTRAINT="$PIP_CONSTRAINT" > /dev/null
    echo "openvla env vars set."
else
    echo "NOTE: conda env 'openvla' not found; re-run after creating it to persist env vars."
fi

# --- VS Code Tunnel (spawn only — login is manual, do NOT auto-answer prompts) ---
if [ ! -x "/workspace/code" ]; then
    curl -fsSL 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /tmp/vscode_cli.tar.gz \
        && tar -xzf /tmp/vscode_cli.tar.gz -C /workspace
    rm -f /tmp/vscode_cli.tar.gz
fi
if [ -x "/workspace/code" ]; then
    if tmux has-session -t vscode-tunnel 2>/dev/null; then
        echo "VS Code tunnel session already running/waiting."
    else
    tmux new-session -d -s vscode-tunnel "cd /workspace && ./code tunnel --accept-server-license-terms --extensions-dir /workspace/.vscode-server/extensions --server-data-dir /workspace/.vscode-server/data"
        echo "VS Code tunnel session spawned."
    fi
    echo ""
    echo "  >>> ACTION REQUIRED: tmux attach -t vscode-tunnel"
    echo "  >>> Select GitHub when prompted, then complete the device-code login."
else
    echo "WARNING: could not install the VS Code CLI at /workspace/code; skipping tunnel."
fi

echo ""
echo "Setup complete. Verifying..."
command -v conda > /dev/null && conda --version || echo "WARNING: conda not on PATH (machine setup incomplete)"
[ -x "$CONDA_BIN" ] && "$CONDA_BIN" env config vars list -n openvla 2>/dev/null

echo ""
echo "Next steps:"
echo "  1. tmux attach -t vscode-tunnel   (select GitHub, log in)"
echo "  2. conda activate openvla"

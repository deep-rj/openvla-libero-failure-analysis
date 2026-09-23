#!/bin/bash
# =============================================================================
# setup_env.sh — Physical AI Spine Project (OpenVLA on LIBERO)
# Re-establishes shell config after a RunPod container restart/redeploy.
# =============================================================================

# Derived from this file's location so a clone under any directory name works.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_DIR

# --- Conda ---
if [ -d "/workspace/miniconda3" ]; then
    /workspace/miniconda3/bin/conda init bash > /dev/null
    source ~/.bashrc
else
    echo "WARNING: /workspace/miniconda3 not found. Conda env may not exist yet."
fi

# --- Git identity + credential storage on persistent volume ---
git config --global user.name "Rohan Jha"
git config --global user.email "22083737+deep-rj@users.noreply.github.com"
git config --global init.defaultBranch main
git config --global --add safe.directory '*'
if ! grep -q "credential.helper" ~/.gitconfig 2>/dev/null; then
    git config --global credential.helper "store --file=/workspace/.git-credentials"
fi

# --- HuggingFace cache on persistent volume ---
export HF_HOME=/workspace/hf_cache
if ! grep -q "HF_HOME=/workspace/hf_cache" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
fi
mkdir -p /workspace/hf_cache

# --- LIBERO config on persistent volume ---
mkdir -p /workspace/.libero
export LIBERO_CONFIG_PATH=/workspace/.libero
if ! grep -q "LIBERO_CONFIG_PATH=/workspace/.libero" ~/.bashrc 2>/dev/null; then
    echo "export LIBERO_CONFIG_PATH=/workspace/.libero" >> ~/.bashrc
fi

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
sed -i '/^export MUJOCO_GL=/d' ~/.bashrc 2>/dev/null
echo "export MUJOCO_GL=$MUJOCO_GL" >> ~/.bashrc
echo "MUJOCO_GL set to: $MUJOCO_GL"

# --- Pip numpy constraint (prevents unpinned deps like bddl from silently upgrading numpy) ---
export PIP_CONSTRAINT="$REPO_DIR/constraints.txt"
sed -i '/^export PIP_CONSTRAINT=/d' ~/.bashrc 2>/dev/null
echo "export PIP_CONSTRAINT=$PIP_CONSTRAINT" >> ~/.bashrc

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

source ~/.bashrc
echo ""
echo "Setup complete. Verifying..."
echo "HF_HOME: $HF_HOME"
which conda && conda --version

echo ""
echo "Next steps:"
echo "  1. tmux attach -t vscode-tunnel   (select GitHub, log in)"
echo "  2. conda activate openvla"

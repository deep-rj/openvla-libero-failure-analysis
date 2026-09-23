# libero-eval-harness-workspace

Language-conditioned VLA agent (OpenVLA + LoRA) evaluated on LIBERO-Long, with an eval
harness, failure taxonomy, and robustness study on top.

## Repo layout

```
.
├── scripts/              # smoke tests, eval harness, diagnostics
├── setup_env.sh          # infra setup — run on every pod restart
├── .gitignore
└── README.md
```

`openvla` and `LIBERO` are cloned as sibling directories on the persistent volume,
outside this repo — they're external dependencies with their own git history, not
vendored in.

## Setup

### 1. Install Miniconda (if not already installed)

```bash
CONDA_DIR=/workspace/miniconda3   # on a pod; use "$HOME/miniconda3" locally
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
"$CONDA_DIR/bin/conda" init bash
source ~/.bashrc
```

On a pod, Miniconda **must** go under `/workspace`: `$HOME` is container storage and is
wiped on every teardown, taking the `openvla` env with it. `setup_env.sh` expects
`/workspace/miniconda3` and re-runs `conda init` on each fresh container, because the
init block it writes lives in the (non-persistent) `~/.bashrc`.

Use the `MacOSX-arm64` / `MacOSX-x86_64` installer instead of `Linux-x86_64` if you're
on a Mac — see the [Miniconda install docs](https://www.anaconda.com/docs/getting-started/miniconda/install)
for the full installer matrix. Skip this step if `conda --version` already works — or,
on a pod, if `/workspace/miniconda3/bin/conda` exists (then just `source setup_env.sh`).

### 2. Create the conda environment

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
```

`setup_env.sh` and the rest of this guide assume an env named `openvla` — the exact
Python patch version isn't pinned at creation time, but everything downstream is
verified against 3.10.20 (see Verify, below).

### 3. Infra setup

```bash
source setup_env.sh   # must be sourced, not run with `bash`, to affect the current shell
conda activate openvla
```

This handles everything environment- and hardware-dependent that needs re-applying on
every fresh pod: HuggingFace cache location, a pip constraint pinning `numpy` (see
Troubleshooting), GPU-aware MuJoCo rendering backend selection (installing system
rendering libraries only if missing), and spawning a VS Code Tunnel session for remote
development.

The tunnel step requires one manual action every time — it does not auto-authenticate:

```bash
tmux attach -t vscode-tunnel
# select GitHub when prompted, complete the device-code login, then Ctrl-b d to detach
```

### 4. Clone and install dependencies

```bash
git clone https://github.com/openvla/openvla.git
pip install -e openvla
```

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r openvla/experiments/robot/libero/libero_requirements.txt
```

Notes:

- `pip install -e LIBERO` registers the `libero` package as importable. It does **not**
  install LIBERO's runtime dependencies (robosuite, bddl, etc.) — that's what
  `libero_requirements.txt` handles. Both steps are required.
- `libero_requirements.txt` lives inside the **openvla** repo (not openvla-oft), at
  `experiments/robot/libero/`, and is version-pinned to coexist with openvla's own
  torch/transformers stack. Don't use LIBERO's own top-level `requirements.txt` — that
  one pairs with LIBERO's standalone instructions (Python 3.8, torch 1.11+cu113), which
  conflicts with the openvla env's modern torch stack.

### Verify

```bash
python --version
# Expected: Python 3.10.20
```

```bash
python -c "import torch, flash_attn, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, flash_attn.__version__)"
# Expected: 2.2.0+cu121 True 1.26.4 2.5.5   (CUDA True requires a GPU pod)
```

### 5. Validate the LIBERO environment (CPU or GPU)

```bash
python scripts/libero_smoke_test.py
```

Loads a LIBERO-10 task and runs a random-action rollout, asserting observation
shapes/dtypes and the saved HDF5 output are well-formed. Works on CPU-only pods
(`osmesa` rendering) as well as GPU pods (`egl`) — see MuJoCo headless rendering under
Troubleshooting if rendering fails. Written as a jupytext percent-format notebook
(`# %%` cell markers) — run cell-by-cell in an interactive kernel rather than as a
straight script when iterating on it.

### 6. Validate the OpenVLA checkpoint (GPU only)

```bash
python scripts/smoke_test.py
```

Downloads `openvla/openvla-7b` (~15GB, lands in `$HF_HOME`) on first run, then executes
a single forward pass to confirm the model, GPU, and flash-attn are all wired together
correctly. Expect ~15GB peak memory and a 7-dim action vector as output. Requires a GPU
pod — `flash_attention_2` has no CPU implementation, so this step fails outright on a
CPU-only machine.

---

## Troubleshooting

### `bddl` silently upgrades numpy

`robosuite==1.4.1` (pinned by `libero_requirements.txt`) pulls in `bddl` unpinned.
`bddl`'s own `setup.py` declares a bare `numpy` dependency with no version constraint,
so pip resolves to the latest available — breaking `torch` import (numpy 2.x's ABI is
incompatible with numpy-1.x-compiled wheels).

Fixed via a pip constraints file rather than editing the vendored requirements file
directly, so it applies to every future install in the env, not just this one:

```bash
echo "numpy==1.26.4" >> "$REPO_DIR/constraints.txt"
export PIP_CONSTRAINT="$REPO_DIR/constraints.txt"   # already set by setup_env.sh
```

If numpy has already drifted in an existing env: `pip install numpy==1.26.4 --force-reinstall`.

### `robosuite` pulls an opencv build that looks incompatible

`robosuite` also pulls `opencv-python` unpinned, landing on a version that declares
`numpy>=2` — conflicting with the pin above. Pip prints a dependency-conflict warning,
but this does not break at runtime. Verified directly (not just checked at import):

```python
import numpy as np, cv2
arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
print(cv2.resize(arr, (64, 64)).shape)   # (64, 64, 3) — confirmed working
```

Pip's conflict warnings are declared-constraint checks, not runtime tests — worth
verifying actual interop (especially anything crossing the numpy C-API boundary)
before chasing a fix that isn't needed.

### MuJoCo headless rendering

MuJoCo needs an explicit backend on a pod with no display:
- `glfw` (default) — needs a real display, fails headless.
- `osmesa` — software rendering, CPU-only.
- `egl` — GPU-accelerated, headless. Preferred whenever a GPU is present.

`setup_env.sh` auto-detects via `nvidia-smi` and sets this on every restart, since this
project moves between GPU and CPU-only pods and a value baked in once would go stale.

`osmesa` requires system libraries not installable via pip. The symptom of them being
missing is:
```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```
Diagnose with `ldconfig -p | grep -i osmesa` (empty output confirms missing); fixed via
`apt-get install -y libosmesa6 libosmesa6-dev libglfw3 libglew2.0 libgl1-mesa-glx` +
`ldconfig` to rebuild the shared-library cache. `setup_env.sh` handles this
automatically, guarded to only run when the libraries aren't already present.

### LIBERO's editable install fails with `ModuleNotFoundError: No module named 'libero'`

`libero/__init__.py` is missing from the LIBERO repo itself — confirmed via `git log
--all --oneline -- libero/__init__.py` returning no history, not something broken in a
specific clone. `setup.py` uses `find_packages()`, which requires `__init__.py` to
discover a package at all, so `pip install -e LIBERO` never registers `libero` even
though `pip show libero` reports it as installed.

Symptom: `pip show libero` looks fine, but `import libero` raises
`ModuleNotFoundError`.

Fix:
```bash
touch /workspace/LIBERO/libero/__init__.py
cd /workspace && pip install -e LIBERO --force-reinstall --no-deps
```

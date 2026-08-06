# libero-eval-harness

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

### 1. Infra setup

```bash
source setup_env.sh
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

### 2. Clone and install dependencies

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

### 3. Pull the base checkpoint

```bash
python scripts/smoke_test.py
```

Downloads `openvla/openvla-7b` (~15GB, lands in `$HF_HOME`) on first run, then executes
a single forward pass to confirm the model, GPU, and flash-attn are all wired together
correctly. Expect ~15GB peak memory and a 7-dim action vector as output.

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
echo "numpy==1.26.4" > /workspace/constraints.txt
export PIP_CONSTRAINT=/workspace/constraints.txt   # already set by setup_env.sh
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

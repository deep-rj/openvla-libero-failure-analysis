# libero-eval-harness-workspace

Language-conditioned VLA agent (OpenVLA + LoRA) evaluated on LIBERO-Long, with an eval
harness, failure taxonomy, and robustness study on top.

## Repo layout

```
.
├── scripts/              # smoke tests, eval harness, diagnostics
├── setup_env.sh          # infra setup — run on every pod restart
├── constraints.txt       # pip constraints applied via PIP_CONSTRAINT
├── .gitignore
└── README.md
```

`openvla` and `LIBERO` are cloned as sibling directories on the persistent volume
(`/workspace/openvla`, `/workspace/LIBERO`), outside this repo — they're external
dependencies with their own git history, not vendored in.

The policy under evaluation is `openvla/openvla-7b-finetuned-libero-10` everywhere: smoke
tests, prefetch, and the eval harness all use the same checkpoint.

## Setup

Machine-level shell and git config (conda hooked into new shells, git identity and
credentials, the Hugging Face cache persisted on `/workspace`) is assumed to be in place;
this repo's setup doesn't touch `~/.bashrc` or `~/.gitconfig`.

Setup has three stages. Everything in stages A and C works on a CPU-only pod, so the
expensive parts of a fresh volume can be done without paying for a GPU.

| Stage | Runs | Where | Steps |
|---|---|---|---|
| A. Volume setup | once per volume | CPU or GPU pod | 0–7 |
| B. GPU setup | once per volume | GPU pod | 8–10 |
| C. Every pod start | every fresh container | CPU or GPU pod | [Every pod start](#every-pod-start) |

Run everything from `/workspace` on a pod. Locally, substitute a directory of your choice
for `/workspace` (siblings of this repo) and `$HOME/miniconda3` for the Miniconda path.

### Stage A — volume setup (CPU pod is fine)

#### 0. Clone this repo

```bash
cd /workspace
git clone https://github.com/deep-rj/libero-eval-harness-workspace.git
cd libero-eval-harness-workspace
```

Git credentials and identity come from machine setup, so the clone needs them in place
first.

#### 1. Install Miniconda (skip if `/workspace/miniconda3/bin/conda` exists)

```bash
CONDA_DIR=/workspace/miniconda3   # on a pod; use "$HOME/miniconda3" locally
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"   # current shell only
```

On a pod, Miniconda **must** go under `/workspace`: `$HOME` is container storage and is
wiped on every teardown, taking the `openvla` env with it. `setup_env.sh` expects
`/workspace/miniconda3`. Hooking conda into new shells is part of machine setup, not
this repo.

Use the `MacOSX-arm64` / `MacOSX-x86_64` installer instead of `Linux-x86_64` if you're
on a Mac — see the [Miniconda install docs](https://www.anaconda.com/docs/getting-started/miniconda/install)
for the full installer matrix.

#### 2. Create the conda environment

```bash
conda create -n openvla python=3.10 -y
```

`setup_env.sh` and the rest of this guide assume an env named `openvla` — the exact
Python patch version isn't pinned at creation time, but everything downstream is
verified against 3.10.20 (see step 4).

#### 3. Infra setup

```bash
source setup_env.sh   # must be sourced, not run with `bash`, to affect the current shell
conda activate openvla
```

This handles everything environment- and hardware-dependent that needs re-applying on
every fresh pod. It stores `LIBERO_CONFIG_PATH` (on `/workspace`), a pip constraints file
(see Troubleshooting), and a GPU-aware MuJoCo rendering backend as `openvla` env vars
(`conda env config vars`), applied on every `conda activate openvla`. It also installs
system rendering libraries if missing, and spawns a VS Code Tunnel session for remote
development (downloading the standalone VS Code CLI to `/workspace/code` first if it is
missing).

The tunnel step requires one manual action every time — it does not auto-authenticate:

```bash
tmux attach -t vscode-tunnel
# select GitHub when prompted, complete the device-code login, then Ctrl-b d to detach
```

**Claude Code on a pod:** the `SessionStart` hook rewrites and pushes `STATUS.md`, which
conflicts with the same hook running on another machine. Skip it for the pod's clone once;
the file is gitignored and lives on the persistent volume:

```bash
echo '{"env": {"SKIP_STATUS_BACKFILL": "1"}}' > .claude/settings.local.json
```

Don't use `disableAllHooks` for this: it also disables user-level hooks and the custom
status line.

#### 4. Clone and install dependencies

With the `openvla` env active:

```bash
cd /workspace
git clone https://github.com/openvla/openvla.git
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
touch LIBERO/libero/__init__.py
pip install -e openvla
pip install -e LIBERO
pip install -r openvla/experiments/robot/libero/libero_requirements.txt
```

Notes:

- `touch LIBERO/libero/__init__.py` is required: the file is missing upstream, so without it
  `pip install -e LIBERO` reports success but `import libero` fails (see Troubleshooting).
- `pip install -e LIBERO` registers the `libero` package. It does **not** install LIBERO's
  runtime dependencies (robosuite, bddl, etc.) — that's what `libero_requirements.txt`
  handles. Both steps are required.
- `libero_requirements.txt` lives inside the **openvla** repo (not openvla-oft), at
  `experiments/robot/libero/`, and is version-pinned to coexist with openvla's own
  torch/transformers stack. Don't use LIBERO's own top-level `requirements.txt` — that
  one pairs with LIBERO's standalone instructions (Python 3.8, torch 1.11+cu113), which
  conflicts with the openvla env's modern torch stack.
- `pip install -e openvla` pulls `torch==2.2.0` (the cu121 build) together with its NVIDIA
  runtime libraries, even on a CPU-only pod. That is expected: on a GPU pod the same install
  picks up the host driver with no reinstall (driver supporting CUDA 12.1, i.e. ≥ 525.60).
  Blackwell GPUs (sm_120) are not supported by this torch version.

Check the interpreter:

```bash
python --version
# Expected: Python 3.10.20
```

#### 5. Seed the LIBERO config

```bash
echo N | python -c "import libero.libero"
```

The first import of `libero` prompts interactively for a dataset path when
`$LIBERO_CONFIG_PATH/config.yaml` doesn't exist (`setup_env.sh` points this at
`/workspace/.libero`). Answering `N` writes the default config non-interactively, so later
scripts don't block on the prompt.

#### 6. Prefetch the checkpoint

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openvla/openvla-7b-finetuned-libero-10')"
```

~15GB into the Hugging Face cache. Before downloading, confirm the cache resolves under
`/workspace` (machine setup redirects it, via `HF_HOME` or a symlink), or the download lands
in `$HOME` and is lost on teardown:

```bash
readlink -f "$(python -c 'from huggingface_hub import constants as c; print(c.HF_HUB_CACHE)')"
# Expected: a path under /workspace
```

Doing it here keeps the download off the GPU-billed clock. The checkpoint's
`dataset_statistics.json` holds one normalization key, `libero_10`, which is the
`unnorm_key` the scripts use.

#### 7. Validate the LIBERO environment (CPU or GPU)

```bash
python scripts/libero_smoke_test.py
```

Loads a LIBERO-10 task and runs a random-action rollout, asserting observation
shapes/dtypes and the saved HDF5 output are well-formed. Works on CPU-only pods
(`osmesa` rendering) as well as GPU pods (`egl`) — see MuJoCo headless rendering under
Troubleshooting if rendering fails. Written as a jupytext percent-format notebook
(`# %%` cell markers) — run cell-by-cell in an interactive kernel rather than as a
straight script when iterating on it.

### Stage B — GPU setup (first GPU pod on the volume)

#### 8. Install flash-attn

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

flash-attn needs `nvcc` (`nvcc --version`). If the pod has it, this can also be run in stage
A, but if pip falls back to compiling from source, that can take a long time — stop it
and run it on the GPU pod instead (`MAX_JOBS=4` bounds memory use). Once installed it lives
in the env on the volume, so this is one-time.

#### 9. Verify the GPU stack

```bash
python -c "import torch, flash_attn, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, flash_attn.__version__)"
# Expected: 2.2.0+cu121 True 1.26.4 2.5.5
```

#### 10. Validate the OpenVLA checkpoint

```bash
python scripts/smoke_test.py
```

Loads `openvla/openvla-7b-finetuned-libero-10` from the Hugging Face cache (downloading it if step 6
was skipped), then executes a forward pass to confirm the model, GPU, and flash-attn are
wired together, and prints the 7-dim action vector, timings, and peak memory. Expect
~15GB peak memory. Requires a GPU pod — `flash_attention_2` has no CPU implementation, so
this step fails outright on a CPU-only machine.

### Every pod start

On a fresh container (`$HOME` is wiped; `/workspace` persists), after machine setup has
run:

```bash
cd /workspace/libero-eval-harness-workspace
source setup_env.sh
conda activate openvla
tmux attach -t vscode-tunnel   # device-code login, then Ctrl-b d to detach
```

---

## Troubleshooting

### `bddl` silently upgrades numpy

`robosuite==1.4.1` (pinned by `libero_requirements.txt`) pulls in `bddl` unpinned.
`bddl`'s own `setup.py` declares a bare `numpy` dependency with no version constraint,
so pip resolves to the latest available — breaking `torch` import (numpy 2.x's ABI is
incompatible with numpy-1.x-compiled wheels).

Fixed via a pip constraints file rather than editing the vendored requirements file
directly, so it applies to every future install in the env, not just this one.
`constraints.txt` pins `numpy==1.26.4` (this issue), `mujoco==2.3.7`, and
`tensorflow-metadata==1.14.0` (see below); `setup_env.sh` sets it as the `PIP_CONSTRAINT`
`openvla` env var.

If numpy has already drifted in an existing env: `pip install numpy==1.26.4 --force-reinstall`.

### `tensorflow-metadata` drifts past the pinned `protobuf`

Symptom: `ImportError: cannot import name 'runtime_version'`. If drifted, install in this
order:

```bash
pip install tensorflow-metadata==1.14.0 protobuf==4.25.9
```

Installing `tensorflow-metadata` alone downgrades `protobuf` to 3.20.3, which is too old for
tensorflow and wandb. A residual pip warning about `protobuf<4.21` is a stale upper bound,
not a real break.

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

`setup_env.sh` auto-detects via `nvidia-smi` and sets it as an `openvla` env var on every
restart, since this project moves between GPU and CPU-only pods and a value baked in once
would go stale.

`osmesa` requires system libraries not installable via pip. The symptom of them being
missing is:
```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```
Diagnose with `ldconfig -p | grep -i osmesa` (empty output confirms missing); fixed via
`apt-get install -y libosmesa6 libosmesa6-dev libglfw3 libglew2.2 libgl1` + `ldconfig` to
rebuild the shared-library cache (package names as on Ubuntu 22.04; older releases ship
`libglew2.0` and `libgl1-mesa-glx`). `setup_env.sh` handles this automatically, guarded to
only run when the libraries aren't already present.

### LIBERO's editable install fails with `ModuleNotFoundError: No module named 'libero'`

`libero/__init__.py` is missing from the LIBERO repo itself — confirmed via `git log
--all --oneline -- libero/__init__.py` returning no history, not something broken in a
specific clone. `setup.py` uses `find_packages()`, which requires `__init__.py` to
discover a package at all, so `pip install -e LIBERO` never registers `libero` even
though `pip show libero` reports it as installed.

Symptom: `pip show libero` looks fine, but `import libero` raises
`ModuleNotFoundError`.

Step 4 creates the file before installing. If the install was already done without it:
```bash
touch /workspace/LIBERO/libero/__init__.py
cd /workspace && pip install -e LIBERO --force-reinstall --no-deps
```

# libero-eval-harness-workspace

Language-conditioned VLA agent (OpenVLA + LoRA) evaluated on LIBERO-Long, with an eval
harness, failure taxonomy, and robustness study on top.

## Repo layout

```
.
├── scripts/              # smoke tests, eval harness, diagnostics, outputs archiving
├── provision.sh          # builds the env, sibling repos, and checkpoint — once per machine
├── setup_env.sh          # per-shell setup: rendering backend, env vars, VS Code tunnel
├── requirements-lock.txt # exact pip versions of the verified env
├── constraints.txt       # pip constraints applied via PIP_CONSTRAINT
├── .gitignore
└── README.md
```

`openvla` and `LIBERO` are cloned as sibling directories of this repo, outside it — they're
external dependencies with their own git history, not vendored in. `provision.sh` pins
both to fixed commits.

The policy under evaluation is `openvla/openvla-7b-finetuned-libero-10` everywhere: smoke
tests, prefetch, and the eval harness all use the same checkpoint.

## Setup

Pods are ephemeral: no network volume, so nothing survives teardown and any datacenter
works. Every pod builds the env, clones the siblings, and downloads the checkpoint onto
container disk. The one thing kept across pods is `outputs/`, copied to a RunPod global
volume (object storage) at the end of each eval run.

### Prerequisites

- **Machine setup:** [Miniforge](https://github.com/conda-forge/miniforge#install) in
  `~/miniforge3` with conda hooked into new shells, plus git credentials.
- **Pod template:**
  - Container disk **≥ 60GB**: about 8.5GB of env plus the 15GB checkpoint plus outputs.
  - A global volume attached, with `OUTPUTS_ARCHIVE_DIR` set to a directory on it.
    It's safe to leave unset, but then outputs aren't archived.
- **GPU:** RTX 4090 only, so numerics stay comparable across runs (see
  `docs/archive/plan_parallel_eval_harness.md`, Pod configuration). Filter listings on vCPU as well,
  about 32 for 8 env workers. Whatever the GPU, it must be pre-Blackwell (compute
  capability < 10): torch 2.2.0+cu121 has no kernels for sm_100/sm_120, and
  `provision.sh` checks this first and stops.

Keep git repos, the conda env, and the HF cache **off** the global volume. It's object
storage: it can't set permission bits, symlinks and file locks are unreliable, and git
fails on it.

### Local development

Everything except real OpenVLA inference runs on CPU: the stub policy, predicates, tests, and
LIBERO under `osmesa`. `provision.sh` installs the lock's `torch==2.2.0+cu121`, so don't run it
over an existing local CPU env; it would replace that torch build. Keep `~/.condarc` free of
`channels: [defaults]`, which overrides Miniforge's conda-forge default.

### Every pod

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/deep-rj/libero-eval-harness-workspace.git
cd libero-eval-harness-workspace
bash provision.sh
source setup_env.sh   # must be sourced, not run with `bash`, to affect the current shell
conda activate openvla
tmux attach -t vscode-tunnel   # select GitHub, complete the device-code login, Ctrl-b d
```

`provision.sh` is idempotent and skips work already done. It prints each step's wall time
and saves the summary under `outputs/provision_*.log`. It:

- clones `openvla` and `LIBERO` next to this repo at pinned commits, then creates
  `LIBERO/libero/__init__.py` (missing upstream; see Troubleshooting);
- creates the `openvla` env with `python=3.10.20`. conda-forge's latest 3.10 is newer, and
  the stack is verified on 3.10.20;
- installs `requirements-lock.txt` with `--no-deps`, then `openvla` and `LIBERO` as
  editable installs, also with `--no-deps`. The lock lists every transitive dependency,
  so nothing needs resolving. Resolving it would fail anyway on opencv's declared
  `numpy>=2` (see Troubleshooting);
- seeds `~/.libero/config.yaml`. The first `import libero` otherwise prompts for a
  dataset path;
- prefetches the checkpoint into the Hugging Face cache using `hf_transfer`;
- on a GPU, installs `flash-attn==2.5.5`. If pip finds no prebuilt wheel it compiles from
  source, which takes a long time and shows up in the timing summary.

`setup_env.sh` stores these as `openvla` env vars (`conda env config vars`), applied on
every `conda activate openvla`:

- a GPU-aware MuJoCo rendering backend;
- `PIP_CONSTRAINT` (see Troubleshooting);
- `HF_HUB_ENABLE_HF_TRANSFER=1`. hf_transfer's error messages are terse; if a download
  fails, retry with `HF_HUB_ENABLE_HF_TRANSFER=0` to see the real error.

It also installs system rendering libraries if they're missing, and spawns a VS Code
Tunnel session with the standalone CLI at `~/.local/bin/code`. The tunnel login is manual
on every pod.

The `SessionStart` hook that rewrites and pushes `STATUS.md` skips itself on RunPod pods
(`RUNPOD_POD_ID` is set), so it doesn't conflict with the same hook on another machine.

### Validate

```bash
python --version                  # Expected: Python 3.10.20
python scripts/libero_smoke_test.py
python -c "import torch, flash_attn, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, flash_attn.__version__)"
# Expected: 2.2.0+cu121 True 1.26.4 2.5.5
python scripts/smoke_test.py
```

- **`libero_smoke_test.py`** loads a LIBERO-10 task and runs a random-action rollout. It
  asserts that observation shapes/dtypes and the saved HDF5 output are well-formed. It
  works with both `osmesa` and `egl` rendering (see MuJoCo headless rendering under
  Troubleshooting). It's written as a jupytext percent-format notebook (`# %%` cell
  markers), so run it cell by cell in an interactive kernel when iterating.
- **`smoke_test.py`** loads the checkpoint, runs a forward pass, and prints the 7-dim
  action vector, timings, and peak memory (~15GB). It needs a GPU:
  `flash_attention_2` has no CPU implementation.

The checkpoint's `dataset_statistics.json` holds one normalization key, `libero_10`,
which is the `unnorm_key` the scripts use.

### Outputs

`scripts/libero_eval.py` writes raw traces (`.h5`), profiles, and the manifest CSV to
`outputs/` on local disk. At the end of the run it calls `scripts/archive_outputs.sh`,
which copies `outputs/` to a new snapshot `$OUTPUTS_ARCHIVE_DIR/<UTC time>-<git sha>-<pod id>/`
and verifies the `MANIFEST.sha256` it writes there. Snapshots are never overwritten. Run
`bash scripts/archive_outputs.sh [run_dir]` by hand before tearing down a pod if anything else
wrote there (the Stage A and baseline drivers don't call it). Files are written locally and
copied afterwards, rather than written straight to the volume, because object storage
handles HDF5's incremental writes and file locks, and appends to the manifest, badly.

#### Pulling outputs locally

A global volume has no S3 API and can't be read off-pod, so pulling needs a running pod
with the volume attached (the cheapest one will do; no GPU work). Start it with SSH over
exposed TCP (public IP), then from the laptop:

```bash
bash scripts/pull_outputs.sh -H <ip> -p <port> --list                # run dirs and sizes
bash scripts/pull_outputs.sh -H <ip> -p <port> <run> --dry-run       # preview
bash scripts/pull_outputs.sh -H <ip> -p <port> <run> --no-video --only '*.json' --only '*.h5'
```

Files land in `outputs/<run>/`; reruns skip what's already there. `PULL_HOST`, `PULL_PORT`
and `REMOTE_ROOT` (default `/workspace/libero-eval-harness-workspace/outputs`, the archive root on the volume) replace the flags. The pod needs `rsync`
(`apt-get install -y rsync`). Terminate the pod afterwards.

### Regenerating `requirements-lock.txt`

The lock is the verified GPU env's `pip freeze --exclude-editable`, with `flash-attn`
removed (`provision.sh` installs it separately, since it needs `--no-build-isolation`
and torch already present). Keep its `--extra-index-url` line: the `+cu121` torch wheels
exist only on PyTorch's index. After deliberately changing a dependency, regenerate the
lock from a pod where the smoke tests pass.

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
`tensorflow-metadata==1.14.0` (see below). `provision.sh` applies it to its own installs,
and `setup_env.sh` sets it as the `PIP_CONSTRAINT` `openvla` env var for later ones. The
lock already pins these versions; the constraints still guard ad-hoc installs.

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

### `PermissionError` from `transformers` under `/workspace/.cache/huggingface`

RunPod images set `HF_HOME=/workspace/.cache/huggingface/` and
`PIP_CACHE_DIR=/workspace/.cache/pip`. With a global volume mounted at `/workspace`,
both caches land on object storage, which rejects `chmod`. Hub downloads only warn
("Could not set the permissions"). Loading the checkpoint fails, because `transformers`
chmods the remote-code modules it copies into `$HF_HOME/modules`.

`setup_env.sh` and `provision.sh` set both caches under `~/.cache` instead. The
`openvla` env var overrides the image's value on activate. Check with `echo $HF_HOME`
inside the env.

### LIBERO's editable install fails with `ModuleNotFoundError: No module named 'libero'`

`provision.sh` handles this on fresh setups. It applies only to manual installs and envs
built before that fix.

`libero/__init__.py` is missing upstream, so `find_packages()` in LIBERO's `setup.py`
never registers `libero`. `pip show libero` looks fine, but `import libero` raises
`ModuleNotFoundError`.

Fix:
```bash
touch ../LIBERO/libero/__init__.py   # from this repo's root
pip install -e ../LIBERO --force-reinstall --no-deps
```

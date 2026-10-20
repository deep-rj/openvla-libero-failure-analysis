# Setup

Three levels, by what you want to run. Each level includes the requirements of the ones above it.

| Level | What runs | Needs |
|---|---|---|
| 1 | Recompute the detector result from committed files | Python ≥ 3.10, `numpy`, `h5py` |
| 2 | The full harness with a stub policy: tests, LIBERO rollouts, labels, detector pipeline | Linux, conda, OSMesa; no GPU |
| 3 | Real OpenVLA inference | Level 2 plus an NVIDIA GPU (see below) |

## 1. Recompute the detector result

```bash
pip install numpy==1.26.4 h5py
python scripts/run_detector.py metrics \
  --scores docs/results/heldout/detector/scores.csv \
  --frozen docs/results/baseline/detector/frozen_detector.json --out /tmp/detector
```

`/tmp/detector/detector_eval.json` matches `docs/results/heldout/detector/detector_eval.json`
byte for byte. No simulator, model or sibling repo is involved.

## 2. The harness without a GPU

Requirements:

- Debian or Ubuntu Linux. Used on Ubuntu 24.04 (including WSL2) and on Ubuntu-based cloud
  containers.
- [Miniforge](https://github.com/conda-forge/miniforge#install) (or another conda) in
  `~/miniforge3` or `~/miniconda3`, or on `PATH` via `CONDA_EXE`. Keep `~/.condarc` free of
  `channels: [defaults]`, which overrides conda-forge.
- System rendering libraries for MuJoCo under OSMesa. `setup_env.sh` installs them with
  `apt-get`, which assumes root (as in a container); otherwise install the packages it lists
  yourself.
- About 10 GB of disk for the env and the sibling repos.

```bash
git clone https://github.com/deep-rj/openvla-libero-failure-analysis.git
cd openvla-libero-failure-analysis
bash provision.sh
source setup_env.sh   # must be sourced, not run with `bash`
conda activate openvla

python -m pytest -m "not env"          # unit tests on synthetic trajectories
python -m pytest                       # adds tests that build real LIBERO envs (slower)
python scripts/libero_smoke_test.py    # LIBERO env load and a random-action rollout
python scripts/run_eval.py --policy stub --tasks 0:1 --init-states 0:2 --seed 0
```

`openvla` and `LIBERO` are cloned next to this repo, not inside it, at the commits pinned in
`provision.sh`. The stub policy emits seeded random actions in OpenVLA's output format, so every
stage after the policy (traces, world extraction, labels, detector) runs unchanged.

`provision.sh` installs the lock's CUDA build of torch, which also runs on CPU. If you already
have an env with a CPU torch build (`torch==2.2.0+cpu`), don't run `provision.sh` over it: it
would replace that build.

## 3. Real OpenVLA inference

Requirements, on top of level 2:

- **An NVIDIA GPU with at least 16 GB of memory.** Peak use is about 15 GB (the reported runs
  used 24 GB cards), and `flash_attention_2` has no CPU path.
- **Compute capability below 10 (pre-Blackwell).** torch 2.2.0+cu121 has no kernels for
  sm_100/sm_120; `provision.sh` checks this first and stops.
- **About 30 GB more disk:** the 15 GB checkpoint plus outputs.

On a GPU machine, `provision.sh` also prefetches `openvla/openvla-7b-finetuned-libero-10` and
installs `flash-attn==2.5.5`. Every reported result used an RTX 4090. Other GPU models give
slightly different bf16 numerics, so individual episodes can differ.

### Validate the GPU setup

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


## What the setup scripts do

`provision.sh` is idempotent and skips work already done. It prints each step's wall time and
saves the summary under `outputs/provision_*.log`. It:

- clones `openvla` and `LIBERO` next to this repo at pinned commits, then creates
  `LIBERO/libero/__init__.py` (missing upstream; see Troubleshooting);
- creates the `openvla` env with `python=3.10.20`. conda-forge's latest 3.10 is newer, and the
  stack is verified on 3.10.20;
- installs `requirements-lock.txt` with `--no-deps`, then `openvla` and `LIBERO` as editable
  installs, also with `--no-deps`. The lock lists every transitive dependency, so nothing needs
  resolving. Resolving it would fail anyway on opencv's declared `numpy>=2` (see
  Troubleshooting);
- seeds `~/.libero/config.yaml`. The first `import libero` otherwise prompts for a dataset path;
- on a GPU, prefetches the checkpoint into the Hugging Face cache using `hf_transfer`, and
  installs `flash-attn==2.5.5`. If pip finds no prebuilt flash-attn wheel it compiles from
  source, which takes a long time and shows up in the timing summary.

`setup_env.sh` installs missing rendering libraries and stores these as `openvla` env vars
(`conda env config vars`), applied on every `conda activate openvla`:

- a MuJoCo rendering backend: `egl` with a GPU, `osmesa` without;
- `PIP_CONSTRAINT` (see Troubleshooting);
- `HF_HOME` and `PIP_CACHE_DIR` under `~/.cache`;
- `HF_HUB_ENABLE_HF_TRANSFER=1`. hf_transfer's error messages are terse; if a download fails,
  retry with `HF_HUB_ENABLE_HF_TRANSFER=0` to see the real error.

With `SETUP_VSCODE_TUNNEL=1` it also starts a VS Code tunnel in a tmux session
(`tmux attach -t vscode-tunnel` to complete the device-code login), for editing on a remote
machine.

## Outputs

Runs write traces (`.h5`), videos, the manifest and a summary under `outputs/` on local disk;
`outputs/` is gitignored, and committed results are summaries only (`docs/results/`).

On machines that are torn down after use, `bash scripts/archive_outputs.sh [run_dir]` copies a
run to an immutable snapshot `$OUTPUTS_ARCHIVE_DIR/<UTC time>-<git sha>-<host>/` and verifies a
`MANIFEST.sha256` there; it does nothing when `OUTPUTS_ARCHIVE_DIR` is unset. Runs are written
locally and copied afterwards because network and object-storage volumes handle HDF5's
incremental writes, file locks and manifest appends badly. `scripts/pull_outputs.sh` fetches
snapshots back with rsync over SSH:

```bash
bash scripts/pull_outputs.sh -H <host> -p <port> -r <archive dir> --list
bash scripts/pull_outputs.sh -H <host> -p <port> -r <archive dir> <run> --no-video --only '*.h5'
```

Sharded runs (one command per machine, then one local merge) are described in
`docs/pod_runbook.md`.

## Reference setup for the reported results

All GPU work ran on ephemeral cloud containers (RunPod) with an RTX 4090 and no persistent disk.
Each container ran `provision.sh` from scratch. `OUTPUTS_ARCHIVE_DIR` pointed at a network volume
mounted at `/workspace`, which is object storage: it can't set permission bits, symlinks and file
locks are unreliable, and git fails on it, so repos, the env and the Hugging Face cache stay on
local disk. Pulling from that volume needs a running container with it attached and SSH over an
exposed TCP port. Development happened on a CPU-only WSL2 laptop at level 2.

## Regenerating `requirements-lock.txt`

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

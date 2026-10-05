# Block 1 — pod checklist

Everything a coding agent must implement, run or validate on the pod.
`docs/archive/block1_spec.md` is the source of truth for the tests; the schema is in
`docs/block1_schema.md` and the labeling and blinding rules in `docs/labeling_protocol.md`.
This file only lists
what needs a GPU, the real checkpoint, or the pod environment. Findings behind these
items are in `docs/block1_findings.md`.

## Before the pod

- [ ] Working tree clean and committed; the recorded git SHA must not be dirty.
- [ ] `pytest -m "not env"` passes (fast), then `pytest -m env` (slow, builds real envs; run
      it alone, the laptop has ~8 GB RAM). Unit tests cover: decode path, schema, wait-step alignment, hash
      comparison, static restore (T4), obs-vs-`qpos` cross-check. All run on the stub
      and osmesa, no GPU.
- [ ] `StageTimer` is off for any run whose timing is reported.

## Pod setup (RTX 4090 only)

- [ ] `bash provision.sh && source setup_env.sh && conda activate openvla`; the
      VS Code Tunnel login is manual.
- [ ] `python --version` is 3.10.20 and `torch.cuda.is_available()` is True.

## Load-time guards (first thing the run does; hard-fail on any mismatch)

Expected sha256 of the hub files at revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`
(computed from downloaded copies; the pod's executed files must reproduce them):

| File | sha256 |
|---|---|
| `modeling_prismatic.py` | `9ce241c5ca09a4bed73654d0ca509baaff0fc5887d0bdd62e360ca8254f7794a` |
| `configuration_prismatic.py` | `68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68` |
| `processing_prismatic.py` | `06563586bbc73d36501a14a231a3174904ede830d33174d0c625ece8e931ef1e` |

- [ ] Hash the files behind `type(vla)`, `type(vla.config)` and `type(processor)`;
      `type(vla).__module__` starts with `transformers_modules`.
- [ ] `vla.norm_stats[unnorm_key]` (q01, q99, mask) equals `config.json` `norm_stats`.
- [ ] `lm_head.out_features == 32064`; generation config has no non-default processors.
- [ ] On one real step: `out.scores` and `out.logits` are bitwise equal, and
      `argmax(logits)` equals the generated token at every position.
- [ ] Record the resolved hub commits (code repo, checkpoint), GPU type and library
      versions in episode metadata.

## Stage A (unattended)

```bash
python scripts/run_stage_a.py --out outputs/stage_a --policy openvla --parity
```

Loads the checkpoint once and reuses it for every run. `--parity` adds a run that computes
both the logged and the vendored action at every step (twice the inference cost) and
lists any policy step where they differ. Writes `stage_a_report.md` and `.json`.

Tasks 0, 7, 3, 9 with 2 episodes each, plus one forced-truncation episode
(`EVAL_MAX_STEPS` ≈ 60). Order: T1, T2, T3, T5, then the cost report.

- [ ] T1: two logger-off episodes, same seed. Report bitwise equality; on divergence
      give the first divergent step and the top-2 margin from the logger-on run, then stop.
- [ ] T2: logger-on vs logger-off actions, bitwise (or within the T1 tolerance).
- [ ] T3, every step: `chosen_token_id == argmax_full_id`; `clipped_flag` rule;
      decode matches vendored `predict_action`; `executed_action` equals what
      `env.step` received.
- [ ] T5: first non-wait `step_idx` equals the wait count (10); model fields are
      sentinel on wait steps; the forced-truncation episode ends without `done`.
- [ ] Every episode file is written atomically; fixture tasks 3 and 9 have non-empty
      `fixture_joint_qpos`, and the joints move in the logged trace.
- [ ] Videos exist for every episode, frame count equals step count, overlay shows
      `step_idx` only.
- [ ] Cost report: measured s/step (un-instrumented), projected GPU-hours and dollars
      for the 250-episode baseline, bytes per episode (flag if far from ~2 MB).
- [ ] `bash scripts/archive_outputs.sh`; confirm the files are on the global volume
      before teardown. Push the report and any summary CSV; raw traces stay gitignored.

T4 (static restore and the obs-vs-`qpos` check) runs locally on the pulled Stage A files:
`python scripts/check_restore.py <run>/logged/episodes/<id>.h5` (try one fixture episode, task 3 or 9).

## Baseline (fresh pod, only after a go)

```bash
python scripts/run_eval.py --policy openvla --tasks 0:10 --init-states 0:25 --out outputs/baseline
```

Resumable: rerunning skips episodes already in `manifest.csv`. Refuses to run from a dirty
tree unless `--allow-dirty` is passed.

- [ ] 10 tasks × init states 0–24, logger and video on, one pass.
- [ ] `episode_id` deterministic, so sharded pods cannot collide; per-pod manifest.
- [ ] Archive to the global volume; push per-episode summary CSVs.

### Sharded baseline (several pods)

Shard by init state so every pod runs a mix of tasks. Five chunks of 50 episodes (~100 min
each); a pod can take several by widening the range (`--init-states 0:10`). Deploy pods as GPUs
appear, give each an unclaimed chunk, and keep the ledger below current.

| chunk | `--init-states` | pod | archived | pulled |
|---|---|---|---|---|
| 1 | `0:5` | | | |
| 2 | `5:10` | | | |
| 3 | `10:15` | | | |
| 4 | `15:20` | | | |
| 5 | `20:25` | | | |

Per pod (same commit on every pod; confirm `git rev-parse --short HEAD` matches):

```bash
python scripts/run_eval.py --policy openvla --tasks 0:10 --init-states 0:5 --out outputs/baseline_i00-04
bash scripts/archive_outputs.sh outputs/baseline_i00-04
```

Locally, after pulling each snapshot:

```bash
python scripts/merge_runs.py <shard>... --out outputs/baseline --tasks 0:10 --init-states 0:25
```

The merge refuses duplicate episodes, shards with different run settings, and shards from
different commits (`--allow-mixed-commits` overrides the last). It lists any missing
(task, init state) pairs, which become the next chunk. Sources are only read; the merged
directory records each shard's environment in `merged_from.json`.

Cross-pod determinism, free from the baseline: the shard covering init states 0 and 1 reruns
the eight Stage A episodes on a different host.

```bash
python scripts/compare_runs.py <stage_a>/logged outputs/baseline
```

Report the result either way: bitwise agreement, or the first divergent action, logit and state
step per episode. A divergence changes how reproducibility is worded; it does not invalidate
the baseline as a statistical result.

Seed-1 sample (queue on any free pod; not a gate): ten episodes, init state 0 of every task.

```bash
python scripts/run_eval.py --policy openvla --seed 1 --tasks 0:10 --init-states 0:1 --out outputs/seed1_i00
bash scripts/archive_outputs.sh outputs/seed1_i00
python scripts/compare_runs.py outputs/baseline outputs/seed1_i00   # matched by task and init state
```

Outcome flips and first-divergence steps against seed 0 give the seed-to-seed noise floor
(`docs/block1_findings.md`, "The env seed is not inert").

After the baseline (all local, on the pulled or archived run directory):

```bash
python scripts/summarize_run.py <run>                 # success rate with n and Wilson interval, s/episode
python scripts/pilot_report.py <run> --stage-a <stage_a>/stage_a_report.json   # pooled stats only
python scripts/select_labeling.py <run> --n 20 --seed 0  # blind draw + empty label CSV
```

Blinding: the pilot report is pooled statistics only and is safe to produce first. Do not plot
per-episode logits, entropy or confidence until `labeling/labels_template.csv` has been filled
in and committed.

## Do not, on the pod

- Add schema fields, or run the baseline without a go.
- Plot or summarize model outputs per episode (blinding); the pilot report is pooled only.
- Patch the `openvla` or `LIBERO` siblings.

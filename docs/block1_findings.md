# Block 1 findings — §2 verifications

Environment: transformers 4.40.1, torch 2.2.0, LIBERO and openvla siblings at the
pinned commits. Checked on the laptop (CPU, no checkpoint weights). Items marked
**pod** need the real checkpoint and are confirmed by Stage A.

## 2.1 Raw logits vs processed scores

- `generate(..., output_logits=True, return_dict_in_generate=True)` exists in 4.40.1;
  `out.logits` are the raw pre-processor logits, `out.scores` the processed ones.
- Tiny random Llama (vocab 32064, bf16, greedy, 7 new tokens): `scores` and `logits`
  are bitwise equal, both fp32, and argmax equals the generated token at every position.
- The checkpoint snapshot has no `generation_config.json`, so `generate` uses the default
  config: no logits processors under greedy decoding. **pod**: confirm on the real model.
- `LlamaForCausalLM.forward` computes `lm_head(hidden).float()`, so logits are fp32
  holding bf16-exact values.
- fp16 storage is lossless for bf16 values with |x| ≥ 2^-14 (≈ 6.1e-5). Below that,
  values lose low mantissa bits (observed: 7 of 224,448 elements, all |x| < 5e-6). The
  error is far below anything that changes a softmax or an argmax.
- Executed code path: `config.json` `auto_map` points at the hub's
  `openvla/openvla-7b` remote code, and with `trust_remote_code=True` transformers
  prefers it over the locally registered classes. The hub's `modeling_prismatic.py` and
  `configuration_prismatic.py` (revision `47a0ec7f`) are byte-identical to the sibling
  repo's. The hub revision is unpinned.
- Verified without weights: after the same Auto-class registrations as `get_vla`,
  `AutoConfig.from_pretrained(..., trust_remote_code=True)` returns the hub's
  `OpenVLAConfig` (`transformers_modules.openvla.openvla-7b.47a0ec7f...`), so
  `has_remote_code = True` and `has_local_code = False`, and `from_pretrained` takes the
  remote-code branch. The loaded model class is the hub's, not the sibling's.

## 2.2 Slice indices

- `config.json`: `text_config.vocab_size = 32064`, `pad_to_multiple_of = 64`, so
  `vocab_size` in `predict_action` is 32000. **pod**: confirm `lm_head.out_features`.
- Raw bin `32000 - id - 1` in [0, 255] ⇔ id in [31744, 31999]; slice index `j` is token
  `31744 + j`, bin `255 - j` before clipping.
- `bins = linspace(-1, 1, 256)`, `bin_centers` are the 255 midpoints, clip upper bound 254.

## 2.3 Unnormalization mask

- `libero_10` norm stats: `mask = [T, T, T, T, T, T, F]`. Dims 0–5 use q01/q99; the
  gripper is not unnormalized (its q01/q99 are 0/1 and unused).
- The gripper's decoded value stays a bin center. Training labels were {0, 1}, so the
  values are ≈ {0.0, 0.996} (bin 127 center is exactly 0; bin 254 center is 254/255).
  `normalize_gripper_action` maps `a → 2a - 1` and takes `sign`, so the threshold is at
  a = 0.5. No bin center equals 0.5, so `sign(0)` cannot occur.

## 2.4 Object state access

- LIBERO exposes `{name}_pos` and `{name}_quat` in obs for every entry of
  `objects_dict`. Quaternions are converted to **xyzw** from MuJoCo's wxyz
  (`bddl_base_domain.py`), so `object_poses` quaternions are xyzw as exposed.
  `robot0_eef_quat` is also xyzw (observed ≈ [1, 0, 0, 0] for a downward gripper).
- Objects per task (n_obj): task 0: 8, 1: 8, 2: 2, 3: 2, 4: 5, 5: 2, 6: 4, 7: 5, 8: 2, 9: 2.
  This includes distractors.
- **Fixtures have no pose obs.** `fixtures_dict` is non-empty for tasks 2 (stove),
  3 (cabinet, wine rack), 5 (caddy), 8 (stove), 9 (microwave); none appear in obs. Body
  pose would not capture their articulation (stove on/off, drawer, microwave door)
  anyway. Tasks 2, 3 and 9 have goals that depend on that state.
- `sim_state` (`env.get_sim_state()`, flattened MuJoCo state) is 84 floats for task 7 and
  123 for task 0, so ≈ 0.7–1.0 KB/step.

## 2.5 LIBERO eval constants

- Wait steps: `num_steps_wait = 10` (openvla `GenerateConfig`); our harness `NUM_STEPS_WAIT = 10`.
- Max steps for libero_10: 520 (both). The loop runs to `max_steps + num_steps_wait`.
- Control frequency: `control_freq = 20` (observed on the env instances).
- Gripper post-processing before `env.step`: `normalize_gripper_action(binarize=True)`
  (`2a - 1`, then `sign`), then `invert_gripper_action` (× -1). Wait steps send
  `[0, 0, 0, 0, 0, 0, -1]`.

## Open items

- Fixture articulation state is not covered by `object_poses`. Needs a decision before
  the schema is final.
- Real-checkpoint confirmations (**pod**): no default logits processors, `lm_head`
  width, executed remote-code class.

## Observation timing (found while testing the world-state reader)

- robosuite obs (EEF pose, `{name}_pos`/`{name}_quat`) come from kinematics computed at the
  start of the last physics substep, so they lag `qpos` by one substep. The model timestep
  is 0.002 s and a 20 Hz control step is 25 substeps. After `sim.forward()` on the same env,
  the recomputed EEF site position differs from the obs by about 0.4–0.5 mm on a moving arm,
  matching the difference seen after restoring the state in a fresh env.
- `sim_state` and `fixture_joint_qpos` are read from `qpos` and are current; the obs-derived
  poses are up to one substep older. Restoring a state and running `forward()` therefore
  reproduces `sim_state` and joint positions exactly, and obs-derived poses to within
  v·2 ms (T4 uses 5 mm).
- `sim.forward()` also renormalizes free-joint quaternions in the state (change of 2e-16),
  so the logging path must not call it; T2 compares actions bitwise.
- Free-joint `qpos` stores quaternions as wxyz; obs expose xyzw.

## Stage A results (RTX 4090, 2026-10-02)

Run: `scripts/run_stage_a.py --policy openvla --parity` at harness `2326ab0`, clean tree.

- **Checks.** T1 (two logger-off episodes) and T2 (logger on vs off) are bitwise identical.
  T3, T5 and the 30-step parity run against the vendored `predict_action` report no
  violations. The forced-truncation episode ended at 60 steps without success. The
  load-time guards (executed hub-file hashes, norm stats, `lm_head` width, generation
  config) passed on the real model, so the pinned hashes are correct.
- **Cost.** 0.276 s per policy step, 444 policy steps per episode (8 episodes), 3.2 MB per
  episode including video. The 250-episode baseline projects to 8.5 GPU-hours, about $6.4
  at $0.75/h. The `generate` call alone has a median of 239 ms (p90 240 ms).
- **Real episode.** Task 0, init state 0: success in 279 policy steps (289 steps with the
  10 wait steps), the same as the 2026-09-25 profiling run. The metadata records the
  checkpoint commit `80970322…`, hub code commit `47a0ec7f…`, the hub model class module and
  the 4090. The video has one frame per step at 20 fps, upright, with a step-index overlay
  that reads `WAIT` on settle steps. T4 (static restore) passed at a mid-episode step and at
  a wait step.
- **Token 31744 is used, not untrained.** It is the token for values at the upper edge:
  the gripper-open value (decoded 0.996) accounts for 141 of 279 gripper steps, and z for 16
  steps. Token 31745 received no mass on the gripper. Both decode to bin 254, so entropy
  merges them. `clipped_flag` is true for 31744 by definition, so about 8% of dimension-steps
  in this episode are flagged without being anomalous; the pilot report's
  `chosen_edge_fraction` is expected to be about 0.5 for the gripper.
- **Fixture task on a real episode.** Task 3, init state 0 (cabinet, three slide joints):
  520 policy steps without success. `fixture_joint_qpos` has shape (530, 3) and records the
  articulation: the bottom drawer goes from -0.146 to -0.030 (0.119 of travel) and the middle
  drawer moves 0.002, while the top drawer stays at 0. Static fixture poses are in the
  metadata. The video has 530 frames for 530 steps. T4 passed at the default mid-episode step
  and at step 260, where the drawer joint is mid-travel. Task 9 was not pulled.
- **Host.** The pod's main thread was pinned at 100% CPU while the GPU sat at about 60%
  utilization (a 10.2-CPU cgroup quota on an AMD EPYC 7K62). Throttling counters were not
  collected; see STATUS.md.

## The env seed is not inert (laptop replay, 2026-10-05)

`env.seed(s)` calls `np.random.seed(s)` (LIBERO `bddl_base_domain.py`). The rollout resets the
env and then calls `set_init_state`, so the reset-time randomness was assumed to be
overwritten. Replaying fixed actions in a fresh env per seed (CPU, osmesa) shows otherwise.

- **Same seed reproduces.** Repeated seed-0 and seed-1 envs built in different orders give
  bitwise-identical `sim_state` trajectories. This matches T1, which tested only same-seed
  reproducibility and says nothing about changing the seed.
- **Different seeds differ from step 0.** With 30 random actions on task 0, about 100 of 123
  `sim_state` columns differ by up to about 6e-14 after the first step.
- **The difference amplifies.** Replaying the Stage A actions open-loop (256-pixel cameras),
  task 0 init 0 ends with state differences of order 1 for seed 1 versus 0, but about 4e-5
  for seed 2; task 3 init 0 reaches order 1 for both seeds. Frames differ in 200–211 of 290
  frames on task 0 and in every one of 531 frames on task 3, including the first frame.
- **Not established.** The channel through which the seed enters is unidentified. Open-loop
  replay exaggerates divergence, and the closed-loop effect on success rate is unmeasured.

Consequence: the baseline is pinned to seed 0 and is reproducible, but its spread covers
sampling over init states only, not seed-to-seed variation. A seed-1 rerun of the baseline
episodes, counting outcome flips, would measure that noise floor; any Block 5 effect smaller
than it cannot be attributed to the perturbation. Not run.

## Baseline pod hosts (2026-10-05)

Five RTX 4090 pods from the same template ran the sharded baseline at harness `adbd165`. Pods
are ephemeral, so a later pod may land on any of these host types. The run metadata records
only the GPU name; the rest below was read off each pod by hand (`lscpu`, the cgroup quota,
`nvidia-smi`). Chunk-to-pod mapping assumes the pods were listed in chunk order.

| Chunk | CPU | CPU quota | Driver | s/step |
|---|---|---|---|---|
| i00-04 | AMD EPYC 7K62 (48-core) | 10.2 CPUs | 570.195.03 | pending |
| i05-09 | AMD EPYC 7K62 (48-core) | 10.2 CPUs | 580.159.04 | pending |
| i10-14 | AMD EPYC 7642 (48-core) | 10.2 CPUs | 595.91.07 | pending |
| i15-19 | AMD EPYC 75F3 (32-core) | 13.6 CPUs | 570.195.03 | pending |
| i20-24 | AMD EPYC 7402 (24-core) | 7.65 CPUs | 595.91.07 | pending |

The 7.65 figure assumes the default 100 ms cgroup period; that pod's quota was read from the
v1 file, which holds the quota only. The Stage A pod was an EPYC 7K62 with a 10.2-CPU quota; its
driver version was not recorded.

**Speed and throttling (partial, mid-run).** On matched 520-step episodes the Zen 3 pod (EPYC
75F3, 13.6 CPUs) ran at 0.245–0.255 s/step, against 0.278–0.298 on the Zen 2 pods (7.65 and
10.2 CPUs) and 0.273–0.274 on Stage A. The 7.65-CPU pod was not slower than the 10.2-CPU pods.
The cgroup counters read 60 s apart agree: 11 of 600 periods throttled on a 10.2-CPU pod
(1.8%; 1.7% since start) and 4 of 600 on the 7.65-CPU pod (0.7%; 0.5% since start). Throttling
is rare, so the quota is not what limits the step time; per-core speed is the likelier limit.
n is small, the pods differ in CPU generation and quota together, and all samples are from
tasks 0–2.

**Pod 4 lost its GPU mid-run.** The chunk-4 pod (EPYC 75F3, driver 570.195.03) stopped writing
episodes at 00:33:34 UTC, 41 episodes into its 50. `nvidia-smi` reported "Failed to initialize
NVML: Unknown Error", a fresh `torch.cuda.is_available()` returned False, and the run process
stayed alive without a traceback until it was killed, about 27 minutes later. The cause could not
be determined from inside the container. The 41 completed episodes were archived (pod id
`mpvz0f3nbm11b8`); the nine missing ones (task 8 init states 16–19, task 9 init states 15–19)
are rerun as separate shards. The job had no stall watchdog.

## Baseline results (2026-10-05)

250 episodes (10 tasks × init states 0–24), seed 0, logger and video on, harness `adbd165`, five
RTX 4090 pods. The merged run is `outputs/baseline` (gitignored): four 50-episode chunks, pod 4's
partial chunk (41 episodes) and two rerun shards (4 + 5 episodes). `merge_runs.py` reported 0 of
250 missing.

- **Success rate:** 130/250 = 52.0%, Wilson 95% interval 45.8% to 58.1%. Terminated by success
  130, by the step cap 120. The interval covers sampling over init states only (seed pinned).
- **Throughput:** 110.3 s per episode, 0.277 s per policy step (the Stage A value), 399 mean policy
  steps. About 7.7 GPU-hours for the 250 episodes, against 8.5 projected from Stage A's 444 steps.
- **Per task (n = 25 each):**

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Successes | 13 | 19 | 14 | 10 | 8 | 18 | 15 | 10 | 8 | 15 |

- **Pooled pilot report** (99,634 non-wait steps; `scripts/pilot_report.py`): argmax on a text or
  padding token is 0.0000 in every dimension and out-of-slice mass peaks at 0.0037 (dim 4). The
  gripper's exp(H) is at most 1.05 in 92% of steps (q95 1.21) and it chose token 31744 in 57%.
  Dims 0–2 have a median exp(H) of 1.3–1.7 and q95 of 4.4–6.3. Top-2 margins of at most 1 ulp
  occur in 5.8–7.7% of steps on dims 0–2 and 2.7–4.1% on dims 3–5; 51–58% of steps on dims 0–2
  are within 16 ulps. On dim 0 at the 1-ulp cutoff, 38% of near ties are 10 or more bins apart.
- **Cross-host determinism.** Stage A against the baseline shard that reran its eight episodes:
  8 of 8 identical in actions, logits and simulator state. Stage A against a rerun of four of them
  on an EPYC 7402 pod (different driver): 4 of 4 identical. Every CPU in that comparison is Zen 2;
  the one Zen 3 pod (EPYC 75F3) lost its GPU, and its 41 episodes have not been checked against
  another generation.
- **Seed 1 against seed 0** (init state 0 of every task; one pod, EPYC 7642):

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Outcome, seed 0 → seed 1 | T→T | T→T | T→F | F→T | F→F | T→F | F→F | F→T | T→F | T→T |
| First logit divergence (step) | 79 | 76 | 10 | 10 | none | 10 | none | 60 | 10 | 10 |

  Five of ten outcomes flipped and eight of ten trajectories diverged; tasks 4 and 6 are bitwise
  identical across seeds over their full 530 steps. Net successes were 6 and 5. The first
  simulator-state difference appears at step 1, 11 or 16, and the logits differ from the first
  policy step (step 10) in four episodes. n = 10, so a 5/10 flip rate has a wide interval; the
  entry channel of the seed is still unidentified.

## Pilot report gate (2026-10-06)

**What the gate controls.** It gates the label draw. The labels feed both the taxonomy and the
detector, so they are needed whether or not the detector premise holds; that premise is tested
later by the separability gate, on an outcome split that blinding forbids before the labels are
committed. This gate can therefore only test necessary conditions, and it is an
instrument-validity check: no-go requires an internal inconsistency or an impossible value.
A surprising but consistent reading is a finding, not a reason to stop.

**Error costs.** Labels come from video and world state, not from logged model outputs.
- False go, derived-metric bug: fix and recompute from the logged HDF5; no GPU, no labels lost.
- False go, capture bug: rerun the baseline (~7.7 GPU-h, ~$6, a few hours across pods);
  labels survive because T2 shows logging does not change actions.
- False go, action-path bug: the baseline is wrong and the ~5–6 h of labels are lost.
- False no-go: about a day chasing a non-bug, plus the risk of altering a correct metric until
  it looks plausible. The bar for go is low.

Kill conditions were written after the pilot report had been read, not before.

### Kill conditions (`scripts/instrument_checks.py outputs/baseline`, 250 episodes, 99,634 steps)

| # | Premise | Kill condition | Result |
|---|---|---|---|
| 1 | The chosen token is the decoded token | T3: `chosen ≠ argmax_full`, decode mismatch, or executed ≠ sent | 0 violations, all 250 episodes (Stage A covered 9) |
| 2 | The slice holds the probability mass | `lse_slice − lse_full > 1e-5` | 0; max excess 3.7e-6 |
| 3 | Logits are the model's bf16 values, stored losslessly as fp16 | Any logged slice logit not exactly bf16 (fp16 subnormals excluded) | 0 of ~178M logits |
| 4 | Entropy is computed correctly | exp(H) outside [1, 255] | 0; max 30.6 (dim 0), 2.32 (gripper) |
| 5 | Entropy and margins come from one distribution | exp(H) < 1 + e^(−δ) for a bin margin of δ nats | 0 |
| 6 | The edge-bin merge is correct | Token margin ≠ bin margin where both edge tokens are > ln 2 below the second token | 0; covers 98.7–99.3% of steps on dims 0–5, 5.8% on the gripper |
| 7 | Determinism and logger non-interference (T1, T2) | Any differing action | Stage A only; Zen 2 hosts only |
| 8 | Wait-step alignment (T5) | First non-wait step ≠ 10, or model output on a wait step | 0 violations, all 250 episodes |

Tolerances (1e-5 absolute on lse, 1e-5 relative on exp(H)) sit above fp32/fp64 summation error
and below any physical effect.

### Findings from the checks

- **The out-of-slice clamp hides only rounding.** `pilot_report.py` clamps
  `lse_slice − lse_full` at 0 for the mass, so an impossible positive value reads as zero mass;
  the report now counts steps above 1e-5 in a separate column. The difference is positive on
  51–69% of steps per dimension, at most 3.7e-6: float summation order, not capture. Mass below
  ~5e-6 (including the p95 of ~1.6e-6) is at this rounding floor.
- **bf16-valued logits are verified, not assumed.** Every stored logit is bf16-exact, so the
  margins in ulps measure the model's own output grid.
- **The gripper has exactly two modes.** Chosen bins are 254 (57.3%, open) and 127 (42.7%,
  close); exp(H) never exceeds 2.32. Every gripper near tie is open versus close.
- **No action is the modal action on every motion dimension.** The bin containing raw action 0
  takes 33–43% of steps on dims 0–2 and 47–68% on dims 3–5.
- **No hard lattice in chosen bins.** All 255 bins are used on dims 0–2, 4 and 5 (249 on dim 3).
  Bin-to-bin counts deviate from their neighbours' mean by a median 27–45%, several times
  independent-sampling noise, but consecutive steps are correlated, so that comparison
  overstates it. Whether a soft comb explains the 2–9-bin near ties is unresolved.
- **Far near ties are often stop-versus-move.** Among ≤1-ulp ties at ≥10 bins apart, the raw-0
  bin is one of the top two in 24–37% on dims 0–2 and 44–48% on dims 3–5.
- **Bin width in env units** ((q99 − q01)/255, times the OSC scale of 0.05 m or 0.5 rad per
  unit; setpoint per 50 ms step): 0.28, 0.32, 0.33 mm on dims 0–2; 0.46, 0.60, 1.2 mrad on
  dims 3–5. Ten bins is ≈3 mm per step on dim 0.

### Predictions for the separability gate

Each statistic is per episode, failed (n = 120) against successful (n = 130), tested by the
AUROC 95% CI.

- **P1. Gripper near ties at contact separate.** Gripper steps with bin margin ≤ 4 ulps while
  the EEF is within 5 cm of the task's target object, per step in that window, are higher in
  failures. Thresholds fixed 2026-10-06, before any outcome split. Falsified if the CI includes 0.5. Requires time-aligned model output, so it runs
  only after the label CSV is committed. At ≤ 1 ulp there are ~2 gripper ties per episode,
  too few for power; ≤ 4 ulps gives ~6.4.
- **P2. Rotation is concentrated on no rotation.** Recorded as an observation, not a test: its
  draft 50% threshold was not fixed before the check ran. Observed 68.1%, 54.7% and 47.1% on dims
  3–5, against 32.8–43.4% on translation, so the concentration is strongest on rotation but not
  specific to it.
- **P3. Rotation entropy does not separate.** Mean exp(H) over dims 3–5. Falsified if the CI
  excludes 0.5.
- **P4. Gripper saturation does not separate on average.** Fraction of gripper steps with
  exp(H) > 1.05. Falsified if the CI excludes 0.5.

Parked: out-of-slice emission as a failure signature is absent under clean conditions (argmax
never leaves the slice); whether it appears under perturbation is a Block 3 question. A
physical-unit distance class may be added as an exploratory cut beside the pre-registered
adjacent / 2–9 / ≥10 classes, never in place of them.

**Decision: go** (2026-10-06). Every kill condition passes. Row 7 rests on Stage A and on Zen 2
hosts only.

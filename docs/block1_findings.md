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
- **Not yet checked on a real episode.** The fixture-task paths (`fixture_joint_qpos` on tasks
  3 and 9, and T4 with articulated joints) were exercised only with the stub on CPU.
- **Host.** The pod's main thread was pinned at 100% CPU while the GPU sat at about 60%
  utilization (a 10.2-CPU cgroup quota on an AMD EPYC 7K62). Throttling counters were not
  collected; see STATUS.md.

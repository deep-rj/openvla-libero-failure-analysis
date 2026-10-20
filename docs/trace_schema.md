# Episode trace schema (version 1)

One HDF5 file per episode, written by `harness.schema.write_episode` and read with
`harness.schema.read_episode`. Constant per-episode facts are in the file's `metadata`
attribute (JSON); the per-step columns below are stacked datasets indexed by `step_idx`.
Writes are atomic: a temporary file is renamed only after the whole episode is written.

## Run layout

```
<run>/run_config.json        config + environment (git SHAs, library versions, GPU)
<run>/manifest.csv           one row per finished episode
<run>/episodes/<id>.h5       trace (logged and parity modes)
<run>/videos/<id>.mp4        labeling video
<run>/actions/<id>.npy       actions the env's step() actually received, every mode
<run>/parity/<id>.json       policy steps whose vendored action differed (parity mode)
```

`<id>` is `<suite>-t<task>-i<init state>-s<seed>-<perturbation>`, for example
`libero_10-t09-i004-s0-none`.

## Timing convention

Step `k` holds the observation and world state **before** action `k`. `done` and `success`
are the result of executing action `k`. `step_idx` counts from env reset and includes the
10 wait (settle) steps; wait steps send the no-op `[0, 0, 0, 0, 0, 0, -1]`, run no
inference, and carry sentinels in every model field (`-1` for integers, NaN for floats).

`eef_pos`, `eef_quat_raw` and `object_poses` come from robosuite observations, which lag
`qpos` by one physics substep (2 ms; a 20 Hz control step is 25 substeps). `sim_state` and
`fixture_joint_qpos` are read from `qpos` and are current.

## Per-step columns

| Column | Shape | Dtype | Notes |
|---|---|---|---|
| `step_idx` | [T] | int32 | 0 … T-1, contiguous |
| `is_wait_step` | [T] | bool | |
| `slice_logits` | [T, 7, 256] | float16 | raw logits over token ids 31744–31999; slice index `j` is token `31744 + j` |
| `lse_full` | [T, 7] | float32 | log-sum-exp over the full 32064-wide head |
| `argmax_full_id` | [T, 7] | int32 | argmax token over the full head |
| `chosen_token_id` | [T, 7] | int32 | token `generate()` returned; equals `argmax_full_id` under greedy decoding |
| `bin_idx` | [T, 7] | int32 | after clipping to [0, 254] |
| `normalized_action` | [T, 7] | float32 | bin center in [-1, 1] |
| `unnormalized_action` | [T, 7] | float32 | after q01/q99 (dims 0–5); gripper unchanged |
| `clipped_flag` | [T, 7] | bool | raw bin fell outside [0, 254] |
| `executed_action` | [T, 7] | float64 | exactly what was passed to `env.step` |
| `inference_latency_ms` | [T] | float32 | the `generate` call only, synchronized |
| `wall_timestamp` | [T] | float64 | `time.time()` at the start of the step |
| `sim_state` | [T, S] | float64 | flattened MuJoCo state: `[time, qpos, qvel]` |
| `object_poses` | [T, n_obj, 7] | float32 | position + quaternion (xyzw, as the env exposes it) |
| `fixture_joint_qpos` | [T, n_joint] | float32 | articulated fixture joints (rad or m) |
| `eef_pos` | [T, 3] | float64 | |
| `eef_quat_raw` | [T, 4] | float64 | env-native order (xyzw); not converted |
| `gripper_qpos` | [T, 2] | float64 | |
| `done`, `success` | [T] | bool | `success` is false on wait steps |

Row order of `object_poses` and `fixture_joint_qpos` is given by the `object_names` and
`fixture_joints` metadata entries.

## Action decoding

`token → bin = clip(32000 - token - 1, 0, 254)`; the bin centers are the 255 midpoints of
`linspace(-1, 1, 256)`. Consequences:

- Higher token id means lower bin; the ordinal direction is inverted relative to the id.
- Tokens below 31744 (ordinary text) clip to bin 254, which normalizes to +1 (q99).
- Tokens 32000–32063 (padding) clip to bin 0, which normalizes to -1 (q01).
- Token 31744 has raw bin 255 and clips to 254, the same as token 31745, so there are
  **255 effective bins**. `clipped_flag` is true for it and for every out-of-slice token.
- Dimensions 0–5 are unnormalized with q01/q99. The gripper (dimension 6) is masked out:
  its decoded value is a bin center near 0 (closed) or 0.996 (open), binarized at 0.5 and
  sign-inverted for the env (`-1` open, `+1` close).

All entropy, softmax and margin quantities are derived offline from `slice_logits` and
`lse_full`; nothing derived is stored. The fp16 slice is lossless for the bf16 logits the
model produces, except for values within about 6e-5 of zero.

## Metadata

`schema_version`, `episode_id`, `suite`, `task_id`, `init_state_idx`, `seed`,
`perturbation_config_id`, `task_instruction`, `prompt` (exact string), `object_names`,
`fixture_joints` (name, type, range, qpos address), `static_fixture_poses` (once per
episode), `mjcf_sha256`, `checkpoint`, `checkpoint_commit`, `hub_code_commit`,
`model_class_module`, `unnorm_key`, `center_crop`, `image_flip`, `control_freq_hz`,
`num_wait_steps`, `max_steps`, `action_stats` (q01, q99, mask), `git` (harness and sibling
SHAs with dirty flags), `versions`, `gpu`, `final_success`, `terminated_by`
(`success` or `max_steps`), `num_policy_steps`, `episode_length`.

## Limits

- `sim_state` restores poses and joint positions; it does not include controller state, so
  it is not sufficient for dynamics replay. Restoring and running `forward()` reproduces
  `sim_state` and `fixture_joint_qpos` exactly and the observation-derived poses to within
  one substep of motion.
- Camera frames are not stored; the labeling video is the only image record. The exact
  model inputs cannot be re-fed offline.
- Size is about 2 MB per episode for ~420 steps; `slice_logits` is about 70% of it.

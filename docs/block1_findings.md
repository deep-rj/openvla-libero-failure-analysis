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
  repo's. The hub revision is unpinned. **pod**: confirm which class runs via
  `type(vla).__module__`.

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

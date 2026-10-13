# Block 4: failure detector

Predict terminal failure from the policy's internal signals early enough to act (`PLAN.md`).
Started 2026-10-12, after the separability gate (`docs/separability_gate.md`) and before Block 3.

**Protocol.** Signals are explored on the baseline 250 episodes; every number from them is a
development number. The detector (features, window, thresholds, operating point) is frozen
before init states 25–49 are run, and is evaluated on them once.

## Design

- **Inputs: what a real robot has.** The policy's outputs (`slice_logits`, `lse_full`, emitted
  actions) and proprioception (`eef_pos`, `eef_quat_raw`, `gripper_qpos`). World state
  (`object_poses`, `fixture_joint_qpos`, `sim_state`) is for labels and evaluation only. Hidden
  states are not logged and are out of scope for this block.
- **When it speaks: at each close command.** The detector reads every step and emits a verdict
  when the policy's gripper action switches from open to close. The gate's anchors cannot run
  online (they use object poses and look back from the close), and the close command lands within
  one step of the anchor for 52 of 60 failures and 174 of 198 successes, so the gate's 50-step
  window carries over. Covers the 61 failures with a closing anchor, of 79: 9 open descents, 5 with
  no anchor, 4 fixture. Alerting at every step is parked.
- **Lead time is measured against the width baseline.** Not against onset: a grasp failure's onset
  is the bottom of its descent by protocol, so onset minus anchor is 0 (median). Not against a
  no-return point: none is labeled, since LIBERO failures end at the step cap.
- **Width baseline: 3 mm within 60 steps of the command** (`harness/detector.py`). Tuned on the
  baseline 250, frozen with the detector and evaluated once on init states 25–49. 4 mm is the
  development knee (37 of 60, 1 of 198) but sits 0.3 mm below the narrowest successful bowl-rim
  grasp; 3 mm keeps 1.3 mm.
- Open: at a close command, predict "this attempt fails" or "this episode fails".

## Exploration log

Same anchors and units as the gate, after its anchor fix
(`docs/results/baseline/gate/anchors.csv`: 70 grasp and perception failures, 202 success grasps), same-task AUROC with failures above successes, and
99.17% bootstrap CIs over episodes.

### Far-bin mass (2026-10-12)

The probability a step puts on bins at least 10 from its top bin: a genuinely different action
(10 bins ≈ 3 mm per step on x). Window mean over the last 40, 50 and 75 pre-grasp steps, on all
steps and on steps where the no-motion bin is not in the top two (`scripts/gate_far_bins.py`,
`docs/results/baseline/block4/far_bins.md`). 2,000 draws; 39 tests, each with a beyond-exp(H)
version.

- It separates about as well as exp(H): x 0.63–0.64, pitch 0.64–0.67 (all steps); roll
  0.69–0.73, pitch 0.62–0.66, z 0.65 and y 0.62 at 75 steps, x 0.62 at 50 (stop not in the top
  two).
- Failures put more probability on a different action. Stop-filtered roll averages 0.106 in
  failures against 0.075 in successes over 75 steps; pitch 0.103 against 0.084.
- It tracks exp(H) closely (r 0.61–0.92 on motion dims, 0.99 on the gripper). After removing
  its linear dependence on window exp(H), no dim separates (0.43–0.61). At this n, far-apart
  doubt adds nothing measurable beyond entropy.

![Far-bin mass per step](results/baseline/block4/far_bins_curves.png)

### Width baseline (2026-10-13)

An empty close read from finger width alone (`scripts/block4_width_baseline.py`,
`docs/results/baseline/block4/width_baseline.md`): at the close command nearest each closing
anchor, fire at the first step within 60 whose width (`gripper_qpos` q0 − q1) is below the
threshold. 60 failures and 198 successes have a command within 10 steps of the anchor.

- At 3 mm it fires on 34 of 60 failures and 1 of 198 successes, a median 18 steps (0.9 s sim)
  after the command. The fingers approach full close slowly, so lower thresholds fire later and on
  fewer failures (1 mm: 14 of 60, 38 steps) with the same false alarm.
- Above 4 mm, successful thin grasps enter: bowl rims (narrowest 4.3 mm), the porcelain mug handle
  (5.7 mm). 6 mm fires on 23 successes.
- The false alarm at every threshold from 1 to 4 mm is one butter grasp (`t01-i008`, 1.0 mm), not
  yet checked on video.
- 15 failures stay at 15 mm or wider and are invisible to width.
- Minimum width over the horizon separates better than the internal signal: same-task AUROC 0.76
  (99.17% CI 0.63 to 0.88) against 0.62–0.63 for exp(H) in the gate. It is read up to 60 steps
  after the command, so the detector's case is timing and the failures width cannot see, not
  classification.
- Failures release the close quickly: the close command holds a median 5.5 steps in failures
  against 54.5 in successes. A post-hoc signal; not pursued.

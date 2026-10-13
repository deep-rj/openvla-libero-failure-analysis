# Block 4: failure detector

Predict terminal failure from the policy's internal signals early enough to act (`PLAN.md`).
Started 2026-10-12, after the separability gate (`docs/separability_gate.md`) and before Block 3.

**Protocol.** Signals are explored on the baseline 250 episodes; every number from them is a
development number. The detector (features, window, thresholds, operating point) is frozen
before init states 25–49 are run, and is evaluated on them once.

## Exploration log

Same anchors and units as the gate (`docs/results/baseline/gate/anchors.csv`: 70 grasp and
perception failures, 202 success grasps), same-task AUROC with failures above successes, and
99.17% bootstrap CIs over episodes.

### Far-bin mass (2026-10-12)

The probability a step puts on bins at least 10 from its top bin: a genuinely different action
(10 bins ≈ 3 mm per step on x). Window mean over the last 40, 50 and 75 pre-grasp steps, on all
steps and on steps where the no-motion bin is not in the top two (`scripts/gate_far_bins.py`,
`docs/results/baseline/block4/far_bins.md`). 2,000 draws; 39 tests, each with a beyond-exp(H)
version.

- It separates about as well as exp(H): x 0.63, pitch 0.64–0.67 (all steps); roll 0.69–0.73,
  pitch 0.62–0.66, z 0.65 and y 0.63 at 75 steps (stop not in the top two).
- Failures put more probability on a different action. Stop-filtered roll averages 0.106 in
  failures against 0.078 in successes over 75 steps; pitch 0.103 against 0.084.
- It tracks exp(H) closely (r 0.60–0.92 on motion dims, 0.99 on the gripper). After removing
  its linear dependence on window exp(H), no dim separates (0.43–0.61). At this n, far-apart
  doubt adds nothing measurable beyond entropy.

![Far-bin mass per step](results/baseline/block4/far_bins_curves.png)

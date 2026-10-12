# Separability gate

The check before Block 4 (`PLAN.md`): does any internal signal of the policy differ between
failed and successful rollouts **before** the moment a grasp fails? If nothing does, the project
falls back to the descriptive study (taxonomy plus robustness sweep).

## Pre-registration (2026-10-12, before any outcome split)

**Units.**
- Failures: `perception` and `control: grasp` failures at objects
  (`failure_mechanisms.csv`). Fixture failures are excluded (no grasp descent):
  `t02-i007`, `t02-i008`, `t02-i014`, `t03-i008`.
- Successes: each place subgoal of each successful episode, i.e. one unit per object grasped.

**Anchor.** The bottom of a grasp descent at a goal object: the pad-midpoint low point before
a close, or of a descent that never closes (the classifier's onset rule).
- Failure: the descent nearest the labeled onset, within 60 steps. Pilot onsets predate the
  bottom-of-descent rule, so the labels only pick the attempt. A slip (a goal object held at
  the step before the onset) anchors on the grasp that began the slipped hold.
- Success: the closing descent that began the object's last hold.
- Failures with no descent found are excluded: `t00-i020`, `t00-i022`, `t00-i024`,
  `t01-i022`, `t05-i018`. One success unit has none: `t05-i010` (book).

**Window.** Up to 50 non-wait steps before the anchor (the anchor excluded); units with fewer
than 10 are dropped. None were (shortest 28).

**Metrics**, per unit, averaged over the window. exp(H) is over the 255-bin distribution
(`pilot_report.derive_step_metrics`).
- Primary: gripper (dim 6) mean exp(H).
- Exploratory: mean exp(H) of each motion dim 0–5, one test each.
- Secondary, reported only: gripper near-tie rate (bin margin ≤ 4 bf16 ulps).

**Test.** AUROC = P(failure value > success value), ties half. Same-task AUROC compares only
pairs within one (task, object) stratum and is decisive; pooled AUROC is reported beside it.
CIs come from a percentile bootstrap over episodes, resampled within (task, outcome), with
10,000 draws and seed 0. Levels: primary 95%; exploratory 1 − 0.05/6 = 99.17% each.

**Pass rule.**
- The gate passes if the primary or any exploratory same-task CI excludes 0.5, in either
  direction.
- **P1** (failures are less decided on the gripper before the grasp) is supported only if the
  primary same-task CI lies above 0.5.

**Curves** (descriptive, not tested): per-step exp(H) per dim, from −50 to +20 steps around the
anchor, as a mean and interquartile band per outcome.

## Changes to the 2026-10-06 predictions (`block1_findings.md`)

- **P1 amended.** It was a per-episode gripper near-tie count within 5 cm of the target. It
  becomes the windowed gripper exp(H) above. Reason: blind pooled counts showed 69% of windows
  with zero near-ties at ≤ 4 ulps (83% at ≤ 1, 43% at ≤ 16), so the AUROC would mostly compare
  zeros. The near-tie rate stays as the secondary.
- **P3 withdrawn** (rotation entropy does not separate). Covered by the per-dim exploratory
  tests.
- **P4 withdrawn** (gripper saturation does not separate). Covered by the primary.

## Seen before the freeze

- Pooled near-tie counts per window, both outcomes mixed (above).
- A dry run on labels shuffled within task. Null calibration over 40 shuffles: the 95% CI
  excluded 0.5 in 2/40 (gripper exp(H)), 2/40 (near-tie rate) and 3/40 (x exp(H)), against
  2/40 expected.

## Reproduce

```bash
python scripts/gate_anchors.py outputs/baseline \
    --mechanisms docs/results/baseline/labeling/failure_mechanisms.csv --out outputs/baseline/gate
python scripts/separability_gate.py outputs/baseline --out outputs/baseline/gate
```

## Result

Not run.

# Labeling protocol (provisional until the pilot labels exist)

Status: living. The milestone and mechanism coding rules are provisional; they are revised
after the ~20 pilot labels and then frozen (commit + hash) before the separability gate's
first plot. Any later change is logged as a decision, and the gate and Block 4 are re-run.
The agent does not label.

## Independence from model outputs (hard rule)

Labels, event detectors and detector thresholds never use model outputs. Until the label CSV
is committed:

- No per-episode or time-aligned plots of logits, entropy or confidence.
- No model-output values in video overlays or filenames.
- The pooled pilot report (`scripts/pilot_report.py`) is allowed: it has no per-episode or
  time-aligned output and no split by outcome.

Allowed while labeling: the video (its overlay shows only the step index and `WAIT`) and
world-state signals (finger width, end-effector position and speed, object poses, fixture
joints). They come from the simulator, not the model.

## The draw

`scripts/select_labeling.py <run> --n 20 --seed 0` reads outcome and task only. It draws
failures round-robin across tasks with `random.Random(seed)`; fewer than 20 failures means all
of them. It writes `labeling/label_selection.json` and an empty `labeling/labels_template.csv`,
and never overwrites an existing label file. The pilot labels are the development set for the
coding rules; agreement is measured on the remaining ~30 of Block 2's ~50.

This seed is the draw's own and has no relation to the run seed.

## Label columns

One row per selected episode (`harness/labeling.py`, `LABEL_COLUMNS`).

| Column | Meaning |
|---|---|
| `k_completed`, `phase` | The milestone, as a pair: subgoals completed at the highest point the episode reached, and the phase of the next subgoal at that point. |
| `mechanism` | The earliest error that made success unreachable. Starts from perception / grounding / control / recovery plus `other`. |
| `mechanism_notes` | Free text on what went wrong. |
| `first_error_step`, `no_return_step` | The two onset anchors, as the overlay's step index (wait steps included): the earliest error, and the point after which success was unreachable. |
| `no_onset` | No meaningful onset. These episodes are their own class and are excluded from alignment analysis. |
| `wrong_object` | The robot grasped or moved an object that is not part of the goal. |
| `regression` | Progress was undone, for example a placed object was later removed. |
| `stalled` | The arm was essentially still for an extended stretch before the episode ended. |
| `no_reattempt` | After the error, the robot made no re-attempt at the failed subgoal. |
| `notes` | Free text. |

What follows the earliest error goes in the flags, not in a second mechanism.

## Milestone rules

- **Highest reached, not end state.** A subgoal that was completed counts even if later
  undone ("ever placed"). Regressions are recorded in `regression`.
- **Correct objects only.** Only objects named in the goal advance the milestone. A wrong
  object is a flag, never a milestone.
- **Observable behavior, not intent.** "Seemed to be attempting recovery" is not a label.
- **Perceptual definitions are enough for hand labels.**
  - *Grasped:* the object is held off the surface, the fingers stopped at the object's width
    (not fully shut, which means holding nothing), and it moves with the end effector.
  - *Carried:* the held object is horizontally over the basket.
  - *Placed:* released with the object in the basket.
  Numeric thresholds are chosen only when detectors are written, after the pilot.
- A failed episode cannot end at the goal; failures occupy `k_completed` below the number of
  subgoals.

Task 7 (put both the alphabet soup and the cream cheese box in the basket) as the worked
example, with `k` counted in objects placed:

| (k, phase) | Highest point reached |
|---|---|
| (0, none) | no correct object grasped |
| (0, grasped) | grasped one correct object, not yet over the basket |
| (0, carried) | carried one correct object to the basket, not released in it |
| (1, none) | one correct object placed |
| (1, grasped) | grasped the second, not yet over the basket |
| (1, carried) | carried the second to the basket, not released in it |

## Open (the human decides)

- How `k` and the phase vocabulary are defined for tasks that are not two-object pick-and-place,
  including the articulated tasks (drawer, microwave door, stove). The pair `(k, phase)` is
  chosen; only task 7 has a ladder written down. These are due before the pilot labels, not
  before the baseline.
- How an episode truncated before a rule can be applied is recorded: as a mechanism value or a
  flag (for example "undetermined, truncated").
- The `stalled` rule (how still, for how many steps) and what counts as a re-attempt in
  `no_reattempt`. Both should be behavioral and may use world-state signals.
- The mechanism categories and their definitions, after the pilot labels.

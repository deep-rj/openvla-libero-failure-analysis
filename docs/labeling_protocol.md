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
joints). They come from the simulator, not the model. Computed columns read only the
world-state fields of a trace.

## The draw

`scripts/select_labeling.py <run> --n 20 --seed 0` reads outcome and task only. It draws
failures round-robin across tasks with `random.Random(seed)`; fewer than 20 failures means all
of them. It writes `labeling/label_selection.json` and an empty `labeling/labels_template.csv`,
and never overwrites an existing label file. The pilot labels are the development set for the
coding rules; agreement is measured on the remaining ~30 of Block 2's ~50.

This seed is the draw's own and has no relation to the run seed.

## Computed vs hand-labeled

Anything the simulator can decide is computed; hand labels cover only judgment.

- **Computed** (pure functions over world-state fields extracted from the replayed
  `sim_state`): `k_completed`, `phase` and the world-state flags below. Predicates are
  LIBERO's own (`libero/libero/envs/predicates`), so the milestone and LIBERO's success check
  use the same definitions.
- **Hand-labeled from video:** `mechanism`, `mechanism_notes`, the two onset steps,
  `no_onset`, `notes`, and `computed_check`: the computed columns that disagree with the
  video, with a reason, or empty. Computed columns are validated this way on the pilot.

## Milestone

### Definition

Each task has a set of subgoals (its BDDL goal predicates) and a dependency table. At each
step, a subgoal **counts** if its predicate has held for the last N steps (debounce) and every
subgoal it depends on also counts at that step.

- `k_completed` = the maximum over steps of the number of counting subgoals. Highest reached,
  not end state: later undoing is recorded in `regression`, and `k` never decreases.
- `phase` = the furthest phase toward a next subgoal reached while `k` was at its maximum.
- A failed episode never has all subgoals counting at once, so `k` is below the number of
  subgoals. Subgoals that each held at some point, but never together, do not add up.
- Only goal objects in their goal regions advance `k`. A goal object in a different region is
  `wrong_target`, never a milestone.

### Phases

- **grasped:** the object is held: off its support surface, fingers stopped at the object's
  width (not fully shut), moving with the end effector.
- **carried:** grasped, and the object's horizontal (xy) position lies inside the target
  region's footprint: the target predicate without its height condition.
- **contacted** (fixtures): the gripper is touching the drawer front, door or knob.
- **partial** (fixtures): the joint has moved at least X% of the way from its initial value
  to the goal threshold.

Placement is decided by the predicate, not by how it looks; video is used only for what the
predicates cannot express.

### Ladders

| Task | Subgoals | Dependencies | Phases toward the next subgoal |
|---|---|---|---|
| 0 | soup `In` basket, tomato sauce `In` basket | none | grasped / carried |
| 1 | cream cheese `In` basket, butter `In` basket | none | grasped / carried |
| 2 | stove `Turnon`, moka pot `On` stove cook region | none ⚑ | knob: contacted / partial · pot: grasped / carried |
| 3 | bowl `In` bottom drawer, drawer `Close` | `Close` after `In` | bowl: grasped / carried · drawer: contacted / partial |
| 4 | white mug `On` left plate, yellow-white mug `On` right plate | none | grasped / carried |
| 5 | book `In` caddy back compartment | — | grasped / carried |
| 6 | white mug `On` plate, pudding `On` region right of plate | none | grasped / carried |
| 7 | soup `In` basket, cream cheese `In` basket | none | grasped / carried |
| 8 | pot 1 `On` stove, pot 2 `On` stove; stove `Turnon` (true at init) | none ⚑ | grasped / carried |
| 9 | yellow-white mug `In` microwave, door `Close` | `Close` after `In` | mug: grasped / carried · door: contacted / partial |

A subgoal depends on another when completing it first would require undoing it to succeed
(closing an empty drawer or microwave). Dependencies come from the physical scene, not the
BDDL file, whose goal is an unordered conjunction.

Fixture facts, from LIBERO's object definitions (`articulated_objects.py`) and a check over
all 50 init states:

- Stove (tasks 2, 8): knob joint ≥ 0.5 rad is on. Off at init in task 2, on at init in task 8;
  in task 8 the stove subgoal is lost only if the knob is knocked off (`regression`).
- Bottom drawer (task 3): prismatic, open 0.14–0.16 m at init; `Close` requires ≥ 0.0 m (fully
  in). Between the two, neither `Open` nor `Close` holds.
- Microwave door (task 9): hinge, open 1.3–2.1 rad at init; `Close` requires within 0.005 rad
  of shut. The arm must withdraw before the door can close.

### ⚑ Judgment points to check on video

- **Task 2 order:** whether a pot on the stove blocks the knob. If it does, the pot's `On`
  depends on `Turnon`.
- **Task 8 order:** whether the first pot can occupy cook-region space the second needs.
- **Stove indicator:** whether the burner's appearance changes when on, so the video can
  confirm `Turnon`.
- **Task 6:** the region right of the plate has no visible boundary; `carried` and placement
  for the pudding are judged from the printout, not the video.
- **Grasp at 256 px:** where a grasp is ambiguous, use the world-state printout (finger width
  vs object width, object height).

## Flags

All world-state flags are computed. Each threshold is set on the pilot 20 with a physical
rationale, swept as a sensitivity check (not searched), and frozen with the coding rules.

| Flag | Definition | Threshold(s) |
|---|---|---|
| `wrong_object` | an object not named in the goal is grasped and moved more than D cm | D |
| `wrong_target` | a goal object, released and at rest, satisfies an `In`/`On` predicate against a declared region that is not its goal region | rest speed, dwell |
| `regression` | a subgoal counted in `k` later stops holding (debounced) | N |
| `order_violation` | a dependent subgoal holds before its prerequisite does | N |
| `dropped` | a grasped object leaves the grasp while not over any declared region | — |
| `collision_disturbance` | an object that is not grasped moves more than D cm (knocked) | D |
| `regrasp_count` | number of grasp onsets on goal objects (integer) | — |
| `stalled` | end-effector speed below v for at least T s | v, T |
| `dither` | over a window W, end-effector path length ≥ R × net displacement | W, R |
| `gripper_chatter` | at least C open/close cycles in a window W with no object between the fingers | C, W |
| `object_off_table` | a goal object falls below table height | — |
| `workspace_limit` | a joint within ε of its limit, or the end effector at the reach boundary | ε |
| `truncated_mid_action` | at the step cap, an object is grasped or a goal object or fixture is in motion | rest speed |

`wrong_target` sees only regions declared in the BDDL file; placements elsewhere (floor,
basket rim) go in `notes`, so `wrong_target = 0` does not mean "never misplaced".

Threshold notes:

- **Debounce N.** LIBERO's success check has zero debounce: the episode ends on the first step
  all goal predicates hold. N guards against contact flicker in `On` and objects falling
  through an `In` region; candidates N ∈ {1, 5, 10, 20} at 20 Hz.
- **Stall speed v.** Set above the end-effector jitter measured over the 10 `WAIT` steps, when
  the arm is commanded to hold still.

## Mechanism (hand-labeled)

`mechanism` is the earliest error that made success unreachable. Starts from perception /
grounding / control / recovery plus `other`; categories are revised after the pilot. What
follows the earliest error goes in the flags, not in a second mechanism.

- **Observable behavior, not intent.** "Seemed to be attempting recovery" is not a label.
- **Truncation is a symptom, not a mechanism.** LIBERO has no failure termination; every
  failure ends at the step cap. The mechanism is the cause of the shortfall.
- **Slow but error-free failures** are `no_onset` with `truncated_mid_action`. The cap is 520
  policy steps against a longest training demo of 505 (`run_libero_eval.py`), so slowness
  alone is the budget running out, not a policy error.
- `first_error_step`, `no_return_step`: the overlay's step index (wait steps included).
  An error from the first policy step (for example heading for a wrong object) has onset 10.
- `no_onset`: no identifiable moment of error (slow but error-free, gradual drift). Its own
  class, excluded from alignment analysis.

## Wait steps

Steps 0–9 send a no-op (`[0, 0, 0, 0, 0, 0, -1]`, gripper open) while spawned objects settle
under gravity. Step indices keep counting from 0 everywhere; the first policy step is 10.

- Nothing is judged, labeled or flagged before step 10.
- Displacements ("moved more than D cm") are measured from the settled poses at step 10, not
  the spawn poses at step 0.
- Wait steps supply the end-effector jitter baseline for `stalled`.

## Open (the human decides)

- Threshold values (N, X, D, v, T, W, R, C, ε), on the pilot 20.
- The mechanism categories and their definitions, after the pilot labels.

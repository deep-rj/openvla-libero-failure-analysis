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
  LIBERO's own (`libero/libero/envs/predicates`); placement adds a release condition (below).
- **Hand-labeled from video:** `mechanism`, `mechanism_notes`, the two onset steps,
  `no_onset`, `notes`, and `computed_check`: the computed columns that disagree with the
  video, with a reason, or empty. Computed columns are validated this way on the pilot.

## Milestone

### Definition

Each task has a set of subgoals (its BDDL goal predicates) and a dependency table. At each
step, a subgoal **counts** if it has been **placed** for the last N steps and every subgoal it
depends on also counts at that step. For a place subgoal, placed means its predicate holds
and its object is not grasped; for a fixture subgoal, its predicate holds.

The release condition departs from LIBERO's success check on purpose. LIBERO evaluates the
predicate at one instant with no release or stability test, so a touchdown while still gripped
satisfies it; in the baseline, 70 of 130 successes end with a goal object still gripped (0 of
25 in tasks 3 and 9, where the last subgoal is closing). `k` measures what the robot completed;
LIBERO's success remains the outcome metric, and a success has `k` equal to the number of
subgoals by definition.

- `k_completed` = the maximum over steps of the number of counting subgoals. Highest reached,
  not end state: later undoing is recorded in `regression`, and `k` never decreases.
- `phase` = the furthest phase toward a next subgoal reached while `k` was at its maximum.
- A failed episode never has all subgoals counting at once, so `k` is below the number of
  subgoals. Subgoals that each held at some point, but never together, do not add up.
- Only goal objects in their goal regions advance `k`. A goal object in a different region is
  `wrong_target`, never a milestone.
- A success episode has `k` equal to the number of subgoals by definition. The trace logs the
  state before each action, so the state the succeeding action produced is never recorded.

### Phases

- **grasped:** starts when the object, lifted above its settled height and near the end
  effector with the fingers not open, moves at least 5 mm over a short window while its offset
  from the end effector stays fixed. It continues while the fingers stay closed and the offset
  stays fixed, including while the arm pauses. The object itself must move: a still object
  next to a moving or still gripper keeps a near-fixed offset trivially. Finger width alone
  does not separate a grasp from an empty close: a bowl held by its rim leaves the fingers
  about 4 mm apart.
- **carried:** grasped, and the object's horizontal (xy) position passes the xy part of the
  target predicate: inside a region's box, or within 3 cm of a target object's origin (the
  radius LIBERO's object-on-object `On` uses, so off-center on a plate does not count).
- **engaged** (fixtures): the joint has moved at least 5% of the way from its settled value to
  the goal threshold.
- **partial** (fixtures): at least X% of the way.

Fixture phases come from joint travel, not gripper contact. Contact is checked only at control
steps (every 25 physics substeps), and a push makes and breaks contact within a step: a
drawer closing at a steady 7 mm/step shows no robot contact at any logged state.
`closure_fraction` (the most travel achieved, 0 to 1) is a computed column for each fixture
subgoal, so the buckets can be re-cut later.

Placement is decided by the predicate, not by how it looks; video is used only for what the
predicates cannot express.

### Ladders

| Task | Subgoals | Dependencies | Phases toward the next subgoal |
|---|---|---|---|
| 0 | soup `In` basket, tomato sauce `In` basket | none | grasped / carried |
| 1 | cream cheese `In` basket, butter `In` basket | none | grasped / carried |
| 2 | stove `Turnon`, moka pot `On` stove cook region | none | knob: engaged / partial · pot: grasped / carried |
| 3 | bowl `In` bottom drawer, drawer `Close` | `Close` after `In` | bowl: grasped / carried · drawer: engaged / partial |
| 4 | white mug `On` left plate, yellow-white mug `On` right plate | none | grasped / carried |
| 5 | book `In` caddy back compartment | — | grasped / carried |
| 6 | white mug `On` plate, pudding `On` region right of plate | none | grasped / carried |
| 7 | soup `In` basket, cream cheese `In` basket | none | grasped / carried |
| 8 | pot 1 `On` stove, pot 2 `On` stove; stove `Turnon` (true at init) | none ⚑ | grasped / carried |
| 9 | yellow-white mug `In` microwave, door `Close` | `Close` after `In` | mug: grasped / carried · door: engaged / partial |

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

- **Task 2 order:** resolved, no dependency. The knob is set into the table surface, clear of
  the cook region, so a pot on the stove does not block it (from scene geometry; no pot-first
  episode observed yet).
- **Task 8 order:** whether the first pot can occupy cook-region space the second needs.
- **Stove indicator:** resolved; the coils turn red when on, so the video confirms `Turnon`.
- **Task 6:** the region right of the plate has no visible boundary; `carried` and placement
  for the pudding are judged from the printout, not the video.
- **Grasp at 256 px:** where a grasp is ambiguous, use the world-state printout (finger width
  vs object width, object height).

## Flags

All world-state flags are computed. Each threshold is set on the pilot 20 with a physical
rationale, swept as a sensitivity check (not searched), and frozen with the coding rules.

| Flag | Definition | Threshold(s) |
|---|---|---|
| `wrong_object` | an object not named in the goal is grasped and moved more than D from its settled pose | D = 2 cm |
| `wrong_target` | a goal object, released and at rest for a dwell, satisfies its goal predicate against another receptacle | < 1 cm/s for 10 steps |
| `regression` | a counted subgoal later stops counting for N steps | N |
| `order_violation` | a dependent subgoal's predicate holds while its prerequisite does not count | N |
| `dropped` | a goal object is released away from its target and falls at least 3 cm within 10 steps; setting it down is not a drop | 3 cm, 10 steps |
| `collision_disturbance` | an object moves more than D while not held; excluded: the 10 steps before a grasp onset (closing fingers drag an object about 2 cm before it lifts), the 10 steps after a release, and motion inside its goal container (a bowl in a closing drawer) | D = 2 cm |
| `regrasp_count` | grasp onsets on goal objects beyond the first per object (integer) | — |
| `stalled` | the arm stops: end-effector speed below v for at least 2 s; v = max(3 × median wait-step speed, 5 mm/s) | v, 2 s |
| `oscillation` | at least 6 direction reversals of at least 2 cm along one axis within 4 s. A path-to-displacement ratio does not work: reaching down and lifting has a high ratio (it flagged 112 of 130 successes). Named for what it measures: in control engineering "dither" is a deliberately injected signal, and the robot-learning term for the likely cause, mode switching, is internal to the policy | 6, 2 cm, 4 s |
| `gripper_chatter` | at least 3 empty closes (finger width below 2 cm) within 2 s with nothing held | 3, 2 s |
| `object_off_table` | a goal object at least 5 cm below its settled height | 5 cm |
| `workspace_limit` | an arm joint within 2% of its range from either limit for 10 steps; gripper fingers excluded. A fully extended arm shows as joint 4 at its upper limit, so no separate reach check | 2%, 10 steps |
| `truncated_mid_action` | a failure whose last logged step has a goal object grasped, or a goal object or fixture joint moving | 1 cm/s |

Receptacles are the BDDL regions attached to an object or fixture (caddy compartments, other
drawers, the microwave top) plus every goal target (the other plate). Table regions such as
init regions are not receptacles. Placements elsewhere (floor, basket rim) go in `notes`, so
`wrong_target = 0` does not mean "never misplaced".

Threshold notes:

- **N (steps placed after release) = 10** (0.5 s): a released object that slides or tips does
  so within about half a second (a released moka pot settled within 14 steps). On the
  baseline, N ∈ {1, 5, 10, 20} gives identical `(k, phase)` for all 120 failures.
- **Stall speed v.** Above the end-effector jitter over the 10 `WAIT` steps, when the arm is
  commanded to hold still, with a 5 mm/s floor.
- Joint-limit margins are a fraction of each joint's range, not an angle, so they hold for
  prismatic and revolute joints alike.

## Mechanism (hand-labeled)

`mechanism` is the initiating event of the failure sequence: the earliest error the episode
never recovered from, whether or not success was still physically reachable after it. Starts
from perception / grounding / control / recovery plus `other`; categories are revised after the
pilot. What follows the initiator (the failed recovery and the end state) goes in the flags and
`notes`, not in a second mechanism. A failure is read as an event tree: initiator, then the
recovery barrier (computed where possible, e.g. `regrasp_count`), then the end state.

- **Decidable from observables.** A category must be distinguishable on video or in world
  state. A grasp that closes off-center on the right object is `control`; `perception` needs
  visible evidence such as the arm going to where the object is not. For a grasp: if the target
  is between the fingers at the bottom of the descent, a miss is `control`; if not, `perception`.
  The target is the part meant to be grasped (the moka pot's handle, not its body).
- **`mechanism_notes`** describe the initiator only. Unverified causes are written as
  `Hypothesis: ...`, never as fact.
- **`notes`** hold what follows the initiator and anything else. Post-initiator behavior starts
  with a fixed phrase so it can be counted, e.g. `post: hover` (slow motion confined near one
  spot, neither stopped nor oscillating), `post: stall`, or `post: premature close` (a fixture
  moved toward closed before its prerequisite holds, short of a full close), or `post: jammed`
  (a carried object stays pinned against an obstacle while the arm keeps moving).
- Behavior before the initiator starts with `pre:`, reusing the `post:` phrases (`pre: stall`);
  an error the episode recovered from is `pre: recovered <error>` (`pre: recovered miss`).
  Only failures are labeled, so these counts say nothing about recovery rates. This applies even when a flag covers
  the behavior, since flags do not record when it happened relative to the initiator.

- **Observable behavior, not intent.** "Seemed to be attempting recovery" is not a label.
- **Truncation is a symptom, not a mechanism.** LIBERO has no failure termination; every
  failure ends at the step cap. The mechanism is the cause of the shortfall.
- **Slow but error-free failures** are `no_onset` with `truncated_mid_action`. The cap is 520
  policy steps against a longest training demo of 505 (`run_libero_eval.py`), so slowness
  alone is the budget running out, not a policy error.
- `first_error_step`, `no_return_step`: the overlay's step index (wait steps included).
  An error from the first policy step (for example heading for a wrong object) has onset 10.
  Tolerance ±10 steps (0.5 s).
- `first_error_step` is the onset of the initiator: when the behavior starts, not when the damage
  becomes visible. A risk factor that successes also show (for example a grasp offset within the
  successes' range) is not the initiator.
- `no_return_step` is state-based: the first step at which the world state makes success
  physically impossible (object off the table or out of reach). `none` if that never happens;
  running out of steps is not a no-return point. A toppled object is not a no-return point:
  LIBERO's `On` checks position and contact, not orientation.
- `no_onset` is `0` or `1`. Leave `notes` blank when there is nothing to add.
- `no_onset`: no identifiable moment of error (slow but error-free, gradual drift). Its own
  class, excluded from alignment analysis.

## Wait steps

Steps 0–9 send a no-op (`[0, 0, 0, 0, 0, 0, -1]`, gripper open) while spawned objects settle
under gravity. Step indices keep counting from 0 everywhere; the first policy step is 10.

- Nothing is judged, labeled or flagged before step 10.
- Displacements ("moved more than D cm") are measured from the settled poses at step 10, not
  the spawn poses at step 0.
- Wait steps supply the end-effector jitter baseline for `stalled`.

## Rule changes

Logged as made during the pilot; earlier episodes are re-passed when a change affects them.

- `t00-i013`: mechanism is the initiator of the failure sequence (event tree), not
  the error that made success unreachable; `no_return_step` is state-based with `none`; onset
  tolerance ±10 steps; `mechanism_notes` initiator only, post-initiator behavior in `notes` with
  `post:` phrases; `stalled` means stopped; `dither` renamed `oscillation`.
- `t00-i024`: `post:` phrases also cover flagged behavior (`post: stall`); a grasp
  is `perception` when the target is not between the fingers at the bottom of the descent.
- `t02-i009`: the grasp target is the part meant to be grasped (pot handle); a
  topple is not a no-return point; `post: stall` is judged on the arm, gripper motion aside.
- `t03-i009`: `post: premature close` for a partial close of an empty drawer or door;
  `order_violation` covers only a full close.
- `t09-i015`: `post: jammed`.
- `t01-i010`: `pre:` phrases for behavior before the initiator, `pre: recovered <error>` for
  recovered errors; applied to `t09-i015` (`pre: stall`).

## Open (the human decides)

- Threshold sensitivity on the pilot 20; values above are the defaults.
- The mechanism categories and their definitions, after the pilot labels.
- Flag timing: record when each flag fires (intervals, counts, durations), so pre/post-initiator
  ordering is computed rather than tagged. Format undecided (interval table, JSON, or count plus
  interval list). After labeling.
- `truncated_mid_action` fires on a goal object moving at the cap with the gripper not engaged
  (`t00-i024`: cans nudged while hovering). Whether to require gripper involvement.
- The grasp detector misses a handle grasp until the object lifts (`t09-i015`: fingers closed
  on the mug handle from ~130, counted as held from 356), so `collision_disturbance` fired on
  the mug creeping in the gripper.

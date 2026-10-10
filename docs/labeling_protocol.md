# Labeling protocol

Status: frozen for the held-out labels. The coding rules were revised on the 20 pilot labels,
frozen at git tag `block2-freeze`, and amended once by the classifier rule session, frozen
at `block2-classifier-freeze` (see "Freeze"). Any later change is logged as a decision, and the
gate and Block 4 are re-run. The agent does not label.

## Independence from model outputs (hard rule)

Labels, event detectors and detector thresholds never use model outputs. Until all ~50 hand
labels (pilot and held-out) are committed:

- No per-episode or time-aligned plots of logits, entropy or confidence.
- No model-output values in video overlays or filenames.
- No mechanism-classifier output or its scores (such as pad-face cover) for a held-out episode;
  the classifier is scored against the held-out labels, so they are written without it.
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

The held-out draw (`--n 30 --seed 1 --prefix heldout_`, with `--exclude` for earlier
selections) takes 3 failures per task and never draws an excluded episode: the pilot 20 and the
3 failures watched on video during the flag fixes (`t02-i005`, `t05-i002`,
`t08-i008`). It was committed before any held-out episode was watched. The held-out labels go in
`heldout_labels.csv`, committed only once all 30 are done.

## Computed vs hand-labeled

Anything the simulator can decide is computed; hand labels cover only judgment.

- **Computed** (pure functions over world-state fields extracted from the replayed
  `sim_state`): `k_completed`, `phase` and the world-state flags below. Predicates are
  LIBERO's own (`libero/libero/envs/predicates`); placement adds a release condition (below).
- **Hand-labeled from video:** `mechanism`, `mechanism_notes`, the two onset steps,
  `no_onset`, `notes`, and `computed_check`: the computed columns that disagree with the
  video, with a reason, or empty. Computed columns are validated this way on the pilot and the
  held-out labels; the computed row is shown only after the hand row is written.

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
- `phase` = the furthest phase reached on any subgoal still open when `k` was last at its
  maximum, whenever that progress was made.
- A failed episode never has all subgoals counting at once, so `k` is below the number of
  subgoals. Subgoals that each held at some point, but never together, do not add up.
- Only goal objects in their goal regions advance `k`. A goal object in a different region is
  `wrong_target`, never a milestone.
- A success episode has `k` equal to the number of subgoals by definition. The trace logs the
  state before each action, so the state the succeeding action produced is never recorded.

### Phases

- **grasped:** starts when the object, lifted at least 1 cm above its support (its lowest
  height over the last 20 steps: table, plate or caddy) and within 12 cm of the end effector
  with the fingers not open (width below 7 cm), moves at least 5 mm over a short window while
  its offset from the end effector stays fixed. It continues while the fingers stay closed and the offset
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
| 8 | pot 1 `On` stove, pot 2 `On` stove; stove `Turnon` (true at init) | none | grasped / carried |
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
- **Task 8 order:** resolved, no dependency. Both pots fit on the stove in either order
  (`t08-i022`). LIBERO's region `On` is the pot inside the cook region plus contact with the
  stove, checked every step, so success fires when the second pot touches down, held or not;
  settling both pots is never tested.
- **Stove indicator:** resolved; the coils turn red when on, so the video confirms `Turnon`.
- **Task 6:** the region right of the plate has no visible boundary; `carried` and placement
  for the pudding are judged from the printout, not the video.
- **Grasp at 256 px:** where a grasp is ambiguous, use the world-state printout (finger width
  vs object width, object height).

## Flags

All world-state flags are computed. Each threshold is set on the pilot 20 with a physical
rationale, swept as a sensitivity check (not searched), and frozen with the coding rules.
Each flag is a list of step intervals (Python slice notation, `;`-separated, a lone step for a
point event; empty means not fired), so its count is the number of entries.

| Flag | Definition | Threshold(s) |
|---|---|---|
| `wrong_object` | an object not named in the goal is grasped and moved more than D from its settled pose | D = 2 cm |
| `wrong_target` | a goal object, released and at rest for a dwell, satisfies its goal predicate against another receptacle | < 1 cm/s for 10 steps |
| `regression` | a counted subgoal later stops counting for N steps | N |
| `order_violation` | a dependent subgoal's predicate holds while its prerequisite does not count | N |
| `dropped` | a goal object is released away from its target and falls at least 3 cm within 10 steps; setting it down is not a drop | 3 cm, 10 steps |
| `collision_disturbance` | an object moves more than D while not held; each firing runs from the step the displacement passes D until the object rests. Excluded: the 10 steps before a grasp onset (closing fingers drag an object about 2 cm before it lifts), an unbroken contact run that ends in a grasp onset (a grip held before lift), the 10 steps after a release, and motion inside its goal container (a bowl in a closing drawer) | D = 2 cm |
| `regrasp_count` | grasp onsets on goal objects beyond the first per object, as point events | — |
| `stalled` | the arm stops: end-effector speed below v for at least 2 s; v = max(3 × median wait-step speed, 5 mm/s) | v, 2 s |
| `oscillation` | at least 6 direction reversals of at least 2 cm along one axis within 4 s. A path-to-displacement ratio does not work: reaching down and lifting has a high ratio (it flagged 112 of 130 successes). Named for what it measures: in control engineering "dither" is a deliberately injected signal, and the robot-learning term for the likely cause, mode switching, is internal to the policy | 6, 2 cm, 4 s |
| `gripper_chatter` | at least 3 empty closes (finger width below 2 cm) within 2 s with nothing held | 3, 2 s |
| `object_off_table` | a goal object at least 5 cm below its settled height | 5 cm |
| `workspace_limit` | an arm joint within 2% of its range from either limit for 10 steps; gripper fingers excluded. A fully extended arm shows as joint 4 at its upper limit, so no separate reach check | 2%, 10 steps |
| `truncated_mid_action` | a failure whose last logged step has a goal object grasped, or a goal object or fixture joint moving; a moving object counts only if the fingers were closed within reach of it in the last 5 s | 1 cm/s, 5 s |

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
never recovered from, whether or not success was still physically reachable after it. What follows the initiator (the failed recovery and the end state) goes in the flags and
`notes`, not in a second mechanism. A failure is read as an event tree: initiator, then the
recovery barrier (computed where possible, e.g. `regrasp_count`), then the end state.

Categories (frozen after the pilot):

| Category | Initiator |
|---|---|
| `perception` | the target is not localized: at a grasp, the target is not between the fingers at any point from the bottom of the descent through the close |
| `grounding` | the instruction is misread: wrong object or wrong target |
| `control: grasp` | right intent, failed execution, from heading for the object until it is lifted and held (computed `phase` none) |
| `control: transport` | from lift-off until the descent toward the target starts, including contact while moving into a receptacle (the microwave edge) (`phase` grasped) |
| `control: place` | from the descent toward the target through release until the object settles (`phase` carried) |
| `external` | a disturbance the policy did not cause (physics, a Block 3 perturbation); expected 0 on the baseline |
| `inaction` | the policy commands no motion from the first policy step to the cap (`stalled` throughout); onset 10 |

`perception` is not split by phase until a non-grasp case appears. Failing to recover after the
initiator is recorded in tags and flags, never as a mechanism.

- **Decidable from observables.** A category must be distinguishable on video or in world
  state. A grasp that closes off-center on the right object is `control`; `perception` needs
  visible evidence such as the arm going to where the object is not. For a grasp: if the target
  is between the fingers at any point from the bottom of the descent through the end of the
  close, a miss is `control`; if not, `perception`. A descent that never closes is judged at its
  bottom.
  The target is any part of the target object: successes grasp mugs by the wall or the handle,
  depending on the task. Between the fingers means at pad height, between the pads; an object
  under a finger, or below pads held up by a finger resting on it, is not between them.
  The target must sit in the gap between the two pad faces: one touched only by the edge or tip
  of a pad, so that the closing fingers push it aside or slide off it rather than squeeze it,
  is not between them. Reference cases: `t07-i013` (a sliver of the can at the pad edge,
  `perception`) and `t03-i009` (the bowl rim between the pads, `control: grasp`). A case
  between the two is labeled as judged, with "Close call with control" or "Close call with
  perception" in the notes.
- **Recovered** means progress resumed on the error's own subgoal: it counts later, or its
  object reaches a higher phase. Progress on another subgoal does not recover it. A miss is
  also recovered by a later close with the same object between the fingers; errors on
  non-goal objects (collisions) are recovered by progress on any subgoal.
- **`mechanism_notes`** describe the initiator only. Unverified causes are written as
  `Hypothesis: ...`, never as fact.
- **`notes`** hold what follows the initiator and anything else. Post-initiator behavior starts
  with a fixed phrase so it can be counted, e.g. `post: hover` (slow motion confined near one
  spot, neither stopped nor oscillating), `post: stall`, or `post: premature close` (a fixture
  moved toward closed before its prerequisite holds, short of a full close), or `post: jammed`
  (the arm or a carried object stays pinned against an obstacle while the arm keeps moving).
- Behavior before the initiator starts with `pre:`, reusing the `post:` phrases (`pre: stall`);
  an error the episode recovered from is `pre: recovered <error>` (`pre: recovered miss`).
  The tag vocabulary is closed for the held-out labels: `post:` hover, stall, jammed, premature
  close, miss; `pre:` the same words plus `recovered <error>`. Anything else goes in prose.
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

Logged as made; a later entry supersedes an earlier one, and labels already written are
re-passed when a change affects them.

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
- `t01-i022`: recovered means the robot resumed progress toward the goal, not that the
  obstruction went away; `post: jammed` covers the arm itself, not only a carried object.
- `t06-i013`: the context card gives each `On` subgoal's end state (holds or not; for
  object-on-object, distance from the target's center against the 3 cm tolerance), since
  a near-miss cannot be judged on video. Earlier rows already account for theirs.
- `t06-i020`: a subgoal completes only once its object is released (kept: the pudding was
  held in its region to the cap, so `k` = 0). Consistent with `t04-i017`, where the white mug
  was within 3 cm only while gripped (118-141) and off after release (147). Hypothesis: the
  policy rarely needs to release, since LIBERO ends an episode at success; 70 of 130 baseline
  successes end with a goal object still gripped (t05 17, t06 15, t02 14).
- `t08-i006`: `post: miss` for a failed grasp after the initiator, with the rule's reading
  (target between the fingers or not) in prose; mechanism categories are not tags.
  Goal predicates true at init stay subgoals, as in the BDDL goal (task 8's stove), so task 8
  failures start at `k` = 1.
- Post-pilot rule session (2026-10-09):
  - `control` split by phase (`grasp` / `transport` / `place`) at the freeze, before the held-out
    labels; the 16 pilot `control` rows are re-passed (8 grasp, 2 transport, 6 place). `grasp`
    starts when the arm heads for the object, so an empty-handed collision on the way is `grasp`
    (`t01-i022`). `recovery` renamed `external` and limited
    to initiators the policy did not cause.
  - `On`-tolerance near-misses are labeled like any other failure (`control: place`); no
    near-miss flag or relaxed success rate, since a single tolerance would be arbitrary across
    objects. Addable later as a pure function over stored world state.
  - Flag timing: the flag columns of `computed.csv` hold intervals instead of 0/1, in Python slice
    notation (end-exclusive), `;`-separated; a lone step is a point event (`145:175;276:417`,
    `265`). Empty means not fired; the interval count is the count. One shared parser.
  - Every flag defect below is fixed before the freeze, test-first from the episode it was
    caught on.
  - The mechanism classifier may abstain where world state cannot decide (close grasp misses);
    abstentions are hand-labeled. Abstention criteria are set on the pilot 20 and frozen before
    scoring the held-out set.
  - Freeze: thresholds confirmed from a sensitivity check on the pilot 20 (only those whose
    plausible change flips a flag against `computed_check`); closed tag vocabulary.
  - Known gaps, documented rather than ruled: no perception test for fixtures (no fixture error
    in the pilot); grasp-error onsets sit at the start, middle or bottom of the descent
    (`t00-i024` 15, `t08-i010` 80, `t08-i006` 110), a known source of onset spread.
- Classifier rule session (2026-10-09), amending the `block2-freeze` rules while held-out
  labeling is paused; held-out labels written before it are re-passed against it:
  - The grasp target is any part of the target object, not the part meant to be grasped.
    Successes grasp the porcelain mug by its wall (tasks 4, 6) and the white-yellow mug by
    its handle (task 9) or its far wall (task 4). No pilot label changes: in `t02-i009` and
    `t02-i023` no part of the pot is between the pads at any close.
  - Between the fingers means at pad height. `t08-i006` relabeled `perception` (was `control:
    grasp`): a finger stuck on the pot top kept the pads above the pot, as in `t02-i023` where
    a finger hit the handle. A test extended 3 cm past the pads (the fingers straddle the
    object) would keep `t08-i006` as `control` but turn `t02-i023` and `t07-i013` into
    `control` too.
  - The between-the-fingers test spans the bottom of the descent through the end of the close,
    not the bottom alone. No pilot label changes.
  - New category `inaction`: the arm never moves (`stalled` from step 11 to the cap). Seen in
    3 unlabeled non-held-out failures (`t03-i016`, `t07-i004`, `t07-i005`), none in the pilot.
    None of the other categories fits: no world state shows where the policy was aiming.
  - Recovered means progress on the error's own subgoal, not on any subgoal (see "Mechanism
    classifier", "Recovered").
- Held-out labeling (2026-10-10): the hand rule for between the fingers gets a visual minimum,
  anchored on reference cases (`t07-i013` below it, `t03-i009` above), matching the pilot labels
  and the classifier's pad-face cover cutoff of 0.3 (about 5 mm of a 1.6 cm pad, not visible at
  256 px). No label changes, pilot or held-out; the classifier is unchanged.

## Mechanism classifier

Assigns each failure a `mechanism` from world state, or abstains. Developed on the pilot 20,
scored against the held-out labels. Rules approved so far:

- **Rule order.** (1) `wrong_object` or `wrong_target` fired: `grounding`. (2) `stalled` from
  the first policy step to the cap: `inaction`, onset 10. (3) Otherwise find the initiator,
  the earliest error not recovered from. (4) The phase at the initiator, not the episode's
  final phase, gives the `control` sub-label: none `grasp`, grasped `transport`, carried
  `place`. (5) At a failed grasp attempt, the between-the-fingers test splits `perception`
  from `control: grasp`.
- **Recovered.** Progress resumed on the error's own subgoal: it counts later, or its object
  reaches a higher phase (held, then held over the target). Progress on another subgoal does
  not recover it (`t04-i017`: the second mug's placement does not undo the first mug's
  near-miss). A miss is also recovered by a later close with the same object between the
  fingers (`t01-i010`, "pre: recovered miss"); a later open descent over it is not enough
  (`t08-i006`). Errors on non-goal objects (collisions) are recovered by progress on any
  subgoal.
- **Abstention.** No initiator found: reported as an abstention with that reason, separate
  from close calls. On the 70 unlabeled non-held-out failures: 14, by pattern mid-air stalls
  away from every goal object, an object held to the cap without placing, and fixture-stage
  errors (no fixture detector).
- **Between the fingers.** Some part of the target object's collision geometry (boxes, from
  `object_geometry.json`, placed by `object_pos` and `object_quat`) lies in the box between
  the finger pads (`finger_pad_pos`, `grip_quat`, `finger_pad_half_size`). Score: pad-face
  cover, the share of the pad face whose closing-axis line passes through the object. Taken
  as the maximum from the bottom of the descent through the end of the close: 12 of 192
  successful grasping closes have nothing between the pads at the bottom and are aligned
  during the close. Cover below 0.3 is `perception`; for the moka pot any cover counts, since
  7 of its 30 successful handle grasps cover less than 0.3 (minimum 0.01). Other objects'
  successful grasps fall below 0.3 in 2 of 162. Matches all 8 pilot grasp-stage labels; no
  abstention band. Rejected: cover at the end of the close (a miss closes on nothing, so
  `t03-i009`, whose rim was between the pads at 0.52, would read as `perception`), and a
  top-down view past the pads (the straddle test above).
- **Slips are `control: grasp` in any phase.** A drop where the object leaves the hand (5 mm
  from the pad midpoint) before the fingers open (pad gap +2 mm) is a slip, unless a collision
  initiator precedes it (`t09-i020`). Fingers opening first or on the same step is a release,
  labeled by its phase (`t04-i004`). Matches both pilot slips (`t00-i013`, `t04-i002`) and two
  checked on video (`t03-i002`, `t07-i019`). At 20 Hz, ties and 2-step margins occur; the
  cutoffs go through the sensitivity check.
- **Completed-subgoal initiators.** A placement whose end state is outside the successes'
  range (tilted, raised, off-center) and blocks a dependent subgoal is a `control: place`
  initiator even though `k` counted it (`t03-i011`). Checked only for placements a later
  subgoal depends on (tasks 3 and 9), so basket drops never are. Defective: tilt (the
  object's settled up axis at step 10 against its axis 10 steps after the placement first
  holds) or rest height outside that task's successes for the object. Task 3 bowl successes
  tilt 0–23.8°; `t03-i011` is at 32.7°. No upright requirement.
- **Sensitivity** (each numeric parameter at x0.5 and x1.5, pilot 20 and the 70 unlabeled
  non-held-out failures): the defaults agree with all 20 pilot labels. Three flip a pilot
  value in one direction only, with at least 50% margin in the other: `cover_cutoff` 0.3
  (0.15 turns `t07-i013` into `control`), `near_m` 12 cm (6 cm drops the four "beside the
  object" `perception` attempts), `shift_m` 5 mm (2.5 mm turns `t06-i020` into
  `transport`). No other parameter flips a pilot value or changes more than 3 of the 70.
- **Onsets.** Slip: the step the object leaves the hand. `perception` and `control: grasp`: the
  bottom of the descent of the first failed attempt. `transport`: the first step the held
  object shifts more than 5 mm relative to the pad midpoint. `place`: the start of the
  descent that brought the object to rest on or near the target, before release or, for an
  object held to the cap, before it last left the target (not the first held-over-target
  step, which near-miss placements never reach). Collision initiators: the collision step. Pilot offsets
  are reported signed and not fitted; hand onsets for grasp errors sit anywhere in the
  descent.
- Grip-site geometry uses simulator-state fields only; `eef_pos` and `gripper_qpos` are
  observations one 2 ms physics substep behind.

## Freeze

Rules, tags and detector definitions are frozen at git tag `block2-freeze` for the held-out
labels. Threshold sensitivity on the pilot 20 (each threshold at x0.5 and x1.5, all 120
failures recomputed): the defaults agree with every pilot hand check. Three sit where either
direction flips a pilot value and were confirmed as they are: `collision_disturbance` distance
2 cm, `stalled` duration 2 s, open-finger width 7 cm. Every other threshold has at least 50%
margin on one side.

The classifier rule session (see "Rule changes") amended the hand-labeling rules: grasp target,
between the fingers at pad height through the close, `inaction`, per-subgoal recovery. Flags,
milestones and thresholds are unchanged. Rules, tags and the classifier are frozen at
`block2-classifier-freeze`; held-out labels written before it are re-passed against it.

Not validated on the pilot (no firings in the 20): `wrong_object`, `wrong_target`,
`regression`, `order_violation`, `regrasp_count`, `oscillation`, `gripper_chatter`,
`object_off_table`. They are frozen as defined and checked through the held-out
`computed_check` where they fire. Across all 120 failures they fire 1, 0, 3, 0, 3, 0, 0 and
2 times.

## Flag defects (fixed before the freeze)

- `truncated_mid_action`: fixed. A goal object moving at the cap counts only if the fingers were
  closed within reach of it in the last 5 s (`t00-i024` never closed near the cans, 0;
  `t09-i020` closed near the mug 5.0 s before the cap, 1). Any window of 5 s or more gives the
  same result on the 250 baseline episodes.
- `collision_disturbance` on a grip before lift: fixed. The detector still needs a lift
  (`t09-i015`: fingers on the mug handle from ~130, lift at 356), but the unbroken contact run
  that ends in a grasp onset is excused. 55 -> 45 firings on 250 episodes; the 10 switched
  episodes were checked on video, none a collision.
- `stalled`: resolved without a change. The disputed arms (`t03-i011` ~1.9 mm/s, `t06-i013`
  ~1-2 mm/s) were as still as agreed stalls (`t06-i020`, `t07-i007`); what read as motion was
  pressing against an obstacle, recorded by `post: jammed`. A threshold cap (8 or 10 mm/s) flips
  agreed stalls (`t03-i009`, `t08-i010`) and leaves `t03-i011` unchanged. `stalled` means the end
  effector is nearly stopped, whether or not it presses on something.
- `regrasp_count`: fixed at the grasp detector. Lift is measured from the object's lowest height
  over the last 20 steps, not its settled table height, so contact with an object resting on a
  plate or the caddy is no longer a grasp (`t03-i011`, `t05-i009`, `t06-i020`; 16 other onsets
  removed, 8 checked on video, none a grasp). Known undercount: a release and regrip while the
  object stays supported is one hold (`t06-i013`, ~278-285).
- `phase`: fixed. It is the furthest phase reached on any subgoal still open when `k` was last
  at its maximum, whenever that progress was made (`t04-i017`: carried). 7 of 120 failures
  change, all from `none`; checked on video.

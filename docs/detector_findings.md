# Detector findings: failure prediction from gripper-token entropy

**Verdict.** Gripper-token entropy over the 30 policy steps before the first close on each goal
object did not meet the pre-registered criterion on held-out init states 25–49: AUROC 0.56 (95% CI
0.49 to 0.63) includes 0.5. The signal is not confirmed, which is not the same as ruled out.

## The test

- **Detector** (`docs/detector_design.md`): at each first close command on a goal object, v1 takes
  the mean of exp(H) of the gripper token over the preceding 30 policy steps and alarms above the
  level that 20% of successful development first attempts exceed. Inputs are policy outputs only.
- **Target**: whether that grasp attempt fails (no hold, or a slip), whether or not the episode
  recovers later. Labels come from world state, never from model outputs.
- **Protocol**: developed and frozen on the baseline 250 episodes (init states 0–24), evaluated
  once on 250 new episodes (init states 25–49) with no refit.
- **Pre-registered pass**: the AUROC CI excludes 0.5 and the excess-over-chance CI (catch rate
  minus false-alarm rate) excludes 0. With 80–90 failed first attempts expected, a true AUROC of
  0.62 would pass with roughly 60–80% probability.

## Result

110 failed and 237 successful first attempts; AUROC is same-task (failure-success pairs within a
(task, object) stratum, ties counted half); 95% CIs bootstrapped over episodes within (task,
episode outcome).

| | AUROC | caught | false alarms | excess over chance | pass |
|---|---|---|---|---|---|
| v1 (primary), development, in-sample | 0.62 | 28 of 88 | 53 of 266 (20%) | 0.12 | n/a |
| **v1 (primary), held-out** | **0.56 (0.49 to 0.63)** | 32 of 110 | 42 of 237 (18%) | 0.11 (0.02 to 0.21) | **no** |
| v2 (secondary), held-out | 0.57 (0.50 to 0.64) | 34 of 110 | 44 of 237 (19%) | 0.12 (0.03 to 0.22) | marginal |

- The threshold transferred: 18% of held-out successes alarmed against the 20% design budget.
- Excess over chance clears 0 at the operating point. It is reported, not claimed: the criterion
  needs both CIs, and choosing after the fact which one carries the verdict is a forking path. The
  same holds for v2, whose AUROC lower bound is 0.501.
- The held-out success rate was 44.0% (38.0% to 50.2%) against 52.0% for the baseline; the
  intervals overlap.

## Is the measurement sound

Fifteen held-out first attempts (five caught, five missed, five false alarms, distinct tasks) were
checked on video against their labels and traces (`docs/results/heldout/detector/review/review.md`).

- One label is wrong (about 7%; Wilson 95% about 1% to 30%): a successful grasp split by a brief
  finger reopen. Two missed failures are correct grasps followed by a stall, failures only because a
  hold requires a lift. Excluding all 9 such held-out attempts leaves v1 at 0.557 (0.484 to 0.630).
- The attempt boundary is the labeler's weak point. One-step close commands split a grasp or merge
  into a later close, and two of four sampled width-baseline firings were such merges, so the width
  baseline's lead times are not used.
- The labeler misses holds on the book (two attempts carried 29–32 cm and labeled failed), and one
  episode is a physics glitch.

## What the signal is

- **The gripper action is effectively binary.** OpenVLA emits one of 256 bins per action dimension,
  but the gripper was trained on {0, 1} and its probability sits on two tokens, so exp(H) is 1
  when certain and 2 at a coin flip, and is at most 2.01 on all but 8 of 203,788 logged gripper
  steps (maximum 2.32).
- **Half of all attempts are flat.** v1 is below 1.001 for 50% of failed and 56% of successful
  first attempts: no step in the window departs from a certain gripper token.
- **Alarms come from a few steps at the switch.** In every reviewed trace the score comes from 1–5
  uncertain steps, nearly all within 12 steps of the close command. One step with about 18% on the other
  token clears the threshold alone. Caught failures and false alarms look the same, in the trace
  and on video, and no reviewed failure showed a visible cue before the close.

## What it is not

- **Not a perception signal.** On development data, failures whose initiating event is the arm off
  target at the grasp are no flatter than other failures (46% flat, AUROC 0.61, 0.49 to 0.71);
  control-grasp failures are less flat (17%, n = 12) but the intervals overlap, and at a tighter
  onset match the two are 0.63 and 0.64 (`docs/results/heldout/detector/review/mechanism_split.md`).
  The weak signal does not follow the failure mechanism.
- **Not a test of entropy in general.** Motion-dimension entropy within retries (z and pitch,
  0.68–0.70 in development) was out of scope, as were every-step alerting and hidden states.

## Limitations

- One seed. A second seed flipped 5 of 10 outcomes at matched init states, so the seed-to-seed
  noise floor is unmeasured and may be large relative to these effects.
- The window and features were chosen on the development set; the in-sample 0.62 is optimistic by
  construction. Why it fell to 0.56 is not established.
- LIBERO has 50 init states per task and both halves are used, so no revised detector can be
  tested on unseen initial conditions.
- Sim only, one policy, one task suite.

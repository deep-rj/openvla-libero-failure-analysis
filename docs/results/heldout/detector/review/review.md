# Held-out review of the frozen detector (v1)

The evaluation (`../detector_eval.md`) checked on video and in the traces before it is read.
Scripts: `scripts/block4_review.py`; pure functions in `harness/review.py`.

## Label check on video

Five first attempts per stratum, distinct tasks, seed 0 (`sample.csv`): caught failures, missed
failures, and successes that alarmed. Each was watched for the attempt label, the close command's
placement, the width baseline's firing, and anything visible in the 30 steps before the close.
Traces: `outputs/heldout/review/stratum_{a,b,c}.png` (`block4_review.py plot`).

| stratum | episode | label | notes |
|---|---|---|---|
| caught | t00-i035 | correct | Closes on nothing (arm off target); normal approach. Width fires at +14. |
| caught | t01-i041 | **wrong** | Successful grasp. The fingers reopen briefly before the hold, so the attempt ends at +5 and is labeled failed. The episode fails later at the butter, whose descent is blocked by the milk carton (around step 240), toppling it. |
| caught | t02-i031 | correct | One finger hits the pot handle and blocks the descent; very short open/close flips. Fingers never close below 3 mm in the attempt. |
| caught | t07-i034 | correct | Fingers do not close around the can before the lift; repeated attempts follow. |
| caught | t08-i049 | correct | A one-step close command, fingers miss the handle, then a stall; the full close comes about 55 steps later inside the same attempt, so width's +72 is an artifact. |
| missed | t00-i031 | correct | First attempt misses; the episode recovers. Width fires at +14. |
| missed | t02-i034 | correct | Hits the handle; clear miss. |
| missed | t07-i026 | correct | Misaligned approach hits the adjacent can and sticks; stalls. The scored close (130) is one step and a later close is merged into the attempt, so width's +71 is an artifact. An earlier close (118) is labeled aborted, so the scored attempt is the second close. |
| missed | t08-i031 | by rule | Clean grasp on the handle, then a stall. No hold registers without a lift. |
| missed | t09-i029 | by rule | Clean grasp on the handle, then a stall. |
| false alarm | t01-i045 | correct | Clean grasp. |
| false alarm | t04-i046 | correct | Clean grasp (handle-held mug). |
| false alarm | t06-i047 | correct | Clean grasp. |
| false alarm | t07-i037 | correct | Clean grasp. |
| false alarm | t08-i044 | correct | Precarious but successful hold; the episode fails at placement, the pot unbalanced on the stove. |

- 1 of 15 labels is wrong (Wilson 95% interval about 1% to 30%); 2 are failures only because a
  hold requires a lift. Close commands sit within ±10 steps of the visible close, except where a
  one-step command precedes the real close.
- The attempt boundary is the weak point: one-step command flips split a grasp (t01-i041) or merge
  into a later close (t08-i049, t07-i026). Two of four sampled width firings are merge artifacts, so
  the width-delay medians in `../detector_eval.md` are not reliable as lead times.
- Outside the sample: `t05-i046` and `t05-i035` are labeled failed although the book is carried
  29–32 cm (holds missed on the book), and `t01-i042` is a physics glitch (the cream cheese leaves
  the frame).

## What drives an alarm

In the 10 alarmed traces gripper exp(H) is 1.000 except for 1–5 steps, nearly all within 12 steps
of the close command. The gripper is effectively a two-token choice, so exp(H) lies in [1, 2], and one step at
exp(H) ≥ 1.61 (about 18% on the other token) clears the threshold on its own. Caught failures and
false alarms have the same pattern and nothing visible before the close; misses have no such step in
the window, or have it on the command step, which the window excludes (t09-i029).

| v1 | failed (110) | successful (237) |
|---|---|---|
| flat (< 1.001) | 55 (50%) | 133 (56%) |
| above flat, below threshold | 23 (21%) | 62 (26%) |
| above threshold | 32 (29%) | 42 (18%) |

## Sensitivity: grasp-then-stall failures

`stall_sensitivity.md`: 9 of 110 failed first attempts are stalls by the rule in `harness.review`.
Excluding them, v1 is 0.557 (0.484 to 0.630), excess 0.120 (0.017 to 0.219); the verdict is
unchanged. The labels stand as frozen.

## Mechanism split (development, in-sample)

`mechanism_split.md`: perception failures are no flatter than other failures (46% flat against 56%
for successes; AUROC 0.61, 0.49 to 0.71). Control: grasp failures are less flat (17%, n = 12) but
the intervals overlap, and at ±3 steps the two are 0.63 and 0.64. The weak signal does not follow
the failure mechanism.

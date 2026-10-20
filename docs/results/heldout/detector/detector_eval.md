# Detector evaluation

First attempts: 110 failed or slipped, 237 successful. Frozen `docs/results/baseline/detector/frozen_detector.json` (harness `539f610`): gripper window 30, roll window 100, budget 20%. Same-task AUROC and excess over chance (catch rate minus false-alarm rate) with 95% CIs, bootstrap over episodes within (task, episode outcome), 2000 draws, seed 0. Pass: both CIs exclude chance. Evaluated at harness `cf6b134`.

| | AUROC | caught | false alarms | random alarm | excess over chance | width also (later, median) | width never | pass |
|---|---|---|---|---|---|---|---|---|
| v1 (primary) | 0.56 (0.49 to 0.63) | 32 | 42 (18%) | 19.5 | 0.11 (0.02 to 0.21) | 12 (21 steps) | 20 | no |
| v2 (secondary) | 0.57 (0.50 to 0.64) | 34 | 44 (19%) | 20.4 | 0.12 (0.03 to 0.22) | 16 (20 steps) | 18 | yes |

Width baseline on the same first attempts: catches 46, false alarms 0.

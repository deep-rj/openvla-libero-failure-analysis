# Detector evaluation (Block 4)

First attempts: 88 failed or slipped, 266 successful. Frozen `docs/results/baseline/block4/frozen_detector.json` (harness `1e4207a`): gripper window 30, roll window 100, budget 20%. Same-task AUROC and excess over chance (catch rate minus false-alarm rate) with 95% CIs, bootstrap over episodes within (task, episode outcome), 2000 draws, seed 0. Pass: both CIs exclude chance. Evaluated at harness `1e4207a`.

| | AUROC | caught | false alarms | random alarm | excess over chance | width also (later, median) | width never | pass |
|---|---|---|---|---|---|---|---|---|
| v1 (primary) | 0.62 (0.55 to 0.69) | 28 | 53 (20%) | 17.5 | 0.12 (0.02 to 0.22) | 13 (18 steps) | 15 | yes |
| v2 (secondary) | 0.65 (0.59 to 0.72) | 28 | 53 (20%) | 17.5 | 0.12 (0.02 to 0.22) | 16 (18 steps) | 12 | yes |

Width baseline on the same first attempts: catches 37, false alarms 3.

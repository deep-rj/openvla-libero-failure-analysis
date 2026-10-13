# Window sweep, first attempts (Block 4)

Window mean of exp(H) before the close command: gripper on all steps (v1), roll on steps where the no-motion bin is not in the top two (v2). Failed and slipped first attempts against successful ones. Same-task AUROC, 99.17% CIs, bootstrap over episodes within (task, episode outcome), 2000 draws, seed 0. Short: windows cut by the episode start. Harness `a829c6a`.

**gripper** (chosen: 30 steps)

| Window | AUROC | failed / successful | short |
|---|---|---|---|
| 5 | 0.58 (0.48 to 0.68) | 88 / 266 | 0 |
| 10 | 0.60 (0.51 to 0.69) | 88 / 266 | 0 |
| 20 | 0.61 (0.52 to 0.71) | 88 / 266 | 0 |
| 30 | 0.62 (0.52 to 0.72) | 88 / 266 | 3 |
| 50 | 0.61 (0.52 to 0.71) | 88 / 266 | 48 |
| 75 | 0.58 (0.48 to 0.69) | 88 / 266 | 140 |
| 100 | 0.56 (0.45 to 0.66) | 88 / 266 | 172 |

**roll** (chosen: 100 steps)

| Window | AUROC | failed / successful | short |
|---|---|---|---|
| 5 | 0.50 (0.29 to 0.72) | 29 / 102 | 0 |
| 10 | 0.56 (0.40 to 0.72) | 44 / 146 | 0 |
| 20 | 0.56 (0.44 to 0.67) | 65 / 207 | 0 |
| 30 | 0.58 (0.48 to 0.69) | 72 / 238 | 3 |
| 50 | 0.60 (0.49 to 0.70) | 85 / 264 | 48 |
| 75 | 0.63 (0.53 to 0.72) | 87 / 265 | 140 |
| 100 | 0.63 (0.53 to 0.73) | 88 / 265 | 172 |
| 150 | 0.63 (0.52 to 0.72) | 88 / 265 | 205 |
| 200 | 0.60 (0.49 to 0.70) | 88 / 265 | 275 |

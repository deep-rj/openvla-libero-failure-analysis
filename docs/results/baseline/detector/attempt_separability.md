# Attempt-level separability (exploratory, Block 4)

Window mean of exp(H) over the 50 policy steps before each attempt's close command. Failed and slipped attempts against successful ones, aborted excluded. Same-task AUROC, failures above successes, 99.17% CIs, bootstrap over episodes within (task, episode outcome), 2000 draws, seed 0. Harness `aac0fe3`.

Attempt index on its object as the score, every attempt: 0.83 (0.71 to 0.88).

**every attempt** (1000 failed, 471 successful)

| Dim | all steps | stop not in top two |
|---|---|---|
| x | 0.50 (0.40 to 0.60) | 0.61 (0.50 to 0.71) |
| y | 0.55 (0.43 to 0.68) | 0.63* (0.53 to 0.74) |
| z | 0.68* (0.57 to 0.76) | 0.74* (0.64 to 0.80) |
| roll | 0.56 (0.47 to 0.65) | 0.63* (0.54 to 0.71) |
| pitch | 0.64* (0.54 to 0.74) | 0.70* (0.62 to 0.76) |
| yaw | 0.54 (0.44 to 0.64) | 0.66* (0.57 to 0.74) |
| gripper | 0.68* (0.58 to 0.76) | - |

**first per object** (88 failed, 266 successful)

| Dim | all steps | stop not in top two |
|---|---|---|
| x | 0.57 (0.45 to 0.68) | 0.56 (0.44 to 0.67) |
| y | 0.56 (0.45 to 0.66) | 0.55 (0.44 to 0.65) |
| z | 0.51 (0.41 to 0.62) | 0.49 (0.40 to 0.60) |
| roll | 0.53 (0.43 to 0.64) | 0.60 (0.49 to 0.70) |
| pitch | 0.56 (0.45 to 0.66) | 0.56 (0.45 to 0.67) |
| yaw | 0.52 (0.41 to 0.63) | 0.57 (0.47 to 0.68) |
| gripper | 0.61* (0.52 to 0.71) | - |

**first per object, failures in failed episodes only** (73 failed, 266 successful)

| Dim | all steps | stop not in top two |
|---|---|---|
| x | 0.56 (0.44 to 0.68) | 0.57 (0.45 to 0.68) |
| y | 0.56 (0.44 to 0.67) | 0.53 (0.41 to 0.63) |
| z | 0.49 (0.38 to 0.59) | 0.48 (0.36 to 0.58) |
| roll | 0.53 (0.42 to 0.64) | 0.60 (0.49 to 0.71) |
| pitch | 0.58 (0.47 to 0.70) | 0.58 (0.46 to 0.70) |
| yaw | 0.54 (0.42 to 0.66) | 0.58 (0.47 to 0.70) |
| gripper | 0.62* (0.51 to 0.73) | - |

**retries** (912 failed, 205 successful)

| Dim | all steps | stop not in top two |
|---|---|---|
| x | 0.46 (0.35 to 0.58) | 0.57 (0.42 to 0.69) |
| y | 0.56 (0.39 to 0.72) | 0.60 (0.46 to 0.72) |
| z | 0.64* (0.51 to 0.76) | 0.70* (0.57 to 0.80) |
| roll | 0.56 (0.44 to 0.67) | 0.58 (0.46 to 0.69) |
| pitch | 0.64 (0.48 to 0.77) | 0.68* (0.55 to 0.76) |
| yaw | 0.55 (0.41 to 0.67) | 0.64* (0.51 to 0.75) |
| gripper | 0.56 (0.40 to 0.68) | - |

**nearest the gate anchors** (63 failed, 192 successful)

| Dim | all steps | stop not in top two |
|---|---|---|
| x | 0.62 (0.49 to 0.75) | 0.60 (0.48 to 0.72) |
| y | 0.61* (0.50 to 0.73) | 0.62* (0.51 to 0.73) |
| z | 0.59 (0.47 to 0.70) | 0.55 (0.43 to 0.67) |
| roll | 0.59 (0.47 to 0.70) | 0.71* (0.60 to 0.80) |
| pitch | 0.60 (0.48 to 0.71) | 0.60 (0.48 to 0.72) |
| yaw | 0.59 (0.46 to 0.71) | 0.63* (0.52 to 0.75) |
| gripper | 0.52 (0.40 to 0.64) | - |

\* CI excludes 0.5.

# Separability gate

Units: 70 failures, 202 success grasps. Same-task AUROC is decisive; pooled is reported beside it. Bootstrap over episodes within (task, outcome), 10000 draws, seed 0. Harness `91848ce`.

| Metric | CI level | Same-task AUROC [CI] | Pooled AUROC [CI] |
|---|---|---|---|
| **gripper exp(H)** (primary) | 0.9500 | 0.438 [0.350, 0.527] | 0.463 [0.389, 0.536] |
| x exp(H) | 0.9917 | 0.627 [0.523, 0.732] | 0.534 [0.448, 0.622] |
| y exp(H) | 0.9917 | 0.621 [0.511, 0.730] | 0.527 [0.431, 0.623] |
| z exp(H) | 0.9917 | 0.583 [0.465, 0.696] | 0.580 [0.490, 0.667] |
| roll exp(H) | 0.9917 | 0.560 [0.439, 0.677] | 0.529 [0.439, 0.622] |
| pitch exp(H) | 0.9917 | 0.619 [0.510, 0.725] | 0.596 [0.509, 0.682] |
| yaw exp(H) | 0.9917 | 0.587 [0.469, 0.701] | 0.534 [0.448, 0.618] |
| gripper near-tie rate (secondary) | 0.9500 | 0.439 [0.371, 0.507] | 0.469 [0.414, 0.523] |

**Gate: pass.** P1 (failures higher on gripper exp(H)): not supported.

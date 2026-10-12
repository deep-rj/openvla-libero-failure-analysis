# Separability gate

Units: 70 failures, 202 success grasps. Same-task AUROC is decisive; pooled is reported beside it. Bootstrap over episodes within (task, outcome), 10000 draws, seed 0. Harness `f117928`.

| Metric | CI level | Same-task AUROC [CI] | Pooled AUROC [CI] |
|---|---|---|---|
| **gripper exp(H)** (primary) | 0.9500 | 0.438 [0.350, 0.527] | 0.462 [0.388, 0.536] |
| x exp(H) | 0.9917 | 0.632 [0.530, 0.738] | 0.536 [0.451, 0.625] |
| y exp(H) | 0.9917 | 0.629 [0.520, 0.739] | 0.530 [0.435, 0.626] |
| z exp(H) | 0.9917 | 0.589 [0.470, 0.703] | 0.581 [0.490, 0.669] |
| roll exp(H) | 0.9917 | 0.563 [0.443, 0.679] | 0.530 [0.440, 0.623] |
| pitch exp(H) | 0.9917 | 0.620 [0.511, 0.728] | 0.595 [0.508, 0.682] |
| yaw exp(H) | 0.9917 | 0.586 [0.469, 0.699] | 0.534 [0.447, 0.617] |
| gripper near-tie rate (secondary) | 0.9500 | 0.439 [0.371, 0.507] | 0.469 [0.414, 0.523] |

**Gate: pass.** P1 (failures higher on gripper exp(H)): not supported.

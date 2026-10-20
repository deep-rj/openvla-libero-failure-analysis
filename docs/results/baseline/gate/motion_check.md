# Motion check (exploratory, after the gate)

Same-task AUROC, failures above successes; bootstrap over episodes within (task, outcome), 10000 draws, seed 0. Speed is the end effector's from world state over the same pre-grasp window. Harness `143729b`.

| Arm speed in the window | AUROC [95% CI] |
|---|---|
| translation | 0.418 [0.332, 0.512] |
| rotation | 0.582 [0.489, 0.671] |

| Dim | Speed | r(exp(H), speed) | AUROC raw [99.17% CI] | AUROC speed-removed [99.17% CI] |
|---|---|---|---|---|
| x | translation | -0.15 | 0.627 [0.523, 0.732] | 0.612 [0.507, 0.719] |
| y | translation | 0.15 | 0.621 [0.511, 0.730] | 0.627 [0.516, 0.738] |
| z | translation | -0.22 | 0.583 [0.465, 0.696] | 0.565 [0.449, 0.677] |
| roll | rotation | 0.29 | 0.560 [0.439, 0.677] | 0.545 [0.435, 0.657] |
| pitch | rotation | 0.12 | 0.619 [0.510, 0.725] | 0.616 [0.506, 0.724] |
| yaw | rotation | 0.31 | 0.587 [0.469, 0.701] | 0.584 [0.469, 0.696] |

Speed-removed: exp(H) minus its least-squares line on speed, fitted once over all units.

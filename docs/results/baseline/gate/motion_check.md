# Motion check (exploratory, after the gate)

Same-task AUROC, failures above successes; bootstrap over episodes within (task, outcome), 10000 draws, seed 0. Speed is the end effector's from world state over the same pre-grasp window. Harness `299cf8e`.

| Arm speed in the window | AUROC [95% CI] |
|---|---|
| translation | 0.433 [0.346, 0.526] |
| rotation | 0.577 [0.485, 0.667] |

| Dim | Speed | r(exp(H), speed) | AUROC raw [99.17% CI] | AUROC speed-removed [99.17% CI] |
|---|---|---|---|---|
| x | translation | -0.15 | 0.632 [0.530, 0.738] | 0.624 [0.518, 0.730] |
| y | translation | 0.18 | 0.629 [0.520, 0.739] | 0.636 [0.526, 0.748] |
| z | translation | -0.21 | 0.589 [0.470, 0.703] | 0.574 [0.458, 0.684] |
| roll | rotation | 0.27 | 0.563 [0.443, 0.679] | 0.545 [0.433, 0.658] |
| pitch | rotation | 0.13 | 0.620 [0.511, 0.728] | 0.620 [0.510, 0.728] |
| yaw | rotation | 0.28 | 0.586 [0.469, 0.699] | 0.583 [0.467, 0.695] |

Speed-removed: exp(H) minus its least-squares line on speed, fitted once over all units.

# v1 by failure mechanism (development, in-sample)

A failed first attempt takes its episode's failure mechanism when the mechanism's onset is within the tolerance of the close command. AUROC is same-task against successful first attempts, 95% CI. Flat: v1 below 1.001.

Onset within ±10 steps:

| group | n | flat | alarm | AUROC |
|---|---|---|---|---|
| success | 266 | 56% | 20% |  |
| failed attempt, not the initiator | 35 | 40% | 34% | 0.63 (0.51 to 0.75) |
| perception | 26 | 46% | 31% | 0.61 (0.49 to 0.71) |
| failed attempt, successful episode | 15 | 53% | 20% | 0.55 (0.38 to 0.69) |
| control: grasp | 12 | 17% | 42% | 0.69 (0.52 to 0.82) |

Onset within ±3 steps:

| group | n | flat | alarm | AUROC |
|---|---|---|---|---|
| success | 266 | 56% | 20% |  |
| failed attempt, not the initiator | 45 | 38% | 38% | 0.63 (0.53 to 0.73) |
| perception | 19 | 47% | 26% | 0.63 (0.50 to 0.74) |
| failed attempt, successful episode | 15 | 53% | 20% | 0.55 (0.38 to 0.69) |
| control: grasp | 9 | 22% | 33% | 0.64 (0.44 to 0.80) |

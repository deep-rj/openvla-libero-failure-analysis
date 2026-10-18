# v1 with grasp-then-stall failures excluded

Stall: a failed first attempt whose fingers settle (width moving under 1 mm per step) between 3 and 66 mm for at least 30 steps on an object that rises under 2 cm (`harness.review`). 9 found: t02-i026, t03-i028, t06-i039, t06-i040 (alarm), t08-i031, t08-i032, t08-i040, t08-i043 (alarm), t09-i029.

| | failed | AUROC | caught | false alarms | excess over chance | pass |
|---|---|---|---|---|---|---|
| as frozen | 110 | 0.558 (0.488 to 0.629) | 32 | 42 | 0.114 (0.022 to 0.208) | no |
| stalls excluded | 101 | 0.557 (0.484 to 0.630) | 30 | 42 | 0.120 (0.017 to 0.219) | no |

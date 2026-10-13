# Width baseline (Block 4 reference)

Trigger: the close command nearest each closing gate anchor (within 10 steps). Fires at the first step of the attempt with finger width below the threshold; the attempt ends when the width exceeds its narrowest since the command by more than 2 mm. Chosen threshold 3 mm. No close command near the anchor: 1 failures, 4 successes. Harness `6d5013d`.

| Threshold | failures fired | successes fired | median delay (steps): failures | successes |
|---|---|---|---|---|
| < 1 mm | 12/60 | 1/198 | 36 | 49 |
| < 2 mm | 22/60 | 1/198 | 21 | 28 |
| < 3 mm (chosen) | 26/60 | 1/198 | 16 | 16 |
| < 4 mm | 28/60 | 1/198 | 15 | 13 |
| < 5 mm | 30/60 | 6/198 | 14 | 19 |
| < 6 mm | 31/60 | 23/198 | 13 | 13 |
| < 10 mm | 36/60 | 32/198 | 12 | 11 |
| < 15 mm | 38/60 | 49/198 | 10 | 10 |
| < 20 mm | 38/60 | 67/198 | 10 | 9 |
| < 30 mm | 42/60 | 97/198 | 9 | 8 |

Same-task AUROC of minimum width over the attempt (narrower above): 0.66, 99.17% CI 0.53 to 0.79, 2000 draws, seed 0.

**Narrowest successful grasp per object**

| Object | n | min (mm) | median (mm) |
|---|---|---|---|
| akita_black_bowl_1 | 10 | 4.1 | 5.0 |
| alphabet_soup_1 | 22 | 39.4 | 62.4 |
| black_book_1 | 17 | 29.2 | 29.4 |
| butter_1 | 19 | 1.0 | 38.8 |
| chocolate_pudding_1 | 15 | 12.6 | 46.0 |
| cream_cheese_1 | 29 | 17.7 | 42.1 |
| moka_pot_1 | 21 | 11.4 | 21.2 |
| moka_pot_2 | 8 | 11.7 | 23.2 |
| porcelain_mug_1 | 23 | 5.7 | 5.7 |
| tomato_sauce_1 | 13 | 37.3 | 60.9 |
| white_yellow_mug_1 | 21 | 9.8 | 18.4 |

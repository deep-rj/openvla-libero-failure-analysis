# Width baseline (detector reference)

Trigger: the close command nearest each closing gate anchor (within 10 steps). Fires at the first step of the attempt with finger width below the threshold; the attempt ends when the width exceeds its narrowest since the command by more than 5 mm. Chosen threshold 3 mm. No close command near the anchor: 1 failures, 4 successes. Harness `3ed94dc`.

| Threshold | failures fired | successes fired | median delay (steps): failures | successes |
|---|---|---|---|---|
| < 1 mm | 13/60 | 2/198 | 37 | 68 |
| < 2 mm | 22/60 | 2/198 | 21 | 49 |
| < 3 mm (chosen) | 27/60 | 2/198 | 16 | 40 |
| < 4 mm | 29/60 | 2/198 | 15 | 38 |
| < 5 mm | 31/60 | 8/198 | 14 | 44 |
| < 6 mm | 32/60 | 25/198 | 13 | 13 |
| < 10 mm | 36/60 | 33/198 | 12 | 11 |
| < 15 mm | 38/60 | 52/198 | 10 | 10 |
| < 20 mm | 38/60 | 70/198 | 10 | 9 |
| < 30 mm | 42/60 | 100/198 | 9 | 8 |

Same-task AUROC of minimum width over the attempt (narrower above): 0.65, 99.17% CI 0.52 to 0.79, 2000 draws, seed 0.

**Narrowest successful grasp per object**

| Object | n | min (mm) | median (mm) |
|---|---|---|---|
| akita_black_bowl_1 | 10 | 4.1 | 4.8 |
| alphabet_soup_1 | 22 | 39.4 | 62.4 |
| black_book_1 | 17 | 29.2 | 29.4 |
| butter_1 | 19 | 1.0 | 38.8 |
| chocolate_pudding_1 | 15 | 12.6 | 46.0 |
| cream_cheese_1 | 29 | 17.7 | 42.1 |
| moka_pot_1 | 21 | 11.4 | 15.4 |
| moka_pot_2 | 8 | 11.7 | 21.3 |
| porcelain_mug_1 | 23 | 5.7 | 5.7 |
| tomato_sauce_1 | 13 | 21.3 | 60.9 |
| white_yellow_mug_1 | 21 | 1.0 | 18.3 |

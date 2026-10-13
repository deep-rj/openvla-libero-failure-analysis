# Width baseline (Block 4 reference)

Trigger: the close command nearest each closing gate anchor (within 10 steps). Fires at the first step in the next 60 with finger width below the threshold; chosen threshold 3 mm. No close command near the anchor: 1 failures, 4 successes. Harness `5296fab`.

| Threshold | failures fired | successes fired | median delay (steps): failures | successes |
|---|---|---|---|---|
| < 1 mm | 14/60 | 1/198 | 38 | 49 |
| < 2 mm | 25/60 | 1/198 | 21 | 28 |
| < 3 mm (chosen) | 34/60 | 1/198 | 18 | 16 |
| < 4 mm | 37/60 | 1/198 | 19 | 13 |
| < 5 mm | 39/60 | 5/198 | 16 | 14 |
| < 6 mm | 41/60 | 23/198 | 15 | 13 |
| < 10 mm | 41/60 | 32/198 | 12 | 11 |
| < 15 mm | 45/60 | 49/198 | 11 | 10 |
| < 20 mm | 45/60 | 70/198 | 11 | 9 |
| < 30 mm | 48/60 | 96/198 | 10 | 8 |

Same-task AUROC of minimum width over the horizon (narrower above): 0.76, 99.17% CI 0.63 to 0.88, 2000 draws, seed 0.

**Narrowest successful grasp per object**

| Object | n | min (mm) | median (mm) |
|---|---|---|---|
| akita_black_bowl_1 | 10 | 4.3 | 5.8 |
| alphabet_soup_1 | 22 | 39.4 | 62.4 |
| black_book_1 | 17 | 29.4 | 29.4 |
| butter_1 | 19 | 1.0 | 38.8 |
| chocolate_pudding_1 | 15 | 12.6 | 46.0 |
| cream_cheese_1 | 29 | 17.7 | 42.1 |
| moka_pot_1 | 21 | 11.4 | 15.8 |
| moka_pot_2 | 8 | 11.7 | 21.7 |
| porcelain_mug_1 | 23 | 5.7 | 5.7 |
| tomato_sauce_1 | 13 | 37.3 | 60.9 |
| white_yellow_mug_1 | 21 | 9.8 | 18.3 |

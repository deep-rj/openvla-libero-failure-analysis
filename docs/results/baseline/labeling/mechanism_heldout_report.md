# Mechanism classifier: held-out report

Classifier frozen at `mechanism-classifier-freeze-2` (`c8ebd45`), scored once against the 30
held-out hand labels (`heldout_labels.csv`, committed in `6a79f8d`). Predictions written by
`mechanism_run.classify_run` on the held-out IDs.

- Agreement: 20/29 decided (69%).
- Abstentions: 1/30 (t05-i019).
- Onset offsets (predicted minus hand, steps): within ±10 for 18/29; sorted: [-64, -55, -54, -23, -18, -17, -16, -8, -5, -4, -3, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 3, 4, 6, 9, 15, 17, 19, 42].

## Per class (hand label)

| hand label | n | abstained | decided | agree |
|---|---|---|---|---|
| perception | 16 | 0 | 16 | 12 |
| control: grasp | 4 | 0 | 4 | 3 |
| control: transport | 3 | 0 | 3 | 2 |
| control: place | 5 | 1 | 4 | 1 |
| inaction | 2 | 0 | 2 | 2 |

## Confusion (decided)

| hand label | predicted | n |
|---|---|---|
| control: grasp | control: grasp | 3 |
| control: grasp | perception | 1 |
| control: place | control: place | 1 |
| control: place | control: transport | 3 |
| control: transport | control: grasp | 1 |
| control: transport | control: transport | 2 |
| inaction | inaction | 2 |
| perception | control: grasp | 4 |
| perception | perception | 12 |

## Per episode

| episode | hand | onset | predicted | onset | reason |
|---|---|---|---|---|---|
| t00-i003 | perception | 228 | control: grasp | 230 | grasp attempt on tomato_sauce_1: pad-face cover 0.67 |
| t00-i006 | perception | 358 | perception | 359 | grasp attempt on tomato_sauce_1: pad-face cover 0.00 |
| t00-i010 | perception | 298 | control: grasp | 296 | grasp attempt on tomato_sauce_1: pad-face cover 0.79 |
| t01-i005 | perception | 65 | perception | 68 | grasp attempt on cream_cheese_1: pad-face cover 0.17 |
| t01-i012 | control: grasp | 240 | control: grasp | 236 | grasp attempt on butter_1: pad-face cover 0.78 |
| t01-i024 | control: transport | 220 | control: grasp | 165 | grasp attempt on cream_cheese_1: pad-face cover 0.44 |
| t02-i006 | perception | 185 | control: grasp | 204 | grasp attempt on moka_pot_1: pad-face cover 0.01 |
| t02-i013 | inaction | 195 | inaction | 187 | stalled to the cap with no unrecovered error |
| t02-i020 | perception | 230 | perception | 207 | grasp attempt on moka_pot_1: pad-face cover 0.00 |
| t03-i006 | perception | 115 | perception | 114 | grasp attempt on akita_black_bowl_1: pad-face cover 0.00 |
| t03-i008 | perception | 270 | perception | 285 | grasp attempt on akita_black_bowl_1: pad-face cover 0.04 |
| t03-i022 | perception | 80 | perception | 81 | grasp attempt on akita_black_bowl_1: pad-face cover 0.00 |
| t04-i000 | control: place | 240 | control: transport | 224 | white_yellow_mug_1 shifted in the hand while carried |
| t04-i008 | control: place | 85 | control: transport | 68 | porcelain_mug_1 shifted in the hand while carried |
| t04-i019 | control: grasp | 200 | perception | 200 | grasp attempt on white_yellow_mug_1: pad-face cover 0.16 |
| t05-i011 | control: place | 172 | control: transport | 108 | black_book_1 shifted in the hand while carried |
| t05-i019 | control: place | 210 | abstain |  | no unrecovered error found |
| t05-i020 | control: place | 185 | control: place | 131 | black_book_1 placement did not count |
| t06-i006 | perception | 250 | perception | 267 | grasp attempt on chocolate_pudding_1: pad-face cover 0.00 |
| t06-i014 | perception | 190 | perception | 194 | grasp attempt on chocolate_pudding_1: pad-face cover 0.00 |
| t06-i016 | perception | 200 | perception | 197 | grasp attempt on chocolate_pudding_1: pad-face cover 0.00 |
| t07-i018 | perception | 190 | perception | 191 | grasp attempt on cream_cheese_1: pad-face cover 0.00 |
| t07-i021 | perception | 233 | perception | 232 | grasp attempt on cream_cheese_1: pad-face cover 0.16 |
| t07-i023 | control: grasp | 158 | control: grasp | 167 | alphabet_soup_1 left the hand before the fingers opened |
| t08-i001 | perception | 270 | control: grasp | 312 | grasp attempt on moka_pot_1: pad-face cover 0.01 |
| t08-i013 | control: grasp | 145 | control: grasp | 144 | grasp attempt on moka_pot_2: pad-face cover 0.38 |
| t08-i018 | perception | 335 | perception | 341 | grasp attempt on moka_pot_1: pad-face cover 0.00 |
| t09-i007 | inaction | 10 | inaction | 10 | stalled from the first policy step to the cap |
| t09-i011 | control: transport | 160 | control: transport | 142 | white_yellow_mug_1 shifted in the hand while carried |
| t09-i013 | control: transport | 225 | control: transport | 220 | white_yellow_mug_1 shifted in the hand while carried |

## Caveats

- Rules prompted by held-out episodes during labeling, so agreement on them is not
  independent: broadened `inaction` (`t02-i013`), grounding in event order (`t04-i019`), the
  fixture-stall abstention (designed knowing `t02-i013`), and the hand rule that an
  off-tolerance release is labeled by its end state (`t01-i024`, `t04-i000`).
- Hand onsets for grasp errors follow the bottom-of-descent convention from `t01-i024` on;
  the first four held-out rows were re-passed to it.

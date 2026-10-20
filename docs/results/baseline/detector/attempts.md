# Grasp attempts

An attempt is a close command within 15 cm of a goal object while none is held, running until the fingers reopen. Success: a hold on a goal object begins before then and does not end in a slip (the classifier's rule). Aborted: the fingers reopen before closing past 7 cm. Harness `3ed94dc`.

| Episode | success | slip | failure | aborted |
|---|---|---|---|---|
| successful (129 with attempts) | 331 | 8 | 146 | 17 |
| failed (110 with attempts) | 140 | 19 | 827 | 161 |

**Check against the gate anchors** (closing anchors; the attempt nearest the anchor, within 10 steps)

| Anchor | success | slip | failure | aborted | no attempt |
|---|---|---|---|---|---|
| failure | 0 | 9 | 46 | 5 | 1 |
| success | 192 | 4 | 2 | 0 | 4 |

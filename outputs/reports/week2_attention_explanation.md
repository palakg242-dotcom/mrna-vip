# Week 2 - What the attention heatmap shows

In self-attention, I first turned my 15 bases into numbers (an embedding matrix). From these, three versions are made for every base: Q (Query) is what that base is looking for, K (Key) is what it offers about itself, and V (Value) is the information it passes on. Multiplying Q and K gives a score for every pair of bases, showing how well they match. I divide these scores by √8 so they do not become too large. Softmax then turns each row into percentage-like numbers that add up to 1. Finally, I mix the V vectors using these numbers to get the output.

In the heatmap, each row is the base that is paying attention and each column is the base it is paying attention to. With the template's small weights (0.1), every cell is about 0.067, which is 1/15, so all bases get almost equal attention. The colours look dramatic only because the colour bar covers a tiny range. When I made the weights ten times larger, a clear pattern appeared: A and C mostly attended to U (about 0.3), and U and G attended to A. Bases that are the same (A1, A5, A9) have identical rows, because I gave the model no position information.

These preferences come from random initialisation, not from biology. A trained model would learn them from data.

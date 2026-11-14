# Week 2 - What the attention heatmap shows

I wrote self-attention from scratch in NumPy for 15 bases. Each base becomes a list of 8 numbers, and from it I make a Query (Q), a Key (K) and a Value (V). I think of Q as "what this base is looking for", K as "what it offers", and V as "the information it passes on".

I multiply Q of one base with K of another to get a match score. I divide the scores by the square root of 8 to keep them small, then apply softmax, which turns every row into numbers that add up to 1. I use these numbers to mix the V values into the output.

In the heatmap, each row is the base that is looking, and each column is the base it looks at. With the small starting weights (x0.1), every cell is between 0.065 and 0.069, almost exactly 1/15 = 0.067, so every base gives equal attention. The colours only look dramatic because the colour bar covers a tiny range.

With weights 10 times bigger, a pattern appeared: A and C mostly look at U (about 0.3), and G and U look at A. Identical bases, like A1, A5 and A9, have identical rows because the model has no position information.

This pattern comes from random starting weights, not from biology.

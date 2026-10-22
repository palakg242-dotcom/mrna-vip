# Week 2 - What the attention heatmap shows

Self-attention starts from the embedding matrix X (15 bases x 8 numbers). Three random weight matrices turn X into Q (what each base is looking for), K (what each base offers) and V (the content it passes on). Q·Kᵀ gives a 15x15 table of similarity scores, dividing by √8 keeps those scores small, softmax turns each row into probabilities that sum to 1, and multiplying by V blends the values using those probabilities.

The heatmap shows how much each base (row) attends to every other base (column). With the template's small weights (random numbers times 0.1), every cell lies between 0.065 and 0.069, almost exactly the uniform value 1/15 = 0.067, so the colour scale exaggerates tiny differences. Even so, a pattern exists: A and C rows lean slightly towards U, G and U rows lean towards A, and a base attends least to itself. Identical bases (A1, A5, A9 and so on) have identical rows, because the model has no position information, only base identity.

With projection weights ten times larger, the same preferences become strong (A and C attend to U with weight about 0.3). These preferences come purely from random initialisation, not from biology; a trained model would learn them from data.

# Week 3 - Retrieval test log

Model: `all-MiniLM-L6-v2`, top-k = 3, 3 chunks in the index.

| # | Question | Top-1 chunk | Similarity | Keyword in top-k | Latency (s) | Relevant? (check by hand) |
|---|---|---|---|---|---|---|
| 1 | What temperature should unopened mRNA vials be stored at? | chunk 0: Moderna mRNA Safety & Dosing Reference Guide (Course case-study reference for the mRNA-VIP... | 0.76 | yes, rank 1 | 0.010 |  |
| 2 | Can a vial be refrozen after it has thawed? | chunk 0: Moderna mRNA Safety & Dosing Reference Guide (Course case-study reference for the mRNA-VIP... | 0.48 | yes, rank 1 | 0.010 |  |
| 3 | Which dose needs secondary clinician review? | chunk 1: Product exhibiting visible particulates, cloudiness inconsistent with the expected formula... | 0.53 | yes, rank 1 | 0.009 |  |
| 4 | What should be done for a patient aged 65 or older with pre-existing conditions? | chunk 2: 3.1 Absolute Contraindications - Known severe allergic reaction (e.g., anaphylaxis) to a p... | 0.32 | yes, rank 2 | 0.010 |  |
| 5 | Within what time must a moderate adverse event be documented? | chunk 2: 3.1 Absolute Contraindications - Known severe allergic reaction (e.g., anaphylaxis) to a p... | 0.51 | yes, rank 1 | 0.009 |  |
| 6 | How long should a patient be monitored after the injection? | chunk 2: 3.1 Absolute Contraindications - Known severe allergic reaction (e.g., anaphylaxis) to a p... | 0.50 | yes, rank 1 | 0.009 |  |
| 7 | Is the vaccine safe to give during pregnancy? | chunk 1: Product exhibiting visible particulates, cloudiness inconsistent with the expected formula... | 0.31 | n/a (out of scope) | 0.008 |  |

Mean latency: 0.009 s, max: 0.010 s.

The 'Keyword in top-k' column is only an automatic hint. The last column must be filled in by reading the returned chunks.

## Manual relevance check

| # | Relevant? |
|---|---|
| 1 | Yes |
| 2 | Yes |
| 3 | Yes |
| 4 | Partly (right chunk at rank 2, not rank 1) |
| 5 | Yes |
| 6 | Yes |
| 7 | N/A (out of scope, wrong chunk returned) |
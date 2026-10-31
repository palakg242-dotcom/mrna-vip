# Week 3 - RAG write-up

## Chunk size and overlap
The guideline file is only about 755 words, so I used chunks of 300-500 tokens as FR-3.1 asks. This gave 3 chunks of 427, 414 and 348 tokens, measured with the embedding model's own tokenizer. Chunks are built from whole sentences, and each new chunk starts with the last sentence(s) of the previous one (at least 50 tokens), so a rule near a boundary is not cut in half. I raised the model's input limit from 256 to 512 tokens so that no chunk is truncated before embedding.

## Results
I asked 7 questions: 6 from the document and 1 deliberately not in it. For 5 of the 6 in-document questions the top-1 chunk contained the right passage. The index has only 3 chunks and I retrieve k = 3, so the top-3 always contains the whole document; top-1 is therefore the meaningful measure. Each query took about 0.01 s.

| # | Question | Top-1 chunk | Right passage | Relevant? |
|---|---|---|---|---|
| 1 | Storage temperature for unopened vials | 0 | Section 1.1 | Yes |
| 2 | Refreezing a thawed vial | 0 | Section 1 | Yes |
| 3 | Dose needing secondary review | 1 | Sections 2.1 / 3.2 | Yes |
| 4 | Patient aged 65+ with pre-existing conditions | 2 | Section 2.2 (found at rank 2) | No, only at rank 2 |
| 5 | Moderate adverse event documentation time | 2 | Section 4.1 | Yes |
| 6 | Monitoring time after injection | 2 | Section 3.3 | Yes |
| 7 | Safety during pregnancy (not in document) | 1 | none | No answer exists |

## Failure case
For question 4 the top-1 chunk (similarity 0.32) was chunk 2, which covers contraindications and adverse events, while the answer is in Section 2.2 and only appeared at rank 2. My hypothesis, which I have not tested, is that chunks of about 400 tokens mix several topics, so their embeddings are blurry, and the words "patient" and "pre-existing conditions" also match the contraindications section. Smaller section-based chunks would probably help, but they would break the 300-500 token rule.

## Out-of-scope question
Question 7 is not covered by the document, yet retrieval still returned a chunk (similarity 0.31). Retrieval never answers "not found", so the Week 4 prompt must tell the LLM to say so when the excerpts do not contain the answer.

# Week 3 - RAG Retrieval Write-up

## What I built
I built a retrieval system over the mRNA safety guidelines. The document is split into chunks, each chunk is converted into an embedding with the `all-MiniLM-L6-v2` model, and the embeddings are stored in ChromaDB. When I ask a question, the system returns the top 3 most similar chunks.

## Chunk size and overlap
FR-3.1 asks for chunks of 300-500 tokens, but the PRD template counts characters, not tokens (a `chunk_size` of 500 characters is only about 100 tokens). I therefore wrote my own chunker that builds chunks from whole sentences and counts tokens with the embedding model's own tokenizer. Each new chunk starts with the last sentence(s) of the previous one (at least 50 tokens), so a rule near a boundary is not cut in half. I also raised the model's input limit from 256 to 512 tokens so that no chunk is truncated before embedding.

The guideline file is short (about 750 words), so the final index has only 3 chunks of 427, 414 and 348 tokens. All of them are inside the required range.

## Results
I tested 7 questions: 6 from inside the document and 1 deliberately outside it. For 5 of the 6 in-scope questions the top-1 chunk contained the right passage; for the sixth (65+ patients) the right chunk was at rank 2. Mean latency was 0.009 s and the maximum 0.010 s, far below the 5 second limit in FR-3.5.

Because the index has only 3 chunks and I retrieve k = 3, the top-3 always contains the whole document. Top-1 is therefore the meaningful measure, and the hit rate at k = 3 looks better than it would on a larger document.

| # | Question | Top-1 chunk | Right passage | Relevant? |
|---|---|---|---|---|
| 1 | Storage temperature for unopened vials | 0 | Section 1.1 | Yes |
| 2 | Refreezing a thawed vial | 0 | Section 1.2 | Yes |
| 3 | Dose needing secondary review | 1 | Sections 2.1 / 3.2 | Yes |
| 4 | Patient aged 65+ with pre-existing conditions | 2 | Section 2.2 (found at rank 2) | Partial (right chunk at rank 2, not rank 1) |
| 5 | Moderate adverse event documentation time | 2 | Section 4.1 | Yes |
| 6 | Monitoring time after injection | 2 | Section 3.3 | Yes |
| 7 | Safety during pregnancy (not in document) | 1 | none | N/A (no answer exists) |

## Failure cases

**Question 4 (wrong chunk first).** The top-1 chunk (similarity 0.32) was chunk 2, which covers contraindications and adverse events, while the answer is in Section 2.2 and only appeared at rank 2. My hypothesis, which I have not tested, is that chunks of about 400 tokens mix several topics, so their embeddings are blurry, and the words "patient" and "pre-existing conditions" also match the contraindications section. Smaller section-based chunks would probably help, but they would break the 300-500 token rule.

**Question 7 (out of scope).** I asked "Is the vaccine safe to give during pregnancy?", which the guidelines do not cover. The system still returned chunk 1 (the dosing section) with a similarity of 0.31. Retrieval always returns the nearest chunk and never says "I don't know".

A simple similarity threshold would not fix this: the in-scope 65+ question scored 0.32 and the out-of-scope pregnancy question scored 0.31, which is almost the same.

## Planned fix
The LLM prompt must tell the model to answer only from the retrieved text and to reply "not covered in the guidelines" when the excerpts do not contain the answer. In Week 4 the rule to default to "not eligible" when information is insufficient covers part of this, but retrieval itself still never abstains.

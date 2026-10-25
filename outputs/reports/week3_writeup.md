# Week 3: RAG Retrieval Write-up

## What I built
I built a retrieval system over the mRNA safety guidelines. The document is split into chunks, each chunk is converted into an embedding with the all-MiniLM-L6-v2 model, and the embeddings are stored in ChromaDB. When I ask a question, the system returns the top 3 most similar chunks.

## Chunking choices
FR-3.1 asks for chunks of 300-500 tokens, but the PRD template counts characters, not tokens. A chunk_size of 500 characters is only about 100 tokens, so I wrote my own chunker that splits by sentence and counts tokens. I used a target of 300 tokens with an overlap of 50. The final index has 3 chunks of 427, 414 and 348 tokens, so all of them are inside the required range. The guidelines document is short, so only 3 chunks were created.

## Results
I tested 7 questions: 6 from inside the document and 1 from outside it. For 5 of the 6 in-scope questions, the correct chunk was at rank 1. For the sixth (patients aged 65 or older), the correct chunk was at rank 2. Mean latency was 0.009 s and the maximum was 0.010 s, far below the 5 second limit in FR-3.5.

Because the index has only 3 chunks and I retrieve the top 3, almost the whole document comes back for every question. This makes the hit rate look better than it would on a larger document.

## Failure case
I asked "Is the vaccine safe to give during pregnancy?", which the guidelines do not cover. The system still returned chunk 1 (the dosing section) with a similarity of 0.31. Retrieval always returns the nearest chunk and never says "I don't know".

A simple similarity threshold would not fix this. The in-scope 65+ question scored 0.32 and the out-of-scope pregnancy question scored 0.31, which are almost the same.

## Planned fix
In Week 4, I will instruct the LLM to answer only from the retrieved text and to reply "not covered in the guidelines" when the answer is missing.
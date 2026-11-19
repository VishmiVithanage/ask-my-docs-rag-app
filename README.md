# RAG over your own documents

Ask questions about your own `.md`, `.txt`, `.pdf` and `.docx` files and get answers
that **cite the passages they came from**.

```
python rag.py index docs/                  # build the index (re-runs only if files changed)
python rag.py ask "How many PTO days can I carry over?"
python rag.py chat                         # interactive
python app.py docs/                        # web UI at http://127.0.0.1:5000
python eval_rag.py docs/                   # measure retrieval quality
export ANTHROPIC_API_KEY=...               # optional: enables Claude-written answers
```

Without an API key the app still works: it shows the best-matching sentences with citations
(extractive mode). With a key, Claude writes the answer from the retrieved passages.
Set `RAG_MODEL` to override the model (default `claude-sonnet-5-5`).

## Architecture

```
files ─► load ─► chunk ─► index ─┬─ BM25 (lexical)  ─┐
                                 └─ LSA vectors      ─┴─► RRF fusion ─► confidence gate ─► prompt ─► Claude ─► citation check
```

## Why each method was chosen

### 1. Loading (`load_documents`)
**Choice:** per-format loaders that keep a *locator* (markdown heading path, PDF page, docx heading).
**Why:** retrieval can never be better than the text you extract, and a citation like
`security_policy.md (Backups and Access Reviews)` is what lets a person verify an answer instead
of trusting it. **Trade-off:** PDF chunks never span pages, so a fact split across a page break is
cut. Scanned PDFs have no text layer and need OCR first.

### 2. Chunking (`chunk_text`)
**Choice:** structure-first splitting (section → paragraph → sentence), ~900 characters, 150 overlap.
**Why:** one vector or score per whole document blurs unrelated facts together, and stuffing whole
documents into the prompt is expensive and buries the answer. Splitting on structure keeps each
chunk a coherent thought; overlap protects answers that straddle a boundary. The size is a
starting point. Too small loses context ("it", "this policy"); too large dilutes relevance.
Check with `python rag.py chunks`, and tune using the eval harness.

### 3. Lexical retrieval: BM25 (`Index._bm25`)
**Choice:** BM25, implemented directly (no library).
**Why:** questions about your own documents are full of exact terms: names, numbers, acronyms,
error codes ("MFA", "35 days", "HTTP 429"). Exact matching is the strongest, most explainable
signal for these, needs no model, and builds instantly. A small text normaliser (lowercase,
stopwords, light suffix stripping) is shared with the dense side so both see identical terms.
**Trade-off:** it cannot connect "keep" to "retained". The suffix stripper is crude; a real
stemmer or lemmatizer is better if you can install one.

### 4. Dense retrieval: LSA vectors (`LSADense`)
**Choice:** TF-IDF (uni+bigrams) → truncated SVD → unit vectors, cosine similarity.
**Why:** a second retriever that scores by *topic overlap* rather than spelling, so paraphrases
can still land. I chose LSA because it runs offline with only scikit-learn, which was all this
environment had (installs were blocked).
**Honest limit:** LSA only learns synonyms that co-occur in *your* documents. It is much weaker
than a neural embedding model, and on this small corpus it behaved almost identically to BM25.
It is a placeholder behind a clean seam (see "Upgrading").

### 5. Fusion: Reciprocal Rank Fusion (`Index.search`)
**Choice:** `score = Σ 1/(60 + rank)` across the two retrievers.
**Why:** BM25 scores are unbounded and cosine similarities lie in −1..1, so adding them needs
fragile normalisation. RRF uses only ranks, has no weights to tune, and rewards chunks that
both retrievers like. It is the standard, robust way to combine rankers.

### 6. Confidence gate (`Retrieval.confident`)
**Choice:** abstain unless ≥34% of the question's content terms appear in the retrieved text, or
the best dense similarity is ≥0.35. Abstaining skips the LLM entirely.
**Why:** the most damaging RAG failure is a fluent answer to a question the documents don't cover.
Declining early is the cheapest defence and also saves an API call. **Trade-off:** thresholds are
heuristics; a heavily paraphrased question can be wrongly refused. Tune `min_coverage` /
`min_dense` on your own questions.

### 7. Prompt design (`SYSTEM`, `build_prompt`)
**Choices and reasons:**
- "Use ONLY the passages" + an exact "I couldn't find that" fallback line → limits hallucination.
- Numbered passages + mandatory `[n]` citations → every claim is checkable.
- Passages wrapped in `<documents>` and declared *data, not instructions* → text inside a document
  ("ignore previous instructions…") shouldn't be obeyed. This reduces prompt injection; it does
  not eliminate it.
- Documents first, question last → works best for long context.
- Top-5 passages: enough to combine facts from a couple of sections, small enough to stay focused
  and cheap.

### 8. Generation over plain HTTP (`call_claude`)
**Why Claude:** it follows grounding and citation instructions reliably, and can merge facts from
several passages into one direct answer. **Why `requests` instead of the SDK:** the SDK wasn't
installed, and one POST is all that's needed. Swap in the official SDK freely.
**Not verified here:** there was no API key or network access in my environment, so the live
Claude call was **tested only with a mocked response**. The request format follows the public
Messages API; try it with your key and check the first few answers.

### 9. Output verification (`answer`)
**Choice:** after generation, check every `[n]` refers to a passage that was actually retrieved,
and warn if there are no citations at all.
**Why:** cheap, deterministic, and it catches fabricated references. It does **not** prove the cited
passage supports the claim; that needs an LLM judge or human spot-checks.

### 10. Graceful degradation
No key, network error, or API failure → extractive answer with a visible warning, rather than a
crash or silent hallucination. **Limit:** the extractive mode is loose; it can include a
related-but-unneeded sentence.

### 11. Storage and incremental indexing
**Choice:** one pickle file plus a SHA-256 manifest of source files; rebuild only if a file changed.
**Why:** at personal/team scale (thousands of chunks) a vector database is unnecessary complexity,
and brute-force search is instant. **Security note:** pickle runs code on load, so only open index
files you created. Move to a vector store (e.g. Chroma, pgvector) past roughly 100k chunks or when
you need concurrent updates.

### 12. Web UI (`app.py`)
Flask because it was installed and the UI is one page and one endpoint. Answers are rendered with
`textContent`, never `innerHTML`, to prevent XSS from document or model text. It binds to
`127.0.0.1` because there is **no authentication**: don't expose it to a network without adding some.

## What the evaluation showed (be careful with this)

`python eval_rag.py docs/` on the bundled 3 sample files (14 chunks, 13 questions, chunk-level):

| mode   | hit@1 | hit@3 | hit@5 |
|--------|-------|-------|-------|
| BM25   | 85%   | 92%   | 100%  |
| dense  | 85%   | 92%   | 100%  |
| hybrid | 85%   | 92%   | 100%  |

All three tie. **This corpus is far too small to show any benefit of hybrid retrieval**, and I did not
find one. I'm keeping hybrid because the reasoning (lexical and semantic signals fail on different
queries) is well established at larger scale, but you should confirm it on your own documents.
Two things the harness did show:
- The one hard question ("How long do you keep my analytics history?" vs "Event data is retained…")
  only reached rank 5 because no content words overlap. This is the vocabulary-gap problem.
- An earlier file-level version of the eval looked better than the truth (it counted a wrong chunk
  from the right file as a hit), so the harness scores at section level.

The unanswerable questions were all correctly refused (3/3), but they were off-topic. Near-miss
questions (on-topic but not covered) are the real test of the gate; add some of your own.

## Upgrading (in order of likely payoff)

1. **Neural embeddings** instead of LSA, implemented by giving `Index` a different dense backend
   with the same two methods (`fit(texts) -> matrix`, `encode_query(q) -> vector`):
   ```python
   class STDense:                      # needs: pip install sentence-transformers
       def __init__(self, name="BAAI/bge-small-en-v1.5"):
           from sentence_transformers import SentenceTransformer
           self.m = SentenceTransformer(name)
       def fit(self, texts):
           return self.m.encode(texts, normalize_embeddings=True)
       def encode_query(self, q):
           return self.m.encode(q, normalize_embeddings=True)
   ```
   (Untested here: the library wasn't available. Also exclude the model from pickling.)
2. **Reranker** (cross-encoder or an LLM) over the top 20 → keep 5. Biggest precision gain.
3. **Query rewriting** with Claude (expand "keep my history" → "retention period") to bridge vocabulary gaps.
4. **Evaluate on your own 30–50 questions** and tune chunk size, `top_k` and gate thresholds.
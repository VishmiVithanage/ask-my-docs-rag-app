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


         

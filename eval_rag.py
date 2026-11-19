# eval_rag.py - measures retrieval quality so design choices are evidence, not belief.

import sys

from rag import build_index

CASES = [
    ("How many days of paid time off do I get per year?", "Time Off", "exact"),
    ("How many vacation days do I get?", "Time Off", "paraphrase"),
    ("Can I work from home?", "Remote Work", "paraphrase"),
    ("How long is parental leave?", "Parental Leave", "exact"),
    ("From what amount are expense receipts required?", "Expenses", "exact"),
    ("How long do you keep my analytics history?", "Data Retention", "paraphrase"),
    ("What is the API rate limit on the Pro plan?", "API Limits", "exact"),
    ("Do you support SAML SSO?", "Single Sign-On", "exact"),
    ("How much does the Basic plan cost?", "Plans and Pricing", "exact"),
    ("What do I do if I think my account was hacked?", "Incident Reporting", "paraphrase"),
    ("How often are backups taken?", "Backups and Access Reviews", "exact"),
    ("Is two-factor authentication required?", "Accounts and Passwords", "paraphrase"),
    ("How long must a password be?", "Accounts and Passwords", "exact"),
]

UNANSWERABLE = [
    "What is the capital of France?",
    "Who won the 2022 World Cup?",
    "What is the CEO's favourite colour?",
]


def rank_of(idx, q, section, mode, k=5):
    hits = idx.search(q, k=k, mode=mode).hits
    for r, h in enumerate(hits, 1):
        if h.chunk.locator.endswith(section):
            return r
    return None


def main(folder: str):
    idx, _ = build_index(folder, path="eval_index.pkl", force=True)
    n = len(CASES)
    print(f"{len(idx.chunks)} chunks from {len(idx.manifest)} files, {n} questions\n")
    print(f"{'mode':8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6}")
    for mode in ("bm25", "dense", "hybrid"):
        ranks = [rank_of(idx, q, s, mode) for q, s, _ in CASES]
        cells = [sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)]
        print(f"{mode:8} " + " ".join(f"{c:6.0%}" for c in cells))

    print("\nPer-question rank of the correct section (hybrid; '-' = not in top 5):")
    for q, s, kind in CASES:
        r = rank_of(idx, q, s, "hybrid")
        print(f"  rank {r or '-'}  [{kind:10}] {q}")

    print("\nUnanswerable questions (should abstain):")
    ab = 0
    for q in UNANSWERABLE:
        r = idx.search(q)
        ab += not r.confident
        print(f"  {'abstained' if not r.confident else 'ANSWERED '}  {q}  "
              f"(coverage {r.coverage:.0%}, dense {r.best_dense:.2f})")
    print(f"abstained on {ab}/{len(UNANSWERABLE)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs")
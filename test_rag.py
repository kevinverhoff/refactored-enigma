"""
RAG pipeline demo -- run this to see how the pipeline works.

Shows:
  1. Basic Q&A with inline citations
  2. Peeking at retrieved sources (what the LLM actually sees)
  3. Filtered queries (year, doc_type, author)
  4. Retrieval-only (no generation) for browsing
  5. A question outside the corpus (honest no-answer behavior)

Run: python test_rag.py
"""

from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
print(f"Pipeline ready -- {pipeline.collection.count():,} chunks indexed\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 68
THIN = "-" * 68


def ask(label: str, query: str, **kwargs):
    """Ask a question and print the answer with source summary."""
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Q: {query}")
    if kwargs:
        print(f"  Filters: {kwargs}")
    print(SEP)

    result = pipeline.answer(query, **kwargs)
    print(result.answer)
    print(f"\n{THIN}")
    print(f"  {result.n_retrieved} chunks retrieved | {result.model}")
    for i, src in enumerate(result.sources, 1):
        print(f"  [{i}] {src['title'][:55]:<55} {src['author']:<14} {src['memo_date']}  score={src['score']}")


def browse(label: str, query: str, **kwargs):
    """Retrieve chunks and print them without calling the LLM."""
    print(f"\n{SEP}")
    print(f"  {label}  (retrieval only -- no LLM)")
    print(f"  Q: {query}")
    if kwargs:
        print(f"  Filters: {kwargs}")
    print(SEP)

    chunks = pipeline.retrieve(query, **kwargs)
    if not chunks:
        print("  No results.")
        return
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['score']}  {c['doc_type']:<10} {c['memo_date']}  {c['author']}")
        print(f"       {c['title'][:65]}")
        print(f"       ...{c['text'][:120].strip()}...")
        print()


# ---------------------------------------------------------------------------
# 1. Basic Q&A
#    Plain question -- no filters. Shows the full answer + citation list.
# ---------------------------------------------------------------------------

ask(
    "Basic Q&A",
    "What are the eligibility requirements for the homestead deduction?",
)

# ---------------------------------------------------------------------------
# 2. Recent guidance only
#    Year filter keeps results to 2025 and later.
#    Useful for "what changed recently?" questions.
# ---------------------------------------------------------------------------

ask(
    "Recent guidance only  (source_year >= 2025)",
    "What has changed recently with homestead deductions or credits?",
    year=2025,
)

# ---------------------------------------------------------------------------
# 3. Filter by doc type -- find forms, not memos
#    When you want the actual petition/template rather than the policy memo.
# ---------------------------------------------------------------------------

ask(
    "Templates only  (doc_type = TEMPLATE)",
    "How do I file an excess levy appeal?",
    doc_type="TEMPLATE",
)

# ---------------------------------------------------------------------------
# 4. Filter by author
#    Scope a question to one DLGF staffer's memos.
# ---------------------------------------------------------------------------

ask(
    "Author filter  (author = Van Dorp)",
    "What are the requirements for a school bus replacement plan?",
    author="Van Dorp",
)

# ---------------------------------------------------------------------------
# 5. Retrieval only -- no LLM
#    Use pipeline.retrieve() directly when you want to browse matches
#    without paying for a generation call, or when feeding into your own prompt.
# ---------------------------------------------------------------------------

browse(
    "Retrieval only -- no generation",
    "agricultural land base rate methodology",
    n=4,
)

# ---------------------------------------------------------------------------
# 6. Out-of-corpus question
#    Ask something the memos don't cover. The LLM should say it doesn't know
#    rather than making something up -- this tests grounding.
# ---------------------------------------------------------------------------

ask(
    "Out-of-corpus question  (should return honest no-answer)",
    "What are the requirements to register to vote in Indiana?",
)
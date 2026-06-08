# RAG Pipeline Design
**Date:** 2026-06-08
**Status:** Approved

---

## Context

The DLGF Memo Intelligence Pipeline has a Chroma vector store (13,589 chunks across 475
documents, embedded with Gemini gemini-embedding-001). The next step is a RAG pipeline that
takes a natural language query, retrieves relevant chunks, and generates a grounded answer
with inline citations.

The LLM is Gemini, hardcoded. Every call site is annotated so anyone forking the project can
find and swap the model without searching the codebase. The README will include a reference
table of all LLM call sites.

---

## Files

### `rag_pipeline.py` (new)

Single file. No provider abstraction layer.

### `README.md` (update)

New "LLM Reference" section listing every model call site across the project.

---

## `rag_pipeline.py`

### Constants

```python
EMBED_MODEL      = "gemini-embedding-001"   # LLM CALL SITE: query embedding
GENERATION_MODEL = "gemini-2.0-flash"       # LLM CALL SITE: answer generation
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "dlgf_memos"
DEFAULT_N        = 6
```

### `RAGResponse` dataclass

```python
@dataclass
class RAGResponse:
    answer: str                  # LLM-generated answer with inline citations
    sources: list[dict]          # retrieved chunks with full metadata + score
    query: str
    model: str                   # generation model used
    n_retrieved: int
```

Each entry in `sources`:
```
{title, author, memo_date, doc_type, source_year, score, source, text}
```

### `RAGPipeline` class

```python
class RAGPipeline:
    def __init__(self, n_results=DEFAULT_N):
        self.gemini   = genai.Client(api_key=...)   # LLM CALL SITE (client init)
        self.chroma   = chromadb.PersistentClient(...)
        self.collection = ...
        self.n_results  = n_results

    def _embed_query(self, text: str) -> list[float]:
        # LLM CALL SITE: Gemini RETRIEVAL_QUERY embedding
        ...

    def _build_where(self, year, doc_type, author, where) -> dict | None:
        # Convenience params -> Chroma $and filter
        # Raw `where` dict overrides all convenience params

    def retrieve(self, query, n=None, year=None, doc_type=None,
                 author=None, where=None) -> list[dict]:
        # Embed query, query Chroma, return enriched chunk list

    def _build_prompt(self, query, chunks) -> str:
        # Numbered source blocks with metadata header, then QUESTION:

    def answer(self, query, n=None, year=None, doc_type=None,
               author=None, where=None) -> RAGResponse:
        # retrieve -> build_prompt -> generate -> RAGResponse
        # LLM CALL SITE: Gemini generate_content
```

### Filter logic

```python
def _build_where(self, year, doc_type, author, where):
    if where:
        return where          # raw dict overrides everything

    conditions = []
    if year:      conditions.append({"source_year": {"$gte": int(year)}})
    if doc_type:  conditions.append({"doc_type":    {"$eq": doc_type}})
    if author:    conditions.append({"author":       {"$eq": author}})

    if not conditions: return None
    if len(conditions) == 1: return conditions[0]
    return {"$and": conditions}
```

### System prompt

```
You are an expert assistant on Indiana DLGF (Department of Local Government Finance)
guidance documents and memos.

Answer the user's question using ONLY the source documents provided. Be specific and
direct. If the answer cannot be found in the sources, say so clearly -- do not
invent information.

Cite sources inline using their number [1], [2], etc. At the end of your answer list
each cited source on its own line:
[N] Title -- Author (Date)
```

### CLI (`main()`)

```
python rag_pipeline.py                      # interactive loop, defaults
python rag_pipeline.py --n 8               # retrieve 8 chunks
python rag_pipeline.py --year 2024         # filter every query to 2024+
python rag_pipeline.py --doc-type MEMO     # filter to memos only
python rag_pipeline.py --author "Wood"     # filter by author
```

Interactive loop prints `answer`, then a source summary line
`[N chunks retrieved | gemini-2.0-flash]`.

---

## README: LLM Reference section

New section added after "Using This Pipeline With Your Own Documents". Contains:

1. A table of every LLM call site in the project (file, purpose, constant/model to change).
2. A note that embedding model and generation model are independent -- you can embed with
   Gemini and generate with Claude or OpenAI.
3. A note on what package to install and what env var to set for each provider.

---

## Error handling

- Missing `GEMINI_API_KEY`: raise `SystemExit` with clear message at pipeline init.
- Chroma collection not found: raise with message pointing to `build_vectorstore.py`.
- No chunks retrieved: return `RAGResponse` with answer set to a clear "no results" message,
  empty sources list.
- LLM API error: let it propagate (caller decides how to handle).

---

## Out of scope

- Streaming responses
- Conversation history / multi-turn
- LangGraph integration (next phase)
- MCP server (LangGraph phase)
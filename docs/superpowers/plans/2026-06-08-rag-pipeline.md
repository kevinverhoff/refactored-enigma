# RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rag_pipeline.py` -- a single-file RAG pipeline that retrieves chunks from Chroma, generates answers with Gemini, and supports an interactive CLI and programmatic use.

**Architecture:** `RAGPipeline` class holds all clients (Gemini, Chroma) as instance state, initialized once and reused across calls. Pure logic methods (`_build_where`, `_build_prompt`) are fully unit-testable without mocks; API-touching methods use injected clients so they can be mocked in tests. README gains an LLM Reference section documenting every model call site across the project.

**Tech Stack:** `google-genai`, `chromadb`, `python-dotenv`, `pytest`

---

## File Map

- **Create:** `rag_pipeline.py`
- **Create:** `tests/__init__.py`
- **Create:** `tests/test_rag_pipeline.py`
- **Modify:** `requirements.txt` -- add `pytest>=8.0`
- **Modify:** `README.md` -- add LLM Reference section

---

### Task 1: Test scaffolding + pytest

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add pytest to requirements.txt**

Add one line at the end:
```
pytest>=8.0
```

- [ ] **Step 2: Install pytest**

```bash
pip install pytest
```

Expected: `Successfully installed pytest-...` or `Requirement already satisfied`

- [ ] **Step 3: Create tests directory and empty __init__.py**

```bash
mkdir tests
```

Create `tests/__init__.py` as an empty file.

- [ ] **Step 4: Create test file with imports**

Create `tests/test_rag_pipeline.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
```

- [ ] **Step 5: Verify pytest discovers the file**

Run from the project root:
```bash
pytest tests/ -v
```
Expected: `no tests ran` (zero tests collected, no errors)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "add pytest and test scaffolding"
```

---

### Task 2: RAGResponse dataclass and constants

**Files:**
- Create: `rag_pipeline.py` (initial skeleton)
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Write failing tests for RAGResponse**

Add to `tests/test_rag_pipeline.py`:
```python
from rag_pipeline import RAGResponse, EMBED_MODEL, GENERATION_MODEL


def test_ragresponse_defaults():
    r = RAGResponse(answer="hello")
    assert r.answer == "hello"
    assert r.sources == []
    assert r.query == ""
    assert r.model == ""
    assert r.n_retrieved == 0


def test_ragresponse_all_fields():
    r = RAGResponse(
        answer="an answer",
        sources=[{"title": "doc"}],
        query="a question",
        model="gemini-2.0-flash",
        n_retrieved=1,
    )
    assert r.n_retrieved == 1
    assert r.sources[0]["title"] == "doc"


def test_constants_are_strings():
    assert isinstance(EMBED_MODEL, str)
    assert isinstance(GENERATION_MODEL, str)
    assert len(EMBED_MODEL) > 0
    assert len(GENERATION_MODEL) > 0
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
pytest tests/test_rag_pipeline.py -v
```
Expected: `ModuleNotFoundError: No module named 'rag_pipeline'`

- [ ] **Step 3: Create rag_pipeline.py with constants and RAGResponse**

Create `rag_pipeline.py`:
```python
"""
RAG pipeline for DLGF memo corpus.

Usage:
    python rag_pipeline.py                     # interactive loop
    python rag_pipeline.py --n 8               # retrieve 8 chunks
    python rag_pipeline.py --year 2024         # filter source_year >= 2024
    python rag_pipeline.py --doc-type MEMO     # filter by doc type
    python rag_pipeline.py --author "Wood"     # filter by author
"""

import argparse
import os
from dataclasses import dataclass, field

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# LLM call sites -- change these constants to swap models.
# See the "LLM Reference" section in README.md for full swap instructions.
# ---------------------------------------------------------------------------

# SWAP: embedding model used at query time (must match model used in build_vectorstore.py)
EMBED_MODEL = "gemini-embedding-001"

# SWAP: generation model for producing answers
GENERATION_MODEL = "gemini-2.0-flash"

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "dlgf_memos"
DEFAULT_N = 6

SYSTEM_PROMPT = (
    "You are an expert assistant on Indiana DLGF (Department of Local Government Finance) "
    "guidance documents and memos.\n\n"
    "Answer the user's question using ONLY the source documents provided. Be specific and "
    "direct. If the answer cannot be found in the sources, say so clearly -- do not invent "
    "information.\n\n"
    "Cite sources inline using their number [1], [2], etc. At the end of your answer list "
    "each cited source on its own line:\n"
    "[N] Title -- Author (Date)"
)


@dataclass
class RAGResponse:
    answer: str
    sources: list = field(default_factory=list)
    query: str = ""
    model: str = ""
    n_retrieved: int = 0
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
pytest tests/test_rag_pipeline.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py requirements.txt
git commit -m "add RAGResponse dataclass and constants"
```

---

### Task 3: `_build_where` filter logic

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `rag_pipeline.py`

- [ ] **Step 1: Write failing tests for _build_where**

The `_build_where` method needs a `RAGPipeline` instance. Use a pytest fixture that patches the external clients so no real API calls are made.

Add to `tests/test_rag_pipeline.py`:
```python
from rag_pipeline import RAGResponse, RAGPipeline, EMBED_MODEL, GENERATION_MODEL


@pytest.fixture
def pipeline():
    with patch("rag_pipeline.genai.Client"), \
         patch("rag_pipeline.chromadb.PersistentClient") as mock_chroma, \
         patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        yield RAGPipeline()


def test_build_where_no_params(pipeline):
    assert pipeline._build_where() is None


def test_build_where_year_only(pipeline):
    assert pipeline._build_where(year=2024) == {"source_year": {"$gte": 2024}}


def test_build_where_doc_type_only(pipeline):
    assert pipeline._build_where(doc_type="MEMO") == {"doc_type": {"$eq": "MEMO"}}


def test_build_where_author_only(pipeline):
    assert pipeline._build_where(author="Wood") == {"author": {"$eq": "Wood"}}


def test_build_where_multiple_params(pipeline):
    result = pipeline._build_where(year=2024, doc_type="MEMO")
    assert result == {"$and": [
        {"source_year": {"$gte": 2024}},
        {"doc_type": {"$eq": "MEMO"}},
    ]}


def test_build_where_all_three(pipeline):
    result = pipeline._build_where(year=2023, doc_type="TEMPLATE", author="Wood")
    assert result == {"$and": [
        {"source_year": {"$gte": 2023}},
        {"doc_type": {"$eq": "TEMPLATE"}},
        {"author": {"$eq": "Wood"}},
    ]}


def test_build_where_raw_overrides_convenience(pipeline):
    raw = {"cluster_id": {"$eq": 16}}
    result = pipeline._build_where(year=2024, doc_type="MEMO", where=raw)
    assert result == raw
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
pytest tests/test_rag_pipeline.py -v -k "build_where"
```
Expected: `ImportError` or `AttributeError` (RAGPipeline not defined yet)

- [ ] **Step 3: Add RAGPipeline class with __init__ and _build_where**

Append to `rag_pipeline.py`:
```python
class RAGPipeline:
    def __init__(self, n_results: int = DEFAULT_N):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "ERROR: GEMINI_API_KEY not set. Add it to your .env file."
            )
        # LLM SWAP: replace genai.Client with your provider's client.
        self.gemini = genai.Client(api_key=api_key)
        self.n_results = n_results

        chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            self.collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:
            raise SystemExit(
                f"ERROR: Chroma collection '{COLLECTION_NAME}' not found. "
                "Run build_vectorstore.py first."
            )

    def _build_where(
        self,
        year: int = None,
        doc_type: str = None,
        author: str = None,
        where: dict = None,
    ) -> dict:
        if where:
            return where

        conditions = []
        if year:
            conditions.append({"source_year": {"$gte": int(year)}})
        if doc_type:
            conditions.append({"doc_type": {"$eq": doc_type}})
        if author:
            conditions.append({"author": {"$eq": author}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
pytest tests/test_rag_pipeline.py -v -k "build_where"
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "add RAGPipeline init and _build_where"
```

---

### Task 4: `_build_prompt`

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `rag_pipeline.py`

- [ ] **Step 1: Write failing tests for _build_prompt**

Add to `tests/test_rag_pipeline.py`:
```python
def test_build_prompt_numbers_sources(pipeline):
    chunks = [
        {"title": "Memo A", "author": "Wood", "memo_date": "2024-01-01",
         "doc_type": "MEMO", "text": "Content of A"},
        {"title": "Memo B", "author": "Shackle", "memo_date": "2024-06-01",
         "doc_type": "MEMO", "text": "Content of B"},
    ]
    prompt = pipeline._build_prompt("What is homestead?", chunks)
    assert "[1] Memo A | Wood | 2024-01-01 | MEMO" in prompt
    assert "[2] Memo B | Shackle | 2024-06-01 | MEMO" in prompt
    assert "Content of A" in prompt
    assert "Content of B" in prompt
    assert "QUESTION: What is homestead?" in prompt


def test_build_prompt_missing_fields_fallback(pipeline):
    chunks = [{"title": "", "author": "", "memo_date": "", "doc_type": "", "text": "Body"}]
    prompt = pipeline._build_prompt("test?", chunks)
    assert "[1] Untitled | Unknown" in prompt
    assert "Body" in prompt


def test_build_prompt_sources_separated(pipeline):
    chunks = [
        {"title": "A", "author": "X", "memo_date": "2024-01-01",
         "doc_type": "MEMO", "text": "Text A"},
        {"title": "B", "author": "Y", "memo_date": "2024-02-01",
         "doc_type": "MEMO", "text": "Text B"},
    ]
    prompt = pipeline._build_prompt("q?", chunks)
    assert "---" in prompt
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
pytest tests/test_rag_pipeline.py -v -k "build_prompt"
```
Expected: `AttributeError: 'RAGPipeline' object has no attribute '_build_prompt'`

- [ ] **Step 3: Add _build_prompt to RAGPipeline**

Add inside the `RAGPipeline` class in `rag_pipeline.py`:
```python
    def _build_prompt(self, query: str, chunks: list) -> str:
        source_blocks = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title") or "Untitled"
            author = chunk.get("author") or "Unknown"
            date = chunk.get("memo_date") or ""
            doc_type = chunk.get("doc_type") or ""
            header = f"[{i}] {title} | {author} | {date} | {doc_type}"
            source_blocks.append(f"{header}\n{chunk['text']}")

        sources_text = "\n\n---\n\n".join(source_blocks)
        return f"SOURCES:\n\n{sources_text}\n\n---\n\nQUESTION: {query}"
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
pytest tests/test_rag_pipeline.py -v -k "build_prompt"
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "add _build_prompt"
```

---

### Task 5: `_embed_query` and `retrieve`

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `rag_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_rag_pipeline.py`:
```python
def test_embed_query_calls_gemini(pipeline):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    pipeline.gemini.models.embed_content = MagicMock(
        return_value=MagicMock(embeddings=[mock_embedding])
    )
    result = pipeline._embed_query("homestead deduction")
    assert result == [0.1, 0.2, 0.3]
    pipeline.gemini.models.embed_content.assert_called_once()


def test_retrieve_returns_enriched_chunks(pipeline):
    pipeline._embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    pipeline.collection.query = MagicMock(return_value={
        "documents": [["chunk text"]],
        "metadatas": [[{
            "title": "Test Memo", "author": "Wood", "memo_date": "2024-01-01",
            "doc_type": "MEMO", "source_year": 2024,
            "source": "https://example.com/test.pdf",
        }]],
        "distances": [[0.25]],
    })
    chunks = pipeline.retrieve("homestead deduction")
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Test Memo"
    assert chunks[0]["score"] == 0.75
    assert chunks[0]["text"] == "chunk text"


def test_retrieve_passes_where_filter(pipeline):
    pipeline._embed_query = MagicMock(return_value=[0.1])
    pipeline.collection.query = MagicMock(return_value={
        "documents": [[]], "metadatas": [[]], "distances": [[]]
    })
    pipeline.retrieve("query", year=2024)
    call_kwargs = pipeline.collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"source_year": {"$gte": 2024}}
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
pytest tests/test_rag_pipeline.py -v -k "embed_query or retrieve"
```
Expected: `AttributeError` on `_embed_query` or `retrieve`

- [ ] **Step 3: Add _embed_query and retrieve to RAGPipeline**

Add inside the `RAGPipeline` class in `rag_pipeline.py`:
```python
    def _embed_query(self, text: str) -> list:
        # LLM SWAP: replace with your provider's embedding call.
        # Use a query-optimized embedding type if your provider supports it.
        result = self.gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    def retrieve(
        self,
        query: str,
        n: int = None,
        year: int = None,
        doc_type: str = None,
        author: str = None,
        where: dict = None,
    ) -> list:
        n = n or self.n_results
        where_clause = self._build_where(
            year=year, doc_type=doc_type, author=author, where=where
        )

        kwargs = dict(
            query_embeddings=[self._embed_query(query)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            kwargs["where"] = where_clause

        r = self.collection.query(**kwargs)

        chunks = []
        for doc, meta, dist in zip(
            r["documents"][0], r["metadatas"][0], r["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "score": round(1 - dist, 3),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "memo_date": meta.get("memo_date", ""),
                "doc_type": meta.get("doc_type", ""),
                "source_year": meta.get("source_year", ""),
                "source": meta.get("source", ""),
            })
        return chunks
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
pytest tests/test_rag_pipeline.py -v -k "embed_query or retrieve"
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "add _embed_query and retrieve"
```

---

### Task 6: `answer`

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `rag_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_rag_pipeline.py`:
```python
def test_answer_no_chunks_returns_no_results_response(pipeline):
    pipeline.retrieve = MagicMock(return_value=[])
    result = pipeline.answer("unknown query")
    assert isinstance(result, RAGResponse)
    assert "No relevant documents" in result.answer
    assert result.sources == []
    assert result.n_retrieved == 0


def test_answer_calls_gemini_and_returns_response(pipeline):
    mock_chunks = [{
        "title": "Test Memo", "author": "Wood", "memo_date": "2024-01-01",
        "doc_type": "MEMO", "text": "Some content", "score": 0.8,
        "source_year": 2024, "source": "https://example.com/test.pdf",
    }]
    pipeline.retrieve = MagicMock(return_value=mock_chunks)
    pipeline.gemini.models.generate_content = MagicMock(
        return_value=MagicMock(text="The answer is [1].")
    )

    result = pipeline.answer("What does this say?")

    assert result.answer == "The answer is [1]."
    assert result.n_retrieved == 1
    assert result.model == GENERATION_MODEL
    assert result.sources == mock_chunks
    assert result.query == "What does this say?"


def test_answer_passes_filters_to_retrieve(pipeline):
    pipeline.retrieve = MagicMock(return_value=[])
    pipeline.answer("query", year=2025, doc_type="MEMO")
    pipeline.retrieve.assert_called_once_with(
        "query", n=None, year=2025, doc_type="MEMO", author=None, where=None
    )
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
pytest tests/test_rag_pipeline.py -v -k "answer"
```
Expected: `AttributeError: 'RAGPipeline' object has no attribute 'answer'`

- [ ] **Step 3: Add answer to RAGPipeline**

Add inside the `RAGPipeline` class in `rag_pipeline.py`:
```python
    def answer(
        self,
        query: str,
        n: int = None,
        year: int = None,
        doc_type: str = None,
        author: str = None,
        where: dict = None,
    ) -> RAGResponse:
        chunks = self.retrieve(
            query, n=n, year=year, doc_type=doc_type, author=author, where=where
        )

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found for this query.",
                query=query,
                model=GENERATION_MODEL,
            )

        prompt = self._build_prompt(query, chunks)

        # LLM SWAP: replace this block with your provider's generation call.
        # Inputs: SYSTEM_PROMPT (str) and prompt (str, contains sources + question).
        # Output: generated answer as a plain string.
        #
        # Claude example:
        #   import anthropic
        #   client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        #   msg = client.messages.create(model="claude-opus-4-8", max_tokens=2048,
        #       system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
        #   response_text = msg.content[0].text
        #
        # OpenAI example:
        #   from openai import OpenAI
        #   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #   resp = client.chat.completions.create(model="gpt-4o", temperature=0.1,
        #       messages=[{"role": "system", "content": SYSTEM_PROMPT},
        #                 {"role": "user", "content": prompt}])
        #   response_text = resp.choices[0].message.content
        response = self.gemini.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )

        return RAGResponse(
            answer=response.text,
            sources=chunks,
            query=query,
            model=GENERATION_MODEL,
            n_retrieved=len(chunks),
        )
```

- [ ] **Step 4: Run all tests -- verify they pass**

```bash
pytest tests/test_rag_pipeline.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "add answer method with LLM swap comments"
```

---

### Task 7: CLI (`main`)

**Files:**
- Modify: `rag_pipeline.py`

- [ ] **Step 1: Add main() to the bottom of rag_pipeline.py**

Append to `rag_pipeline.py`:
```python
def main():
    parser = argparse.ArgumentParser(description="Query DLGF memos via RAG")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter: source_year >= YEAR")
    parser.add_argument("--doc-type", default=None,
                        help="Filter: doc_type equals value (e.g. MEMO, TEMPLATE)")
    parser.add_argument("--author", default=None,
                        help="Filter: author equals value")
    args = parser.parse_args()

    pipeline = RAGPipeline(n_results=args.n)
    print(f"Model: {GENERATION_MODEL} | chunks per query: {args.n}")
    if args.year:
        print(f"  Filter: source_year >= {args.year}")
    if args.doc_type:
        print(f"  Filter: doc_type = {args.doc_type}")
    if args.author:
        print(f"  Filter: author = {args.author}")
    print("Type a question (or 'quit' to exit)\n")

    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.answer(
            query,
            year=args.year,
            doc_type=args.doc_type,
            author=args.author,
        )

        print(f"\n{result.answer}")
        print(f"\n[{result.n_retrieved} chunks retrieved | {result.model}]\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the CLI**

```bash
python rag_pipeline.py --help
```
Expected output showing all flags: `--n`, `--year`, `--doc-type`, `--author`

- [ ] **Step 3: Run all tests to confirm nothing broke**

```bash
pytest tests/test_rag_pipeline.py -v
```
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add rag_pipeline.py
git commit -m "add CLI main()"
```

---

### Task 8: README LLM Reference section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add LLM Reference section after "Using This Pipeline With Your Own Documents"**

Insert the following section before the `## Setup` section in `README.md`:

```markdown
---

## LLM Reference

This project uses two Gemini models. Every call site is marked with a `# LLM SWAP` comment
in the source. To use a different provider, change the relevant constant and replace the
marked code block.

| File | Purpose | Constant | Default model |
|---|---|---|---|
| `build_vectorstore.py` | Embed documents at index time | `EMBED_MODEL` | `gemini-embedding-001` |
| `test_chroma.py` | Embed test queries | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Embed queries at retrieval time | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Generate answers | `GENERATION_MODEL` | `gemini-2.0-flash` |

The embedding model and generation model are independent -- you can embed with Gemini and
generate with Claude or OpenAI. The only hard constraint is that `EMBED_MODEL` must be the
same in `build_vectorstore.py` and `rag_pipeline.py`. Changing it requires rebuilding the
Chroma collection.

### Swapping the generation model

Find the `# LLM SWAP` comment in `rag_pipeline.py::RAGPipeline.answer` and replace the
`generate_content` block.

**Claude** (`pip install anthropic`, add `ANTHROPIC_API_KEY` to `.env`):
```python
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
msg = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=2048,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
)
response_text = msg.content[0].text
```

**OpenAI** (`pip install openai`, add `OPENAI_API_KEY` to `.env`):
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
resp = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.1,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
)
response_text = resp.choices[0].message.content
```

Also update the `__init__` client initialization (marked `# LLM SWAP`) to instantiate your
chosen provider's client instead of `genai.Client`.
```

- [ ] **Step 2: Verify README renders correctly**

Open `README.md` and confirm the table and code blocks look right.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "add LLM reference section to README"
git push
```

---

### Task 9: Live smoke test

**Files:** none

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 2: Start the CLI and ask one question**

```bash
python rag_pipeline.py
```
Type: `What are the requirements for the homestead deduction?`

Expected: a grounded answer with at least one inline citation `[1]` and a source list at the end, followed by `[N chunks retrieved | gemini-2.0-flash]`

- [ ] **Step 3: Test a filter**

```bash
python rag_pipeline.py --year 2025 --doc-type MEMO
```
Type: `What changed with excess levy appeals?`

Expected: results with `memo_date` values in 2025 or later only.

- [ ] **Step 4: Final commit and push**

```bash
git add .
git commit -m "rag pipeline complete"
git push
```
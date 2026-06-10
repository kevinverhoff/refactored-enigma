# LangGraph Tools Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph ReAct agent with 6 purpose-built tools wrapping the existing RAG pipeline, plus a minimal Streamlit chat UI.

**Architecture:** A module-level `RAGPipeline` instance (lazy-initialized) is shared across all `@tool` functions in `tools.py`. `agent.py` wires those tools into a `create_react_agent` graph with a routing system prompt. `app.py` is a thin Streamlit chat wrapper that calls `graph.invoke` with the growing message history on each turn.

**Tech Stack:** `langgraph`, `langchain-google-genai`, `langchain-core`, `streamlit`, `pytest`, `unittest.mock`

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `langgraph`, `langchain-google-genai`, `streamlit` |
| `tools.py` | Create | 6 `@tool` functions, lazy `_pipeline` init |
| `tests/test_tools.py` | Create | Unit tests for all 6 tools |
| `agent.py` | Create | `create_react_agent` graph, system prompt, `AGENT_MODEL` constant |
| `app.py` | Create | Streamlit chat UI |

`rag_pipeline.py` is **not modified**.

---

## Task 1: Install new dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

Replace the last line of `requirements.txt`:

```
requests>=2.31
beautifulsoup4>=4.12
pdfplumber>=0.11
python-docx>=1.1
openpyxl>=3.1
pandas>=2.2
pyarrow>=16.0
plotly>=5.22
kaleido>=0.2
scikit-learn>=1.4
numpy>=1.26
umap-learn>=0.5
google-genai>=1.0
chromadb>=0.6
python-dotenv>=1.0
pytest>=8.0
langgraph>=0.2
langchain-google-genai>=2.0
langchain-core>=0.3
streamlit>=1.35
```

- [ ] **Step 2: Install**

```bash
python -m pip install langgraph langchain-google-genai langchain-core streamlit
```

Expected: packages install without conflict. Verify:

```bash
python -c "import langgraph; import langchain_google_genai; import streamlit; print('OK')"
```

Expected output: `OK`

---

## Task 2: Test scaffold + tools.py skeleton

**Files:**
- Create: `tests/test_tools.py`
- Create: `tools.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_tools.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import tools


SAMPLE_CHUNK = {
    "text": "The circuit breaker credit limits property tax liability.",
    "score": 0.95,
    "title": "Circuit Breaker Guidance",
    "author": "Wood",
    "memo_date": "2025-03-14",
    "doc_type": "MEMO",
    "source_year": 2025,
    "source": "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf",
}


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Replace module-level pipeline with a mock before each test."""
    mock = MagicMock()
    mock.retrieve.return_value = [SAMPLE_CHUNK]
    mock.gemini.models.generate_content.return_value.text = "Mocked generation output."
    tools._pipeline = mock
    yield mock
    tools._pipeline = None


def test_tools_importable():
    assert callable(tools.search)
    assert callable(tools.answer)
    assert callable(tools.summarize)
    assert callable(tools.compare_years)
    assert callable(tools.get_topics)
    assert callable(tools.extract_quotes)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_tools.py::test_tools_importable -v
```

Expected: `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Create tools.py skeleton**

Create `tools.py`:

```python
from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

from google.genai import types
from langchain_core.tools import tool

from rag_pipeline import RAGPipeline, GENERATION_MODEL

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _format_url_block(chunks: list[dict]) -> str:
    return "\n".join(f"[{i}] {c['source']}" for i, c in enumerate(chunks, 1))


@tool
def search(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 10) -> str:
    """Find documents in the DLGF memo corpus relevant to a topic or question.
    Returns a list of matching document titles, authors, dates, relevance scores,
    and source URLs. Use this when the user wants to browse or discover documents
    rather than get a generated answer."""
    raise NotImplementedError


@tool
def answer(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 6) -> str:
    """Answer a question about DLGF memos using retrieved source documents.
    Returns a grounded answer with inline citations [1], [2], etc., each including
    title, author, date, and source URL. Use this when the user asks a direct question."""
    raise NotImplementedError


@tool
def summarize(topic: str, year: int = None, doc_type: str = None,
              author: str = None, n: int = 12) -> str:
    """Summarize what the DLGF memo corpus collectively says about a topic.
    Retrieves more documents than `answer` for broader coverage. Returns a synthesis
    with source URLs. Use this when the user wants a broad overview across documents."""
    raise NotImplementedError


@tool
def compare_years(topic: str, year_start: int, year_end: int, n: int = 6) -> str:
    """Compare how DLGF guidance on a topic changed between two time periods.
    Retrieves documents from each time window separately, then generates a comparison
    with source URLs for both windows. Use this for trend analysis or what-changed questions."""
    raise NotImplementedError


@tool
def get_topics() -> str:
    """Return the major topic clusters found in the DLGF memo corpus with document
    counts per cluster. Use this when the user wants to explore what subjects are
    covered, or to orient a new conversation."""
    raise NotImplementedError


@tool
def extract_quotes(topic: str, year: int = None, n: int = 8) -> str:
    """Find the most notable or quotable passages about a topic in the corpus.
    Returns attributed pull quotes with document title, author, date, and source URL.
    Use this when the user wants specific language or key statements from the memos."""
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_tools.py::test_tools_importable -v
```

Expected: `PASSED`

---

## Task 3: `search` and `answer` tools

**Files:**
- Modify: `tools.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_tools.py`:

```python
def test_search_returns_formatted_results(reset_pipeline):
    result = tools.search.invoke({"query": "circuit breaker credits"})
    assert "Circuit Breaker Guidance" in result
    assert "Wood" in result
    assert "2025-03-14" in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result
    assert "0.95" in result


def test_search_passes_filters_to_retrieve(reset_pipeline):
    tools.search.invoke({"query": "budget", "year": 2024, "doc_type": "MEMO", "author": "Wood"})
    reset_pipeline.retrieve.assert_called_once_with(
        "budget", year=2024, doc_type="MEMO", author="Wood", n=10
    )


def test_search_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.search.invoke({"query": "nonexistent topic"})
    assert "No documents found" in result


def test_answer_returns_answer_and_urls(reset_pipeline):
    from rag_pipeline import RAGResponse
    reset_pipeline.answer.return_value = RAGResponse(
        answer="The circuit breaker credit [1] limits tax liability.",
        sources=[SAMPLE_CHUNK],
        query="What is the circuit breaker credit?",
        model="gemini-2.5-flash",
        n_retrieved=1,
    )
    result = tools.answer.invoke({"query": "What is the circuit breaker credit?"})
    assert "circuit breaker credit" in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_answer_no_results(reset_pipeline):
    from rag_pipeline import RAGResponse
    reset_pipeline.answer.return_value = RAGResponse(
        answer="No relevant documents found for this query.",
        sources=[],
        query="something obscure",
        model="gemini-2.5-flash",
        n_retrieved=0,
    )
    result = tools.answer.invoke({"query": "something obscure"})
    assert "No relevant documents" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_tools.py::test_search_returns_formatted_results tests/test_tools.py::test_answer_returns_answer_and_urls -v
```

Expected: `NotImplementedError`

- [ ] **Step 3: Implement `search` and `answer`**

Replace the `search` and `answer` stubs in `tools.py`:

```python
@tool
def search(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 10) -> str:
    """Find documents in the DLGF memo corpus relevant to a topic or question.
    Returns a list of matching document titles, authors, dates, relevance scores,
    and source URLs. Use this when the user wants to browse or discover documents
    rather than get a generated answer."""
    p = _get_pipeline()
    chunks = p.retrieve(query, year=year, doc_type=doc_type, author=author, n=n)
    if not chunks:
        return "No documents found matching your query."
    lines = [f"Found {len(chunks)} document(s):\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {c['title']} -- {c['author']} ({c['memo_date']}) | {c['doc_type']}\n"
            f"    Score: {c['score']}\n"
            f"    {c['source']}"
        )
    return "\n\n".join(lines)


@tool
def answer(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 6) -> str:
    """Answer a question about DLGF memos using retrieved source documents.
    Returns a grounded answer with inline citations [1], [2], etc., each including
    title, author, date, and source URL. Use this when the user asks a direct question."""
    p = _get_pipeline()
    response = p.answer(query, year=year, doc_type=doc_type, author=author, n=n)
    if not response.sources:
        return response.answer
    url_block = _format_url_block(response.sources)
    return f"{response.answer}\n\nSource URLs:\n{url_block}"
```

- [ ] **Step 4: Run all search/answer tests**

```bash
python -m pytest tests/test_tools.py::test_search_returns_formatted_results tests/test_tools.py::test_search_passes_filters_to_retrieve tests/test_tools.py::test_search_no_results tests/test_tools.py::test_answer_returns_answer_and_urls tests/test_tools.py::test_answer_no_results -v
```

Expected: all 5 `PASSED`

---

## Task 4: `get_topics` tool

**Files:**
- Modify: `tools.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_tools.py`:

```python
def test_get_topics_reads_cluster_csv(tmp_path, monkeypatch):
    csv_content = (
        "cluster_id,cluster_label,doc_count,doc_types,mean_char_count,top_terms,date_range\n"
        "-1,Unassigned,155,MEMO (139),6506,,2021-04-19 to 2026-05-27\n"
        "0,reassessment parcels,33,MEMO (33),954,\"parcels, reassessment\",2022-01-03 to 2026-06-02\n"
        "1,sales disclosure,8,MEMO (8),3203,\"sales disclosure, fee\",2022-12-30 to 2025-12-11\n"
    )
    csv_file = tmp_path / "cluster_summary.csv"
    csv_file.write_text(csv_content)
    monkeypatch.setattr(tools, "_CLUSTER_CSV", csv_file)

    result = tools.get_topics.invoke({})
    assert "reassessment parcels" in result
    assert "sales disclosure" in result
    assert "33 docs" in result
    assert "Unassigned" not in result  # noise cluster filtered out


def test_get_topics_missing_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_CLUSTER_CSV", tmp_path / "nonexistent.csv")
    result = tools.get_topics.invoke({})
    assert "not available" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_tools.py::test_get_topics_reads_cluster_csv tests/test_tools.py::test_get_topics_missing_csv -v
```

Expected: `NotImplementedError` or `AttributeError: module 'tools' has no attribute '_CLUSTER_CSV'`

- [ ] **Step 3: Implement `get_topics`**

Add the path constant near the top of `tools.py` (after the imports):

```python
_CLUSTER_CSV: Path = Path(__file__).parent / "cluster_summary.csv"
```

Replace the `get_topics` stub:

```python
@tool
def get_topics() -> str:
    """Return the major topic clusters found in the DLGF memo corpus with document
    counts per cluster. Use this when the user wants to explore what subjects are
    covered, or to orient a new conversation."""
    if not _CLUSTER_CSV.exists():
        return "Topic cluster summary not available. Run document_clustering.py first."
    df = pd.read_csv(_CLUSTER_CSV)
    df = df[df["cluster_id"] != -1].sort_values("doc_count", ascending=False)
    lines = [f"Found {len(df)} topic clusters in the DLGF memo corpus:\n"]
    for _, row in df.iterrows():
        lines.append(
            f"Cluster {int(row['cluster_id'])}: {int(row['doc_count'])} docs "
            f"| {row['date_range']}\n"
            f"  Top terms: {row['top_terms']}\n"
            f"  Doc types: {row['doc_types']}"
        )
    return "\n\n".join(lines)
```

- [ ] **Step 4: Run get_topics tests**

```bash
python -m pytest tests/test_tools.py::test_get_topics_reads_cluster_csv tests/test_tools.py::test_get_topics_missing_csv -v
```

Expected: both `PASSED`

---

## Task 5: `summarize` and `extract_quotes` tools

**Files:**
- Modify: `tools.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_tools.py`:

```python
def test_summarize_calls_retrieve_with_n12(reset_pipeline):
    result = tools.summarize.invoke({"topic": "circuit breaker credits"})
    reset_pipeline.retrieve.assert_called_once_with(
        "circuit breaker credits", year=None, doc_type=None, author=None, n=12
    )
    assert "Mocked generation output." in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_summarize_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.summarize.invoke({"topic": "nonexistent"})
    assert "No documents found" in result


def test_extract_quotes_returns_quoted_output(reset_pipeline):
    result = tools.extract_quotes.invoke({"topic": "circuit breaker"})
    reset_pipeline.retrieve.assert_called_once_with("circuit breaker", year=None, n=8)
    assert "Mocked generation output." in result
    assert "https://www.in.gov/dlgf/files/25-03-wood-memo-circuit-breaker.pdf" in result


def test_extract_quotes_no_results(reset_pipeline):
    reset_pipeline.retrieve.return_value = []
    result = tools.extract_quotes.invoke({"topic": "nonexistent"})
    assert "No documents found" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_tools.py::test_summarize_calls_retrieve_with_n12 tests/test_tools.py::test_extract_quotes_returns_quoted_output -v
```

Expected: `NotImplementedError`

- [ ] **Step 3: Implement `summarize` and `extract_quotes`**

Add system prompt constants near the top of `tools.py` (after `GENERATION_MODEL` import):

```python
_SUMMARIZE_SYSTEM = (
    "You are an expert on Indiana DLGF (Department of Local Government Finance) "
    "guidance documents. Synthesize the provided source documents into a clear, "
    "structured summary of what they collectively say about the given topic. "
    "Group related points. Be specific. Do not invent information not in the sources. "
    "At the end, list each source: [N] Title -- Author (Date)"
)

_QUOTES_SYSTEM = (
    "You are an expert on Indiana DLGF guidance documents. "
    "From the provided source documents, extract the most notable, quotable, or "
    "policy-significant passages about the given topic. Preserve the exact language. "
    "Format each as:\nQUOTE: \"[exact text]\"\nSOURCE: [N] Title -- Author (Date)"
)
```

Replace the `summarize` stub:

```python
@tool
def summarize(topic: str, year: int = None, doc_type: str = None,
              author: str = None, n: int = 12) -> str:
    """Summarize what the DLGF memo corpus collectively says about a topic.
    Retrieves more documents than `answer` for broader coverage. Returns a synthesis
    with source URLs. Use this when the user wants a broad overview across documents."""
    p = _get_pipeline()
    chunks = p.retrieve(topic, year=year, doc_type=doc_type, author=author, n=n)
    if not chunks:
        return f"No documents found about '{topic}'."
    prompt = p._build_prompt(topic, chunks)
    response = p.gemini.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SUMMARIZE_SYSTEM,
            temperature=0.2,
        ),
    )
    return f"{response.text}\n\nSource URLs:\n{_format_url_block(chunks)}"
```

Replace the `extract_quotes` stub:

```python
@tool
def extract_quotes(topic: str, year: int = None, n: int = 8) -> str:
    """Find the most notable or quotable passages about a topic in the corpus.
    Returns attributed pull quotes with document title, author, date, and source URL.
    Use this when the user wants specific language or key statements from the memos."""
    p = _get_pipeline()
    chunks = p.retrieve(topic, year=year, n=n)
    if not chunks:
        return f"No documents found about '{topic}'."
    prompt = p._build_prompt(topic, chunks)
    response = p.gemini.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_QUOTES_SYSTEM,
            temperature=0.1,
        ),
    )
    return f"{response.text}\n\nSource URLs:\n{_format_url_block(chunks)}"
```

- [ ] **Step 4: Run all summarize/extract_quotes tests**

```bash
python -m pytest tests/test_tools.py::test_summarize_calls_retrieve_with_n12 tests/test_tools.py::test_summarize_no_results tests/test_tools.py::test_extract_quotes_returns_quoted_output tests/test_tools.py::test_extract_quotes_no_results -v
```

Expected: all 4 `PASSED`

---

## Task 6: `compare_years` tool

**Files:**
- Modify: `tools.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_tools.py`:

```python
def test_compare_years_two_retrieval_calls(reset_pipeline):
    early_chunk = {**SAMPLE_CHUNK, "source_year": 2022, "memo_date": "2022-06-01",
                   "source": "https://www.in.gov/dlgf/files/22-early.pdf"}
    late_chunk = {**SAMPLE_CHUNK, "source_year": 2025, "memo_date": "2025-06-01",
                  "source": "https://www.in.gov/dlgf/files/25-late.pdf"}
    reset_pipeline.retrieve.side_effect = [[early_chunk], [late_chunk]]

    result = tools.compare_years.invoke({
        "topic": "circuit breaker", "year_start": 2022, "year_end": 2025
    })

    calls = reset_pipeline.retrieve.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["where"] == {"source_year": {"$lte": 2022}}
    assert calls[1].kwargs["where"] == {"source_year": {"$gte": 2025}}
    assert "Mocked generation output." in result
    assert "22-early.pdf" in result
    assert "25-late.pdf" in result


def test_compare_years_no_results(reset_pipeline):
    reset_pipeline.retrieve.side_effect = [[], []]
    result = tools.compare_years.invoke({
        "topic": "nonexistent", "year_start": 2022, "year_end": 2025
    })
    assert "No documents found" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_tools.py::test_compare_years_two_retrieval_calls -v
```

Expected: `NotImplementedError`

- [ ] **Step 3: Implement `compare_years`**

Add system prompt constant to `tools.py` (with the other `_*_SYSTEM` constants):

```python
_COMPARE_SYSTEM = (
    "You are an expert on Indiana DLGF guidance documents. "
    "You have been provided two sets of documents from different time periods. "
    "Compare how the guidance changed between these periods. "
    "Highlight what stayed the same, what changed, and any notable new requirements or removals. "
    "Be specific. At the end, list each source: [N] Title -- Author (Date) | Period"
)
```

Replace the `compare_years` stub:

```python
@tool
def compare_years(topic: str, year_start: int, year_end: int, n: int = 6) -> str:
    """Compare how DLGF guidance on a topic changed between two time periods.
    Retrieves documents from each time window separately, then generates a comparison
    with source URLs for both windows. Use this for trend analysis or what-changed questions."""
    p = _get_pipeline()
    early = p.retrieve(topic, where={"source_year": {"$lte": year_start}}, n=n)
    late = p.retrieve(topic, where={"source_year": {"$gte": year_end}}, n=n)
    if not early and not late:
        return f"No documents found about '{topic}' in either time window."

    sections = [f"=== DOCUMENTS FROM {year_start} AND EARLIER ===\n"]
    all_chunks = []
    idx = 1
    for c in early:
        header = (f"[{idx}] {c.get('title', 'Untitled')} | {c.get('author', 'Unknown')} | "
                  f"{c.get('memo_date', '')} | PERIOD: up to {year_start}")
        sections.append(f"{header}\n{c['text']}")
        all_chunks.append(c)
        idx += 1
    sections.append(f"\n=== DOCUMENTS FROM {year_end} AND LATER ===\n")
    for c in late:
        header = (f"[{idx}] {c.get('title', 'Untitled')} | {c.get('author', 'Unknown')} | "
                  f"{c.get('memo_date', '')} | PERIOD: {year_end} onwards")
        sections.append(f"{header}\n{c['text']}")
        all_chunks.append(c)
        idx += 1

    prompt = (
        "\n\n---\n\n".join(sections)
        + f"\n\n---\n\nQUESTION: How has DLGF guidance on '{topic}' changed "
        f"between {year_start} and {year_end}?"
    )
    response = p.gemini.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_COMPARE_SYSTEM,
            temperature=0.2,
        ),
    )
    return f"{response.text}\n\nSource URLs:\n{_format_url_block(all_chunks)}"
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all tests `PASSED`

---

## Task 7: `agent.py`

**Files:**
- Create: `agent.py`

No unit tests for the graph itself (LangGraph graph wiring is best verified by running it).

- [ ] **Step 1: Create agent.py**

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import answer, compare_years, extract_quotes, get_topics, search, summarize

load_dotenv()

# LLM SWAP: replace ChatGoogleGenerativeAI with your provider's chat model class.
# Must support tool/function calling. Set AGENT_MODEL to the corresponding model ID.
AGENT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """You are an expert research assistant for Indiana DLGF (Department of Local Government Finance) memos and guidance documents, covering 2022-2026.

You have access to tools for searching, answering questions, summarizing topics, comparing guidance across years, listing topics, and extracting notable quotes.

Guidelines:
- For direct questions, use `answer`.
- For "find documents about X", use `search`.
- For broad synthesis or overviews, use `summarize`.
- For "what changed" or trend questions, use `compare_years`.
- For orientation or topic discovery, use `get_topics` first.
- For notable language or key statements, use `extract_quotes`.
- If a query is ambiguous, ask one clarifying question before calling a tool.
- Always include source URLs when citing documents."""

_tools = [search, answer, summarize, compare_years, get_topics, extract_quotes]
llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0.1)
graph = create_react_agent(llm, _tools, prompt=_SYSTEM_PROMPT)
```

- [ ] **Step 2: Smoke test the agent**

```bash
python -c "
from agent import graph
from langchain_core.messages import HumanMessage

result = graph.invoke({'messages': [HumanMessage(content='What topics are covered in this corpus?')]})
print(result['messages'][-1].content[:500])
"
```

Expected: a response referencing topic clusters from the corpus (may take 10-20s, makes real Gemini + Chroma calls).

---

## Task 8: `app.py`

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

```python
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import graph

st.set_page_config(page_title="DLGF Memo Assistant", layout="wide")
st.title("DLGF Memo Assistant")
st.caption("Ask questions about Indiana DLGF memos and guidance documents (2022-2026)")

with st.sidebar:
    st.header("Optional Filters")
    st.markdown("Include these in your question to narrow results, or set them here as defaults.")
    year_filter = st.number_input("Year (on or after)", min_value=2022, max_value=2026,
                                  value=None, step=1)
    doc_type_filter = st.selectbox("Document type", ["", "MEMO", "TEMPLATE", "FORM", "ATTACHMENT"])
    author_filter = st.text_input("Author")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask about DLGF memos..."):
    filter_parts = []
    if year_filter:
        filter_parts.append(f"year >= {int(year_filter)}")
    if doc_type_filter:
        filter_parts.append(f"doc_type = {doc_type_filter}")
    if author_filter:
        filter_parts.append(f"author = {author_filter}")

    full_prompt = f"[Filters active: {', '.join(filter_parts)}] {prompt}" if filter_parts else prompt

    user_msg = HumanMessage(content=full_prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"messages": st.session_state.messages})
            ai_content = result["messages"][-1].content
        st.markdown(ai_content)

    st.session_state.messages.append(AIMessage(content=ai_content))
```

- [ ] **Step 2: Run the Streamlit app**

```bash
streamlit run app.py
```

Expected: browser opens at `http://localhost:8501`. Test these interactions:
1. Type "What topics are covered?" — should call `get_topics`
2. Type "Summarize what the memos say about reassessment" — should call `summarize`
3. Type "Find memos about sales disclosure" — should call `search`
4. Type "What changed in reassessment guidance between 2022 and 2025?" — should call `compare_years`
5. Type a follow-up like "Now filter that to just MEMO type" — should use conversation history

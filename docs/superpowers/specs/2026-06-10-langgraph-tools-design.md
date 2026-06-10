# LangGraph Tools Layer Design
**Date:** 2026-06-10
**Status:** Approved

---

## Context

The DLGF Memo Intelligence Pipeline has a working RAG pipeline (`rag_pipeline.py`) backed by
a Chroma vector store (13,580 chunks, 475 documents, 2022-2026). The next phase wraps that
pipeline in a LangGraph ReAct agent with a set of purpose-built tools, enabling multi-turn
conversation, summarization, insight discovery, and document search via a Streamlit UI.

---

## Architecture

Four files. `rag_pipeline.py` is untouched -- it remains the core retrieval/generation engine.

```
refactored-enigma/
├── rag_pipeline.py        # existing -- no changes
├── tools.py               # NEW: @tool functions wrapping RAGPipeline
├── agent.py               # NEW: LangGraph graph (create_react_agent)
├── app.py                 # NEW: Streamlit chat UI
└── requirements.txt       # update: langgraph, langchain-google-genai
```

### Data flow

```
User (Streamlit) -> agent.py (graph.invoke)
                 -> LLM reasons about which tool to call
                 -> tools.py calls RAGPipeline
                 -> result returned as ToolMessage
                 -> LLM produces final answer
                 -> displayed in Streamlit
```

### State

`MessagesState` (built into `create_react_agent`) -- the full conversation history is passed
to the LLM on every turn, enabling multi-turn follow-ups with no extra wiring.

### LLM for agent routing

`ChatGoogleGenerativeAI(model="gemini-2.5-flash")` from `langchain-google-genai`. Same
`GEMINI_API_KEY` from `.env`. This is separate from the generation model used inside
`RAGPipeline` -- both happen to be the same model but are independently configurable.

---

## Tools (`tools.py`)

A single `RAGPipeline` instance is created at module load time and shared across all tool
calls, avoiding repeated initialization of the Chroma and Gemini clients.

### @tool decorator

`@tool` (from `langchain_core.tools`) wraps a Python function into a LangChain Tool
object. It builds a JSON schema from the function name, type annotations, and docstring;
that schema is sent to the agent LLM as the function definition. When the LLM decides to
call a tool, LangGraph deserializes the structured response, executes the function, and
appends the return value as a ToolMessage in the conversation history. Docstrings are
therefore functional -- the agent reads them to decide when to use each tool.

### Tool inventory

| Tool | Purpose | Under the hood |
|---|---|---|
| `search` | Find documents on a topic -- returns titles/metadata, no generated answer | `pipeline.retrieve()` |
| `answer` | Full RAG Q&A with inline citations and source URLs | `pipeline.answer()` |
| `summarize` | Synthesize what the corpus says about a topic | Retrieve 12+ chunks -> custom summary prompt |
| `compare_years` | How has guidance on X changed between two time windows? | Two retrieval passes -> comparison prompt |
| `get_topics` | What major topics does this corpus cover? | Reads `cluster_summary.csv` -- no LLM call |
| `extract_quotes` | Pull notable/quotable passages on a topic | Retrieve -> prompt to surface best quotes |

### Signatures

```python
@tool
def search(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 10) -> str:
    """Find documents in the DLGF memo corpus relevant to a topic or question.
    Returns a list of matching document titles, authors, dates, relevance scores,
    and source URLs. Use this when the user wants to browse or discover documents
    rather than get a generated answer."""

@tool
def answer(query: str, year: int = None, doc_type: str = None,
           author: str = None, n: int = 6) -> str:
    """Answer a question about DLGF memos using retrieved source documents.
    Returns a grounded answer with inline citations [1], [2], etc., each including
    title, author, date, and source URL. Use this when the user asks a direct question."""

@tool
def summarize(topic: str, year: int = None, doc_type: str = None,
              author: str = None, n: int = 12) -> str:
    """Summarize what the DLGF memo corpus collectively says about a topic.
    Retrieves more documents than `answer` for broader coverage. Returns a synthesis
    with source URLs. Use this when the user wants a broad overview across documents."""

@tool
def compare_years(topic: str, year_start: int, year_end: int, n: int = 6) -> str:
    """Compare how DLGF guidance on a topic changed between two time periods.
    Retrieves documents from each time window separately, then generates a comparison
    with source URLs for both windows. Use this for trend analysis or what-changed
    questions."""

@tool
def get_topics() -> str:
    """Return the major topic clusters found in the DLGF memo corpus with document
    counts per cluster. Use this when the user wants to explore what subjects are
    covered, or to orient a new conversation."""

@tool
def extract_quotes(topic: str, year: int = None, n: int = 8) -> str:
    """Find the most notable or quotable passages about a topic in the corpus.
    Returns attributed pull quotes with document title, author, date, and source URL.
    Use this when the user wants specific language or key statements from the memos."""
```

### Citation format

All tools that return document references format citations as:

```
[N] Title -- Author (YYYY-MM-DD) | DOC_TYPE
    https://www.in.gov/dlgf/.../filename.pdf
```

The `source` URL is stored in Chroma metadata from the original scraping step and is
available on every retrieved chunk.

---

## Agent (`agent.py`)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import search, answer, summarize, compare_years, get_topics, extract_quotes

AGENT_MODEL = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0.1)
tools = [search, answer, summarize, compare_years, get_topics, extract_quotes]
graph = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
```

### Agent system prompt

Guides routing behavior. Separate from the RAG system prompt in `rag_pipeline.py`.

```
You are an expert research assistant for Indiana DLGF (Department of Local Government
Finance) memos and guidance documents, covering 2022-2026.

You have access to tools for searching, answering questions, summarizing topics,
comparing guidance across years, listing topics, and extracting notable quotes.

Guidelines:
- For direct questions, use `answer`.
- For "find documents about X", use `search`.
- For broad synthesis or overviews, use `summarize`.
- For "what changed" or trend questions, use `compare_years`.
- For orientation or topic discovery, use `get_topics` first.
- For notable language or key statements, use `extract_quotes`.
- If a query is ambiguous, ask one clarifying question before calling a tool.
- Always include source URLs when citing documents.
```

### Invocation

```python
result = graph.invoke({"messages": conversation_history})
```

The full message list grows each turn. Multi-turn follow-ups work because the LLM sees the
entire conversation history on every call.

---

## Streamlit app (`app.py`)

Minimal chat interface:

- `st.session_state.messages` holds the conversation as a list of LangChain message objects
- `st.chat_input` captures user input
- Each turn: append HumanMessage -> graph.invoke -> append AIMessage -> render
- Sidebar: optional year, doc_type, author filter fields injected as a context message at
  conversation start when set
- No streaming for the POC -- full response rendered on completion

---

## New dependencies

```
langgraph
langchain-google-genai
```

`langchain-core` is pulled in automatically by `langchain-google-genai`. No other additions
-- the project already has `google-genai`, `chromadb`, and `python-dotenv`.

---

## LLM call sites

| File | Purpose | Model constant |
|---|---|---|
| `tools.py` | Embedding at query time (via RAGPipeline) | `EMBED_MODEL` in `rag_pipeline.py` |
| `tools.py` | Generation inside answer, summarize, compare_years, extract_quotes (via RAGPipeline) | `GENERATION_MODEL` in `rag_pipeline.py` |
| `agent.py` | Agent routing / tool selection | `AGENT_MODEL` in `agent.py` |

The agent routing model and the generation model are independently swappable.

---

## Error handling

- `RAGPipeline` init errors (missing API key, missing Chroma collection) propagate at
  module load time in `tools.py` -- fail fast before the app starts.
- Empty retrieval results: each tool returns a clear "no results found" string; the agent
  surfaces this to the user rather than fabricating an answer.
- LLM API errors inside tools: propagate to the agent, which returns an error message.
- `cluster_summary.csv` missing: `get_topics` returns a graceful message rather than raising.

---

## Out of scope

- Persistent cross-session memory (MemorySaver / SQLite checkpointer)
- Streaming responses
- Authentication / multi-user sessions
- Contradiction detection between memos
- Structured data extraction (tables, rates, deadlines as JSON)
- New document detection / alerting
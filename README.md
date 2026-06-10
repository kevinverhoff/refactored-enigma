# DLGF Memo Intelligence Pipeline

A pipeline for collecting, extracting, analyzing, and querying the Indiana Department of Local Government Finance (DLGF) memo corpus -- five years of official guidance documents covering property assessment, local budgeting, tax caps, TIF districts, and more.

The full stack is live: a LangGraph-powered research assistant with a Streamlit chat UI that can answer questions, summarize documents, browse topic clusters, compare guidance across years, and surface notable pull quotes -- all grounded in the source memos with inline citations and source URLs.

---

## Data Source

**Indiana DLGF Memos (2022-2026)**
`https://www.in.gov/dlgf/memos-and-presentations/memos/`

479 documents collected across five annual pages. Documents are a mix of PDFs (memos, guidance, reports), Word documents (templates, petitions), and Excel files (levy calculation worksheets, data submissions). Text extraction yielded 475 usable documents -- 10.2 million characters total.

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Collect & download | `get_docs.py` | `docs/<year>/` + `metadata.json` |
| 2. Extract text | `ingest.py` | `documents.parquet` |
| 3. Explore | `explore.py` | `eda/` (charts + flagged docs) |
| 4. Topic modeling | `document_clustering.py` | cluster columns in parquet + `cluster_summary.csv` |
| 5. Build vector store | `build_vectorstore.py` | `chroma_db/` (Chroma collection) |
| 6. RAG pipeline | `rag_pipeline.py` | importable module + interactive CLI |
| 7. Agent + UI | `tools.py`, `agent.py`, `app.py` | LangGraph agent + Streamlit chat interface |

### `get_docs.py`
Scrapes all five year pages, extracts document links, downloads each file, and writes a `metadata.json` manifest. Parses DLGF filename conventions (`YYMMDD-Author-DocType-Title.pdf`) to extract date, author, semantic type, and title. Includes a polite crawl delay and skips files already on disk.

### `ingest.py`
Reads `metadata.json` and extracts text from each downloaded file:
- **PDF** -- `pdfplumber`, page text joined with double newlines
- **DOCX** -- `python-docx`, paragraphs + table cells
- **XLSX** -- `openpyxl`, per-sheet blocks with tab-separated cells

Outputs `documents.parquet` with columns: `doc_id`, `source`, `doc_type`, `text`, `char_count`, `retrieved_at`.

### `explore.py`
Exploratory analysis of the parquet. Produces four outputs in `eda/`, each saved as interactive HTML and static PNG. TF-IDF uses a custom token pattern to exclude pure numbers and underscore artifacts from scanned form templates.

### `document_clustering.py`
Clusters documents by topic using TF-IDF + SVD embeddings and HDBSCAN. No need to specify the number of clusters -- HDBSCAN finds natural density-based groupings, leaving genuinely ambiguous documents unassigned rather than forcing them into a cluster. Reduces to 2D with UMAP for visualization. Adds `cluster_id` and `cluster_label` to both `documents.parquet` and `metadata.json`, and writes `cluster_summary.csv` with per-cluster doc counts, type breakdowns, top terms, and date ranges.

### `build_vectorstore.py`
Chunks each document, embeds with Gemini `gemini-embedding-001` (`task_type=RETRIEVAL_DOCUMENT`), and loads into a persistent Chroma collection named `dlgf_memos`. Each chunk is stored with filterable metadata: `doc_type`, `source_year`, `memo_date`, `author`, `cluster_id`, and `cluster_label`. Re-running is safe -- already-indexed chunks are skipped. The result is 13,589 chunks across 475 documents ready for semantic retrieval.

### `rag_pipeline.py`
Takes a natural language query, retrieves the top-k most relevant chunks from Chroma (embedding the query with Gemini `gemini-embedding-001`, `task_type=RETRIEVAL_QUERY`), builds a grounded prompt with numbered source blocks, and calls Gemini `gemini-2.5-flash` to generate a cited answer. Supports metadata filters (`year`, `doc_type`, `author`) and returns a `RAGResponse` dataclass with the answer, full source list, and retrieval stats. Every LLM call site is marked with a `# LLM SWAP` comment -- see the LLM Reference section.

### `tools.py`, `agent.py`, `app.py`
The LangGraph agent layer. See the **LangGraph Agent & UI** section below.

---

## EDA & Clustering Highlights

### Document type distribution

The corpus is overwhelmingly memos (428 of 475). The remaining documents are templates, Excel attachments, and supplemental resources. The `PDF`/`DOCX`/`XLSX` fallback types are files whose names do not follow the standard DLGF naming convention.

![Document type distribution](eda/doc_type_distribution.png)

### Text length by type

Log-scale box plots reveal a heavily right-skewed distribution. The median document is ~4,500 characters; the mean is ~21,000 characters, pulled up by large Excel worksheets (one XLSX tops 2.6 million characters). No documents fell below the 500-character usefulness threshold -- extraction quality is clean across all file types.

![Text length by type](eda/text_length_by_type.png)

### TF-IDF top terms per document type

Cross-corpus TF-IDF (averaged per document type) surfaces vocabulary that is distinctive to each type -- not just common, but common *relative to the rest of the corpus*. The signal is clear:

- **MEMO** -- `ind code`, `county`, `local government`, `tax` -- statutory/administrative guidance
- **TEMPLATE** -- `excess levy`, `appeal petition`, `levy appeal` -- structured form language
- **ATTACHMENT** -- `personal income`, `employer`, `contributions`, `nonfarm` -- economic data inputs

That separation across types means a simple classifier will likely work well, and retrieval-augmented generation will be able to distinguish document intent before answering a query.

![TF-IDF top terms](eda/tfidf_top_terms.png)

### Topic clusters

18 topics discovered across 475 documents using HDBSCAN on TF-IDF + SVD embeddings, projected to 2D with UMAP. Selected highlights:

| Cluster | Top terms | Docs | What it covers |
|---------|-----------|------|----------------|
| C16 | homestead, ind code, hea | 35 | Homestead deduction legislation and guidance |
| C0  | reassessment, parcels, total number | 33 | Cyclical reassessment monthly status reports |
| C13 | excess levy, appeal, petition | 23 | Levy appeal process -- memos and petition templates cluster together |
| C17 | charter school, governing body | 16 | Charter school governance memos |
| C2  | av, cyclical reassessment | 15 | Assessed value and reassessment methodology |
| C7  | deadline, gateway, code pertains | 15 | Gateway portal submission deadlines |
| C14 | school corporation, bus replacement | 11 | School transportation and capital planning |
| C3  | tif management, commission | 10 | TIF district management |
| C15 | continuing education, hours | 9 | Assessor CE requirements and course listings |

One notable signal: C13 pulls together both MEMO and TEMPLATE documents on the same topic (excess levy appeals), confirming that the semantic clusters cut across document types rather than just recapitulating them. That cross-type coherence is what makes these clusters useful for retrieval -- a query about levy appeals will surface both the policy memo and the petition template.

41.7% of documents are currently unassigned. This is HDBSCAN being conservative -- it leaves ambiguous documents out rather than forcing them into weak clusters. Cluster parameters are being tuned.

![Cluster scatter (UMAP)](eda/clusters_scatter.png)

---

## Semantic Search

The vector store is live. Queries are embedded with `task_type=RETRIEVAL_QUERY` and matched against the indexed chunks using cosine similarity. Results are scored 0-1 (higher = more similar). Metadata filters can scope any query by year, author, doc type, or topic cluster.

A few examples from `test_chroma.py`:

**Plain semantic search** -- no keywords required, meaning drives retrieval

```
Query: "homestead deduction eligibility requirements"

score  memo_date    author   doc_type  title
0.765  2024-06-18  Shackle   MEMO      Legislation Affecting Deductions and Exemptions
0.750  2026-05-27  Cockerill MEMO      Legislation Affecting Deductions, Credits, and Exemptions
0.737  2025-09-18  Cockerill MEMO      County Option Homestead Property Tax Deferral Program
```

**Filtered by year** -- scope to recent guidance only

```
Query: "property tax assessment methodology"  [source_year >= 2024]

score  memo_date    author            doc_type  title
0.719  2025-01-03  Wood              MEMO      Ratio Study Guidance
0.713  2025-05-09  Cockerill         MEMO      2025 Revised Ag. Base Rate Certification Letter
0.711  2026-01-02  Cockerill         MEMO      2026 Agricultural Land Base Rate Packet
```

---

## RAG Pipeline

The RAG pipeline is live. Ask a question in plain English, get a grounded answer with inline citations back to the source memos.

```bash
python rag_pipeline.py                     # interactive loop
python rag_pipeline.py --year 2025         # scope every query to 2025+
python rag_pipeline.py --doc-type MEMO     # memos only
python rag_pipeline.py --author "Wood"     # one author''s memos only
python rag_pipeline.py --n 8              # retrieve 8 chunks instead of 6
```

### Example: plain Q&A

```
Q: What are the eligibility requirements for the homestead deduction?
```

> A "homestead" is defined as property located in Indiana consisting of a dwelling and up to
> one (1) acre of land immediately surrounding the dwelling [1]. It may include one (1)
> additional building not part of the dwelling, if predominantly used for residential purposes [1].
>
> For an applicant to claim the deduction:
> 1. The applicant must own or be buying under contract the real property [2], [6].
> 2. The real property must be used as their homestead (principal place of residence) [2], [6].
> 3. The applicant must have resided in Indiana for at least one (1) year before the assessment date [2], [6].
>
> [1] Legislation Affecting Deductions and Exemptions -- Shackle (2024-06-18)
> [2] Legislation Affecting Deductions, Credits, and Exemptions -- Cockerill (2026-05-27)

### Example: grounding -- honest no-answer

The model only answers from retrieved documents. Out-of-corpus questions get a clean refusal:

```
Q: What are the requirements to register to vote in Indiana?
```

> The provided sources do not contain information about the requirements to register to vote in Indiana.

---

## LangGraph Agent & UI

The agent layer wraps the RAG pipeline in a conversational interface. Start it with:

```bash
streamlit run app.py
```

The Streamlit UI opens at `http://localhost:8501`. The sidebar has optional filters (year, doc type, author) that scope tool calls for the session. The agent shows its reasoning steps -- tool calls and result previews -- live as it works, then collapses them into an "Agent reasoning" expander once the answer is ready.

### How it works

`agent.py` uses LangGraph's `create_react_agent` with Gemini `gemini-2.5-flash` as the routing model. On each turn it receives the full conversation history, reasons about which tool to call, executes it, observes the result, and loops until it has a final answer. Multi-turn follow-ups ("now filter that to 2025 only") work because the full message history is passed on every invocation.

### Tools

Seven tools are available. The agent selects among them based on query intent:

| Tool | When the agent uses it | What it does |
|---|---|---|
| `search` | "Find documents about X" | Semantic retrieval -- returns titles, dates, scores, and URLs. No generation. |
| `answer` | Direct questions | Full RAG Q&A with inline citations and source URLs |
| `summarize` | "Overview of X", "What does the corpus say about Y" | Retrieves 12 chunks, synthesizes across them |
| `compare_years` | "What changed between X and Y", trend questions | Two retrieval passes (early / late window) then a comparative analysis |
| `get_topics` | "What topics are covered", orientation queries | Returns all 18 topic clusters with doc counts and top terms from `cluster_summary.csv` |
| `browse_cluster` | "Tell me more about cluster N", drill-down after `get_topics` | Retrieves documents from a specific cluster; uses cluster top terms as the query if none provided |
| `extract_quotes` | "Key statements about X", "what did DLGF say about Y" | Retrieves chunks and prompts for notable, quotable passages with attribution |

### Example interactions

**Topic discovery + drill-down**
```
User: What topics are covered in these memos?
Agent: [calls get_topics] → lists 18 clusters

User: Tell me more about the TIF district cluster
Agent: [calls browse_cluster(3)] → lists 10 documents with URLs

User: Summarize what those memos say
Agent: [calls summarize("TIF district management")] → synthesis with citations
```

**Trend analysis**
```
User: How has guidance on homestead deductions changed since 2022?
Agent: [calls compare_years("homestead deductions", 2022, 2025)]
     → side-by-side comparison of early vs. recent guidance with source URLs
```

**Pull quotes**
```
User: What are the most important things DLGF has said about circuit breaker caps?
Agent: [calls extract_quotes("circuit breaker caps")]
     → attributed verbatim passages with document title, author, date, and URL
```

---

## Where This Is Going

The core pipeline -- collection, extraction, EDA, clustering, vector store, RAG, and agent -- is complete. What remains is refinement and hardening:

- **Cluster tuning** -- the current 18 clusters leave 41.7% of documents unassigned. HDBSCAN parameters (`min_cluster_size`, `min_samples`) are worth tuning for better coverage, especially for the large unassigned set.
- **Cross-session memory** -- the agent currently holds conversation history only within a single Streamlit session. Adding LangGraph's `MemorySaver` or a SQLite checkpointer would enable persistent memory across sessions.
- **Structured data extraction** -- pull specific values (deadlines, rates, dollar thresholds, form numbers) as structured JSON rather than prose answers.
- **Production deployment** -- containerize the app and host on a cloud provider. Swap the local Chroma store for a hosted vector database (Qdrant Cloud, Pinecone) for multi-user access.

---

## LLM Reference

This project uses Gemini models at three call sites. Every call site is marked with a `# LLM SWAP` comment. To use a different provider, change the relevant constant and replace the marked block.

| File | Purpose | Constant | Default model |
|---|---|---|---|
| `build_vectorstore.py` | Embed documents at index time | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Embed queries at retrieval time | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Generate RAG answers | `GENERATION_MODEL` | `gemini-2.5-flash` |
| `tools.py` | Generate tool responses (summarize, compare, quotes) | `GENERATION_MODEL` (from `rag_pipeline`) | `gemini-2.5-flash` |
| `agent.py` | Agent routing / tool selection | `AGENT_MODEL` | `gemini-2.5-flash` |

The embedding model and generation model are independent -- you can embed with Gemini and generate with Claude or OpenAI. The only hard constraint is that `EMBED_MODEL` must match between `build_vectorstore.py` and `rag_pipeline.py`. Changing it requires rebuilding the Chroma collection.

The agent routing model (`AGENT_MODEL` in `agent.py`) is also independent -- it must support tool/function calling but does not need to match the generation model.

### Swapping the generation model

Find the `# LLM SWAP` comment in `RAGPipeline.answer()` in `rag_pipeline.py` and replace the `generate_content` block.

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

### Swapping the agent routing model

In `agent.py`, replace `ChatGoogleGenerativeAI` with your provider''s LangChain chat model class (marked `# LLM SWAP`). The model must support tool/function calling.

**Claude** (`pip install langchain-anthropic`, add `ANTHROPIC_API_KEY` to `.env`):
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-4-8", temperature=0.1)
```

**OpenAI** (`pip install langchain-openai`, add `OPENAI_API_KEY` to `.env`):
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
```

---

## Setup

> **Python 3.12 recommended.** `langchain-core` uses Pydantic V1 compatibility shims that break on Python 3.14+.

```bash
pip install -r requirements.txt
```

### Build the corpus (one time)

```bash
# Run all steps end-to-end
python run_pipeline.py

# Or step by step
python get_docs.py            # 1. Download documents
python ingest.py              # 2. Extract text
python explore.py             # 3. EDA charts
python document_clustering.py # 4. Topic modeling
python build_vectorstore.py   # 5. Embed + index into Chroma
```

Use `python run_pipeline.py --from 4` to resume from a specific step if interrupted.

### Run the agent

```bash
streamlit run app.py
```

### Run the RAG pipeline (CLI, no UI)

```bash
python rag_pipeline.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

Requires `GEMINI_API_KEY` in `.env` for the vector store and agent. The unit tests in `tests/test_tools.py` and `tests/test_rag_pipeline.py` mock all external calls and run without an API key.

---

## Using This Pipeline With Your Own Documents

The pipeline is general enough to adapt to any corpus of PDFs, Word documents, or Excel files. Here is what to think through before you start.

### 1. Getting your documents

**Option A -- write a scraper.** If your documents live on a public website, `get_docs.py` is a reasonable template. The key pieces to adapt: the page URLs, the CSS/HTML structure of the document links, and whatever filename parsing makes sense for your naming conventions.

**Option B -- bring your own files.** Place documents under `docs/` in whatever folder structure makes sense, then write a script that produces a `metadata.json` matching the schema below.

### 2. Metadata schema

`metadata.json` is a list of objects. The fields used by downstream scripts are:

| Field | Required by | Notes |
|---|---|---|
| `url` | `build_vectorstore.py` | Unique identifier for each document. Use a file URI if there is no web URL. |
| `local_path` | `ingest.py` | Path to the file relative to `docs/` |
| `file_type` | `ingest.py` | `pdf`, `docx`, `doc`, `xlsx`, `xls` |
| `downloaded` | `ingest.py` | `true` to include the file in extraction |
| `doc_type` | `build_vectorstore.py` | Semantic type label (e.g. `MEMO`, `REPORT`, `TEMPLATE`). Used as a filterable metadata field in Chroma. |
| `source_year` | `build_vectorstore.py` | Integer year. Enables year-scoped queries. |
| `memo_date` | `build_vectorstore.py` | ISO date string (`YYYY-MM-DD`) or empty string |
| `author` | `build_vectorstore.py` | Author name or empty string |
| `title` | `build_vectorstore.py` | Document title or empty string |

All other fields (`cluster_id`, `cluster_label`) are added automatically by `document_clustering.py`.

### 3. Things to tune per corpus

**Chunk size** (`build_vectorstore.py` -- `CHUNK_SIZE`, `CHUNK_OVERLAP`)
The default is 1,000 characters with 150-character overlap. Shorter documents benefit from smaller chunks; long dense reports may do better at 1,500-2,000.

**Clustering parameters** (`document_clustering.py`)
Tuned for ~475 documents. Start with `min_cluster_size` at roughly 1-2% of your document count.

**Chroma collection name** (`build_vectorstore.py` -- `COLLECTION_NAME`)
Change `dlgf_memos` to something meaningful for your corpus.

**Agent system prompt** (`agent.py` -- `_SYSTEM_PROMPT`)
Update the corpus description so the agent knows what it''s working with.

### 4. API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

The Gemini free tier is sufficient for corpora up to a few thousand documents. Embedding costs on a paid plan are roughly $0.000025 per 1,000 characters.

### 5. Scale

| Corpus size | Notes |
|---|---|
| < 1,000 docs | Works as-is. Embedding run time is under an hour on the free tier. |
| 1,000 -- 10,000 docs | Increase `EMBED_DELAY` in `build_vectorstore.py` to avoid rate limits. |
| > 10,000 docs | Consider a hosted vector store (Qdrant Cloud, Pinecone) and a paid embedding API. |

---

## Project Structure

```
.
|-- get_docs.py                # Step 1: document collection and download
|-- ingest.py                  # Step 2: text extraction -> parquet
|-- explore.py                 # Step 3: EDA charts and flagging
|-- document_clustering.py     # Step 4: topic modeling and cluster visualization
|-- build_vectorstore.py       # Step 5: chunk, embed, and load into Chroma
|-- rag_pipeline.py            # Step 6: RAG pipeline (retrieve -> generate -> RAGResponse)
|-- tools.py                   # Step 7: LangGraph @tool functions wrapping RAGPipeline
|-- agent.py                   # Step 7: LangGraph ReAct agent graph
|-- app.py                     # Step 7: Streamlit chat UI
|-- run_pipeline.py            # Run steps 1-6 end-to-end (or --from N)
|-- test_chroma.py             # Low-level Chroma queries (no LLM)
|-- test_rag.py                # RAG demo -- queries, filters, grounding
|-- tests/
|   |-- test_rag_pipeline.py   # Unit tests for RAGPipeline (19 tests)
|   `-- test_tools.py          # Unit tests for LangGraph tools (18 tests)
|-- requirements.txt
|-- .python-version            # Python 3.12 recommended
|-- metadata.json              # Per-file metadata manifest (includes cluster assignments)
|-- documents.parquet          # Extracted text corpus (includes cluster assignments)
|-- cluster_summary.csv        # Per-cluster stats and top terms
|-- docs/
|   |-- 2022/ ... 2026/        # Downloaded source documents
|   `-- superpowers/           # Design specs and implementation plans
|-- chroma_db/                 # Persistent Chroma vector store (13,589 chunks)
`-- eda/
    |-- doc_type_distribution.{html,png}
    |-- text_length_by_type.{html,png}
    |-- tfidf_top_terms.{html,png}
    |-- clusters_scatter.{html,png}
    |-- clusters_by_doctype.{html,png}
    `-- short_docs_flagged.csv
```
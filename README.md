# DLGF Memo Intelligence Pipeline

A pipeline for collecting, extracting, analyzing, and querying the Indiana Department of Local Government Finance (DLGF) memo corpus -- five years of official guidance documents covering property assessment, local budgeting, tax caps, TIF districts, and more.

The end goal is a LangGraph-powered assistant with a web UI that can answer questions, summarize documents, and surface trends across the memo archive.

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
| 6. RAG pipeline | `rag_pipeline.py` | interactive CLI + importable module |
| 7. Agent + UI | *next* | LangGraph + web interface |

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

**Filtered by doc type** -- find templates, not memos

```
Query: "excess levy appeal petition filing"  [doc_type = TEMPLATE]

score  memo_date    author                            doc_type  title
0.793  2022-08-25  Excess Levy Appeal Shortfall      TEMPLATE  and Petition
0.787  2022-08-25  Excess Levy Appeal Correction     TEMPLATE  and Petition
0.770  2022-08-25  Excess Levy Appeal Emergency      TEMPLATE  and Petition
```

**Filtered by author** -- everything from a specific DLGF staffer

```
Query: "school bus replacement capital planning"  [author = Van Dorp]

score  memo_date    author    doc_type  title
0.758  2023-06-30  Van Dorp  MEMO      Bus Replacement Plan Templates
0.758  2022-07-13  Van Dorp  MEMO      Bus Replacement Plan and Capital Projects Plan
```

Full document retrieval is also supported -- `get_document(url)` reassembles all chunks for a given source URL back into the original text, ready to pass to an LLM for summarization or Q&A.

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

Answers cite sources inline as `[1]`, `[2]` and list them at the end. The model only uses retrieved chunks -- if the answer isn''t in the corpus, it says so.

### Example: plain Q&A

```
Q: What are the eligibility requirements for the homestead deduction?
```

> A "homestead" is defined as property located in Indiana consisting of a dwelling and up to
> one (1) acre of land immediately surrounding the dwelling [1]. It may include one (1)
> additional building not part of the dwelling, if predominantly used for residential purposes
> and not as an investment or rental property [1].
>
> For an applicant to claim the deduction:
> 1. The applicant must own or be buying under contract the real property on the date the
>    application is filed [2], [6].
> 2. The real property must be used as their homestead (principal place of residence) [2], [6].
> 3. The applicant must have resided in Indiana for at least one (1) year before the assessment
>    date for which the deduction is claimed [2], [6].
>
> [1] Legislation Affecting Deductions and Exemptions -- Shackle (2024-06-18)
> [2] Legislation Affecting Deductions, Credits, and Exemptions -- Cockerill (2026-05-27)
>
> *[6 chunks retrieved | gemini-2.5-flash]*

### Example: year filter

```
Q: What has changed recently with homestead deductions or credits?
   --year 2025
```

> Several changes have recently occurred with homestead deductions and credits:
>
> - **Homestead Standard Deduction Amount:** The deduction will phase down -- $48,000 for
>   the 2025 assessment date, $40,000 for 2026, $30,000 for 2027, down to $0 for 2030 and
>   thereafter [3].
> - **Supplemental Homestead Credit:** Taxpayers are not required to file an application;
>   auditors shall identify eligible property and apply the credit automatically [2].
> - **Deduction for 2% Circuit Breaker Credit Properties:** Added by SEA 1-2025, phasing in
>   over several years [5].
>
> [2] Legislation Affecting Deductions, Exemptions, and Credits -- Cockerill (2025-06-12)
> [3] Legislation Affecting Deductions, Credits, and Exemptions -- Cockerill (2026-05-27)
> [5] Legislation Affecting Deductions, Credits, and Exemptions -- Cockerill (2026-05-27)

### Example: grounding -- honest no-answer

The model only answers from retrieved documents. Out-of-corpus questions get a clean refusal rather than a hallucinated answer:

```
Q: What are the requirements to register to vote in Indiana?
```

> The provided sources do not contain information about the requirements to register to vote
> in Indiana.

### Programmatic use

`RAGPipeline` is importable for use in other scripts or the upcoming LangGraph agent:

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.answer("What is the deadline for Gateway TIF submissions?")
print(result.answer)      # LLM-generated answer with inline citations
print(result.sources)     # list of retrieved chunks with metadata + scores
print(result.n_retrieved) # number of chunks used

# Filtered query
result = pipeline.answer(
    "What changed with excess levy appeals?",
    year=2025,
    doc_type="MEMO",
)

# Retrieval only (no LLM call)
chunks = pipeline.retrieve("agricultural land base rate", n=4)

---

## Where This Is Going

The pipeline -- collection, extraction, EDA, clustering, vector store, and RAG -- is built. The remaining work is the agent layer and user interface. The target capabilities are:

- **Semantic search** -- find relevant memos by meaning, not just keywords
- **Document Q&A** -- ask a question, get a grounded answer with citations back to the source memo
- **Summarization** -- get a concise summary of any document or group of related documents
- **Insight generation** -- surface trends, changes over time, and notable patterns across the corpus ("How has guidance on TIF districts changed since 2022?")

Document Q&A and semantic search are working. Summarization and insight generation are next.

**LangGraph agent** *(next)*
A multi-node graph that routes each query to the right tool -- semantic search, Q&A, summarization, or trend analysis -- based on query intent. Nodes can chain: "find all 2024 memos about excess levy appeals, then summarize what changed" is a two-node traversal. The agent handles follow-up questions and multi-step reasoning naturally.

**UI** *(next)*
A lightweight web interface over the LangGraph agent. The interface will support:
- Free-text search with ranked results
- Question answering with inline source citations
- On-demand summarization of any document or cluster
- Trend and comparison queries across years or topics

---

## LLM Reference

This project uses two Gemini models. Every call site is marked with a `# LLM SWAP` comment in the source. To use a different provider, change the relevant constant and replace the marked code block.

| File | Purpose | Constant | Default model |
|---|---|---|---|
| `build_vectorstore.py` | Embed documents at index time | `EMBED_MODEL` | `gemini-embedding-001` |
| `test_chroma.py` | Embed test queries | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Embed queries at retrieval time | `EMBED_MODEL` | `gemini-embedding-001` |
| `rag_pipeline.py` | Generate answers | `GENERATION_MODEL` | `gemini-2.5-flash` |

The embedding model and generation model are independent -- you can embed with Gemini and generate with Claude or OpenAI. The only hard constraint is that `EMBED_MODEL` must be the same in `build_vectorstore.py` and `rag_pipeline.py`. Changing it requires rebuilding the Chroma collection.

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

Also update the `__init__` client initialization (marked `# LLM SWAP`) to instantiate your chosen provider's client instead of `genai.Client`.

---

## Setup

```bash
pip install -r requirements.txt

# 1. Download documents (writes docs/ and metadata.json)
python get_docs.py

# 2. Extract text (writes documents.parquet)
python ingest.py

# 3. Explore (writes eda/)
python explore.py

# 4. Cluster by topic (updates parquet + metadata.json, writes cluster_summary.csv and eda/clusters_*.{html,png})
python document_clustering.py

# 5. Build vector store (chunks, embeds with Gemini, writes chroma_db/)
python build_vectorstore.py

# Run all steps in one go (use --from N to resume from a specific step)
python run_pipeline.py
```

To start the interactive RAG pipeline after the corpus is indexed:
```bash
python rag_pipeline.py

Requires Python 3.10+. PNG export requires `kaleido`; HTML files are always written as a fallback. UMAP is used for 2D projection if `umap-learn` is installed; otherwise falls back to PCA.

---

## Using This Pipeline With Your Own Documents

The pipeline is general enough to adapt to any corpus of PDFs, Word documents, or Excel files. Here is what to think through before you start.

### 1. Getting your documents

**Option A -- write a scraper.** If your documents live on a public website, `get_docs.py` is a reasonable template. The key pieces to adapt: the page URLs, the CSS/HTML structure of the document links, and whatever filename parsing makes sense for your naming conventions. The metadata fields that matter downstream are described in the next section.

**Option B -- bring your own files.** Place documents under `docs/` in whatever folder structure makes sense, then write a script that produces a `metadata.json` matching the schema below. You do not need a scraper if the files are already on disk.

### 2. Metadata schema

`metadata.json` is a list of objects. The fields used by downstream scripts are:

| Field | Required by | Notes |
|---|---|---|
| `url` | `build_vectorstore.py`, `test_chroma.py` | Unique identifier for each document. Use a file URI if there is no web URL. |
| `local_path` | `ingest.py` | Path to the file relative to `docs/` |
| `file_type` | `ingest.py` | `pdf`, `docx`, `doc`, `xlsx`, `xls` |
| `downloaded` | `ingest.py` | `true` to include the file in extraction |
| `doc_type` | `build_vectorstore.py` | Semantic type label (e.g. `MEMO`, `REPORT`, `TEMPLATE`). Used as a filterable metadata field in Chroma. |
| `source_year` | `build_vectorstore.py` | Integer year. Enables year-scoped queries. |
| `memo_date` | `build_vectorstore.py` | ISO date string (`YYYY-MM-DD`) or empty string |
| `author` | `build_vectorstore.py` | Author name or empty string |
| `title` | `build_vectorstore.py` | Document title or empty string |

All other fields (`cluster_id`, `cluster_label`) are added automatically by `document_clustering.py`.

### 3. Text extraction

`ingest.py` handles PDF, DOCX, and XLSX out of the box. A few things to check for your corpus:

- **Scanned PDFs** -- `pdfplumber` extracts text layer only. If your PDFs are image-based scans, extraction will return empty strings. You will need OCR (e.g. `pytesseract`, AWS Textract, or Azure Document Intelligence) as a pre-processing step before running `ingest.py`.
- **Other file types** -- add an extractor function and register it in the `EXTRACTORS` dict in `ingest.py`. HTML, plain text, and Markdown are straightforward additions.
- **Short documents** -- the 500-character threshold in `explore.py` flags documents that likely extracted poorly. Tune this to your corpus.

### 4. Things to tune per corpus

**Chunk size** (`build_vectorstore.py` -- `CHUNK_SIZE`, `CHUNK_OVERLAP`)
The default is 1,000 characters with 150-character overlap. Shorter documents benefit from smaller chunks; long dense reports may do better at 1,500-2,000. Chunk size directly affects retrieval precision -- too large and a chunk contains multiple topics; too small and it lacks context.

**Clustering parameters** (`document_clustering.py` -- `HDBSCAN_MIN_CLUSTER_SIZE`, `HDBSCAN_MIN_SAMPLES`)
These were tuned for ~475 documents. Larger corpora can tolerate higher `min_cluster_size` (less noise); smaller corpora may need it lower. Start with `min_cluster_size` at roughly 1-2% of your document count.

**TF-IDF stop words and token pattern** (`explore.py`, `document_clustering.py`)
The custom token pattern strips numbers and underscores -- artifacts specific to scanned form templates in this corpus. Adjust the pattern and add domain-specific stop words if your corpus has its own noise patterns.

**Chroma collection name** (`build_vectorstore.py` -- `COLLECTION_NAME`)
Change `dlgf_memos` to something meaningful for your corpus.

### 5. API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

The Gemini free tier is sufficient for corpora up to a few thousand documents. For larger corpora or production use, enable billing -- embedding costs are roughly $0.000025 per 1,000 characters. You can also swap to a different embedding model by changing `EMBED_MODEL` in `build_vectorstore.py`; the rest of the pipeline is model-agnostic.

### 6. Scale

| Corpus size | Notes |
|---|---|
| < 1,000 docs | Works as-is. Run time for embedding is under an hour on the free tier. |
| 1,000 -- 10,000 docs | Increase `EMBED_DELAY` in `build_vectorstore.py` to avoid rate limits. Chroma handles this scale locally without issue. |
| > 10,000 docs | Consider a hosted vector store (Qdrant Cloud, Pinecone) and a paid embedding API. Local Chroma can still work but queries slow down at very high chunk counts. |

---

## Project Structure

```
.
|-- get_docs.py                # Document collection and download
|-- ingest.py                  # Text extraction -> parquet
|-- explore.py                 # EDA charts and flagging
|-- document_clustering.py     # Topic modeling and cluster visualization
|-- build_vectorstore.py       # Chunk, embed, and load into Chroma
|-- rag_pipeline.py            # RAG pipeline: retrieve -> generate -> RAGResponse
|-- run_pipeline.py            # Run the full pipeline end-to-end (or --from N)
|-- test_chroma.py             # Low-level Chroma queries (no LLM)
|-- test_rag.py                # RAG demo -- showcases queries, filters, grounding
|-- requirements.txt
|-- metadata.json              # Per-file metadata manifest (includes cluster assignments)
|-- documents.parquet          # Extracted text corpus (includes cluster assignments)
|-- cluster_summary.csv        # Per-cluster stats and top terms
|-- docs/
|   |-- 2022/
|   |-- 2023/
|   |-- 2024/
|   |-- 2025/
|   `-- 2026/
|-- chroma_db/                 # Persistent Chroma vector store (13,589 chunks)
`-- eda/
    |-- doc_type_distribution.{html,png}
    |-- text_length_by_type.{html,png}
    |-- tfidf_top_terms.{html,png}
    |-- clusters_scatter.{html,png}
    |-- clusters_by_doctype.{html,png}
    `-- short_docs_flagged.csv
```
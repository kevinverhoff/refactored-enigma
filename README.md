# DLGF Memo Intelligence Pipeline

A pipeline for collecting, extracting, and querying the Indiana Department of Local Government Finance (DLGF) memo corpus — five years of official guidance documents covering property assessment, local budgeting, tax caps, TIF districts, and more.

The end goal is a LangGraph-powered assistant with a web UI that can answer questions, summarize documents, and surface trends across the memo archive.

---

## Data Source

**Indiana DLGF Memos (2022–2026)**  
`https://www.in.gov/dlgf/memos-and-presentations/memos/`

479 documents collected across five annual pages. Documents are a mix of PDFs (memos, guidance, reports), Word documents (templates, petitions), and Excel files (levy calculation worksheets, data submissions). Text extraction yielded 475 usable documents — 10.2 million characters total.

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Collect & download | `get_docs.py` | `docs/<year>/` + `metadata.json` |
| 2. Extract text | `ingest.py` | `documents.parquet` |
| 3. Explore | `explore.py` | `eda/` (charts + flagged docs) |
| 4. Index | _next_ | vector database |
| 5. Query | _next_ | RAG + LangGraph + UI |

### `get_docs.py`
Scrapes all five year pages, extracts document links, downloads each file, and writes a `metadata.json` manifest. Parses DLGF filename conventions (`YYMMDD-Author-DocType-Title.pdf`) to extract date, author, semantic type, and title. Includes a polite crawl delay and skips files already on disk.

### `ingest.py`
Reads `metadata.json` and extracts text from each downloaded file:
- **PDF** — `pdfplumber`, page text joined with double newlines
- **DOCX** — `python-docx`, paragraphs + table cells
- **XLSX** — `openpyxl`, per-sheet blocks with tab-separated cells

Outputs `documents.parquet` with columns: `doc_id`, `source`, `doc_type`, `text`, `char_count`, `retrieved_at`.

### `explore.py`
Exploratory analysis of the parquet. Produces four outputs in `eda/`, each saved as interactive HTML and static PNG.

---

## EDA Highlights

### Document type distribution

The corpus is overwhelmingly memos (428 of 475). The remaining documents are templates, Excel attachments, and supplemental resources. The `PDF`/`DOCX`/`XLSX` fallback types are files whose names don't follow the standard DLGF convention.

![Document type distribution](eda/doc_type_distribution.png)

### Text length by type

Log-scale box plots reveal a heavily right-skewed distribution. The median document is ~4,500 characters; the mean is ~21,000 characters, pulled up by large Excel worksheets (one XLSX tops 2.6 million characters). No documents fell below the 500-character usefulness threshold — extraction quality is clean across all file types.

![Text length by type](eda/text_length_by_type.png)

### TF-IDF top terms per document type

Cross-corpus TF-IDF (averaged per document type) surfaces vocabulary that's distinctive to each type — not just common, but common *relative to the rest of the corpus*. The signal is clear:

- **MEMO** — `ind code`, `county`, `local government`, `tax` — statutory/administrative guidance
- **TEMPLATE** — `excess levy`, `appeal petition`, `levy appeal` — structured form language
- **ATTACHMENT** — `personal income`, `employer`, `contributions`, `nonfarm` — economic data inputs

That separation across types means a simple classifier will likely work well, and retrieval-augmented generation will be able to distinguish document intent before answering a query.

![TF-IDF top terms](eda/tfidf_top_terms.png)

---

## What's Next

**Vector database**  
Chunk the parquet text, embed with a sentence transformer or OpenAI embeddings, and load into a vector store (Chroma or Pinecone). Metadata — date, author, doc_type — becomes filterable at retrieval time.

**RAG pipeline**  
Retrieve the top-k most relevant chunks for a query, pass them to an LLM with a grounded prompt. Starting point: question answering over specific memos. Extend to multi-document summarization and trend extraction.

**LangGraph system**  
A multi-node graph that routes queries to the right tool — semantic search, document Q&A, summarization, or trend analysis — based on query intent. Nodes can call each other, enabling complex chains like "find all 2024 memos about excess levy appeals, then summarize the key changes."

**UI**  
A lightweight web interface on top of the LangGraph agent. Target capabilities:
- Natural language search over the full corpus
- Ask a question, get an answer with source citations
- Summarize a document or a cluster of related documents
- "What changed between 2022 and 2026 on topic X?" trend queries

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
```

Requires Python 3.10+. PNG export from Plotly requires `kaleido`; if that fails, HTML files are still written.

---

## Project Structure

```
.
├── get_docs.py          # Document collection and download
├── ingest.py            # Text extraction -> parquet
├── explore.py           # EDA charts and flagging
├── requirements.txt
├── metadata.json        # Per-file metadata manifest
├── documents.parquet    # Extracted text corpus
├── docs/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   ├── 2025/
│   └── 2026/
└── eda/
    ├── doc_type_distribution.{html,png}
    ├── text_length_by_type.{html,png}
    ├── tfidf_top_terms.{html,png}
    └── short_docs_flagged.csv
```
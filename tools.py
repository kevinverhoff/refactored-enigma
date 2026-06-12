from __future__ import annotations

import pandas as pd
from pathlib import Path

from google.genai import types
from langchain_core.tools import tool

from rag_pipeline import RAGPipeline, GENERATION_MODEL

_pipeline: RAGPipeline | None = None

_CLUSTER_CSV: Path = Path(__file__).parent / "data" / "cluster_summary.csv"

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
    'Format each as:\nQUOTE: "[exact text]"\nSOURCE: [N] Title -- Author (Date)'
)

_COMPARE_SYSTEM = (
    "You are an expert on Indiana DLGF guidance documents. "
    "You have been provided two sets of documents from different time periods. "
    "Compare how the guidance changed between these periods. "
    "Highlight what stayed the same, what changed, and any notable new requirements or removals. "
    "Be specific. At the end, list each source: [N] Title -- Author (Date) | Period"
)


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
        header = (
            f"[{idx}] {c.get('title', 'Untitled')} | {c.get('author', 'Unknown')} | "
            f"{c.get('memo_date', '')} | PERIOD: up to {year_start}"
        )
        sections.append(f"{header}\n{c['text']}")
        all_chunks.append(c)
        idx += 1
    sections.append(f"\n=== DOCUMENTS FROM {year_end} AND LATER ===\n")
    for c in late:
        header = (
            f"[{idx}] {c.get('title', 'Untitled')} | {c.get('author', 'Unknown')} | "
            f"{c.get('memo_date', '')} | PERIOD: {year_end} onwards"
        )
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
            f"Cluster {int(row['cluster_id'])} - {row['cluster_label']}: {int(row['doc_count'])} docs "
            f"| {row['date_range']}\n"
            f"  Top terms: {row['top_terms']}\n"
            f"  Doc types: {row['doc_types']}"
        )
    return "\n\n".join(lines)


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

@tool
def browse_cluster(cluster_id: int, query: str = None, n: int = 10) -> str:
    """Browse documents in a specific topic cluster by cluster ID.
    Use get_topics first to discover available cluster IDs and their subjects.
    If query is provided, performs semantic search within the cluster.
    If no query, retrieves a representative sample using the cluster top terms.
    Returns document titles, authors, dates, and source URLs."""
    p = _get_pipeline()

    cluster_info = ""
    if _CLUSTER_CSV.exists():
        df = pd.read_csv(_CLUSTER_CSV)
        row = df[df["cluster_id"] == cluster_id]
        if not row.empty:
            r = row.iloc[0]
            cluster_info = (
                f"Cluster {cluster_id}: {r['cluster_label']}\n"
                f"Documents: {int(r['doc_count'])} | Date range: {r['date_range']}\n"
                f"Top terms: {r['top_terms']}\n\n"
            )
            if query is None:
                query = str(r["top_terms"])
        else:
            if query is None:
                return f"No cluster with ID {cluster_id} found. Use get_topics to see valid cluster IDs."
    elif query is None:
        return "Cluster summary not available. Run document_clustering.py first."

    chunks = p.retrieve(query, where={"cluster_id": {"$eq": cluster_id}}, n=n)
    if not chunks:
        return f"No documents found in cluster {cluster_id}."

    lines = [cluster_info + f"Found {len(chunks)} document(s):\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {c['title']} -- {c['author']} ({c['memo_date']}) | {c['doc_type']}\n"
            f"    {c['source']}"
        )
    return "\n\n".join(lines)
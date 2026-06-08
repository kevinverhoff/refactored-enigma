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
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
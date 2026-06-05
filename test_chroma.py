"""
Quick test harness for the DLGF Chroma vector store.
Run:  python test_chroma.py
"""

import os
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types
import pandas as pd

load_dotenv()

EMBED_MODEL   = "gemini-embedding-001"
CHROMA_DIR    = "chroma_db"
COLLECTION    = "dlgf_memos"
DISPLAY_COLS  = ["score", "memo_date", "author", "doc_type", "title", "chunk_preview"]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

gemini  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma  = chromadb.PersistentClient(path=CHROMA_DIR)
col     = chroma.get_collection(COLLECTION)

print(f"Collection '{COLLECTION}': {col.count():,} chunks\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed(query: str) -> list:
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def search(query: str, n: int = 5, where: dict = None) -> pd.DataFrame:
    """Semantic search. Returns a tidy DataFrame."""
    kwargs = dict(
        query_embeddings=[embed(query)],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    r = col.query(**kwargs)
    rows = []
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        rows.append({
            "score":         round(1 - dist, 3),
            "memo_date":     meta.get("memo_date", ""),
            "author":        meta.get("author", ""),
            "doc_type":      meta.get("doc_type", ""),
            "title":         meta.get("title", "")[:60],
            "cluster_label": meta.get("cluster_label", "")[:50],
            "source":        meta.get("source", ""),
            "chunk_preview": doc[:200].replace("\n", " "),
        })
    return pd.DataFrame(rows)


def get_document(source_url: str) -> str:
    """Reassemble all chunks for a document in order."""
    r = col.get(where={"source": source_url}, include=["documents", "metadatas"])
    pairs = sorted(
        zip(r["documents"], r["metadatas"]),
        key=lambda x: x[1].get("chunk_index", 0),
    )
    return "\n\n".join(doc for doc, _ in pairs)


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 120)
    print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 1. Basic semantic search
# ---------------------------------------------------------------------------

show(
    search("homestead deduction eligibility requirements"),
    "Query: homestead deduction eligibility requirements",
)

# ---------------------------------------------------------------------------
# 2. Filtered by year (2024 and later)
# ---------------------------------------------------------------------------

show(
    search("property tax assessment methodology", n=5,
           where={"source_year": {"$gte": 2024}}),
    "Query: property tax assessment methodology  [source_year >= 2024]",
)

# ---------------------------------------------------------------------------
# 3. Filtered by doc_type
# ---------------------------------------------------------------------------

show(
    search("excess levy appeal petition filing deadline", n=5,
           where={"doc_type": "TEMPLATE"}),
    "Query: excess levy appeal  [doc_type = TEMPLATE]",
)

# ---------------------------------------------------------------------------
# 4. Filtered by author
# ---------------------------------------------------------------------------

show(
    search("school bus replacement capital planning", n=5,
           where={"author": "Van Dorp"}),
    "Query: school bus replacement  [author = Van Dorp]",
)

# ---------------------------------------------------------------------------
# 5. Filtered by cluster (homestead cluster C16)
# ---------------------------------------------------------------------------

show(
    search("homestead standard deduction credit", n=6,
           where={"cluster_id": {"$eq": 16}}),
    "Query: homestead standard deduction  [cluster_id = 16]",
)

# ---------------------------------------------------------------------------
# 6. Retrieve and print a full document
# ---------------------------------------------------------------------------

# Grab the top result from the first query and pull back the full text
top_url = search("homestead deduction eligibility requirements", n=1)["source"].iloc[0]
full_text = get_document(top_url)

print(f"\n{'=' * 70}")
print(f"  Full document: {top_url.split('/')[-1]}")
print("=" * 70)
print(full_text[:1500])
if len(full_text) > 1500:
    print(f"\n  ... ({len(full_text):,} chars total)")
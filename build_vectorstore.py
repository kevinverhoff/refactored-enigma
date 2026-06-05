"""
Chunk documents, embed with Gemini text-embedding-004, and store in Chroma.

Reads:  documents.parquet, metadata.json
Writes: chroma_db/  (persistent Chroma collection "dlgf_memos")

Re-running is safe -- documents already present in the collection are skipped.

Expects GEMINI_API_KEY in a .env file (or set as an environment variable).
"""

import json
import os
import time
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_API_KEY_VAR = "GEMINI_API_KEY"
EMBED_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 150     # characters
EMBED_BATCH_SIZE = 20   # texts per Gemini API call; kept small to respect free-tier TPM limits
EMBED_DELAY = 2.0       # seconds between batches (free tier: ~1,500,000 TPM)

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "dlgf_memos"

PARQUET_FILE = Path("documents.parquet")
METADATA_FILE = Path("metadata.json")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split text into overlapping chunks, preferring paragraph then sentence boundaries.
    Documents shorter than chunk_size are returned as a single chunk.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end == len(text):
            chunks.append(text[start:])
            break

        # Prefer to split at a paragraph, then sentence, then word boundary
        for sep in ["\n\n", "\n", ". ", " "]:
            pos = text.rfind(sep, start + overlap, end)
            if pos > start + overlap:
                end = pos + len(sep)
                break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(gemini_client, texts: list) -> list:
    """Call Gemini embed_content for a batch of texts, retrying on rate-limit errors."""
    from google.genai import errors as genai_errors

    backoff = 60
    for attempt in range(5):
        try:
            result = gemini_client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in result.embeddings]
        except genai_errors.ClientError as exc:
            if "429" in str(exc) and attempt < 4:
                print(f"\n  Rate limited -- waiting {backoff}s (attempt {attempt + 1}/5)...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


def embed_and_store(gemini_client, collection, ids, chunks, metadatas) -> None:
    """
    Embed chunks in small batches and write each batch to Chroma immediately.
    Progress is saved after every batch so re-running skips already-indexed chunks.
    """
    total = len(chunks)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch_ids  = ids[i : i + EMBED_BATCH_SIZE]
        batch_text = chunks[i : i + EMBED_BATCH_SIZE]
        batch_meta = metadatas[i : i + EMBED_BATCH_SIZE]

        print(f"  [{i + len(batch_text):>6,} / {total:,}] embedding + storing...", end="\r")
        embeddings = embed_batch(gemini_client, batch_text)

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_text,
            metadatas=batch_meta,
        )

        if i + EMBED_BATCH_SIZE < total:
            time.sleep(EMBED_DELAY)

    print()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _str(val) -> str:
    """Coerce a value to string, replacing None/NaN with empty string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def _int(val, default: int = -1) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def build_metadata_lookup(metadata_path: Path) -> dict:
    """Return {url: metadata_dict} from metadata.json."""
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {r["url"]: r for r in records if r.get("url")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Validate API key
    api_key = os.getenv(GEMINI_API_KEY_VAR)
    if not api_key:
        raise SystemExit(
            f"ERROR: {GEMINI_API_KEY_VAR} not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    gemini_client = genai.Client(api_key=api_key)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(PARQUET_FILE)
    meta_lookup = build_metadata_lookup(METADATA_FILE)
    print(f"  {len(df):,} documents")

    # Connect to Chroma
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"  Collection '{COLLECTION_NAME}': {len(existing_ids)} chunks already indexed")

    # Chunk, filter already-indexed docs, embed, store
    all_ids, all_chunks, all_embeddings, all_metadatas = [], [], [], []
    skipped = 0

    print("\nChunking documents...")
    for _, row in df.iterrows():
        doc_meta = meta_lookup.get(row["source"], {})
        chunks = chunk_text(row["text"])

        # Skip this document if its first chunk is already in the collection
        first_id = f"{row['doc_id']}_0"
        if first_id in existing_ids:
            skipped += 1
            continue

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{row['doc_id']}_{idx}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadatas.append({
                "doc_id":        row["doc_id"],
                "source":        _str(row["source"]),
                "doc_type":      _str(row["doc_type"]),
                "source_year":   _int(doc_meta.get("source_year")),
                "memo_date":     _str(doc_meta.get("memo_date")),
                "author":        _str(doc_meta.get("author")),
                "title":         _str(doc_meta.get("title")),
                "cluster_id":    _int(row.get("cluster_id"), default=-1),
                "cluster_label": _str(row.get("cluster_label")),
                "chunk_index":   idx,
                "chunk_count":   len(chunks),
            })

    print(f"  {len(df) - skipped} documents to index ({skipped} already present)")
    print(f"  {len(all_chunks):,} chunks total")

    if not all_chunks:
        print("\nNothing to index. Chroma is up to date.")
        return

    # Embed and store incrementally -- each batch is written to Chroma immediately,
    # so re-running skips already-indexed chunks automatically.
    print("\nEmbedding and storing...")
    embed_and_store(gemini_client, collection, all_ids, all_chunks, all_metadatas)

    total_in_collection = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {total_in_collection:,} chunks.")
    print(f"Chroma DB written to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
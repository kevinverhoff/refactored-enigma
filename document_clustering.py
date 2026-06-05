"""
Cluster DLGF documents using TF-IDF + SVD embeddings and HDBSCAN.

Pipeline:
  Text -> TF-IDF -> TruncatedSVD(100D) -> L2 normalize
       -> HDBSCAN -> cluster assignments
       -> UMAP/PCA(2D) -> scatter visualization

Outputs:
  documents.parquet                   -- adds cluster_id, cluster_label columns
  metadata.json                       -- adds cluster_id, cluster_label per record
  cluster_summary.csv                 -- per-cluster stats and top terms
  eda/clusters_scatter.{html,png}     -- 2D projection colored by cluster
  eda/clusters_by_doctype.{html,png}  -- same projection colored by doc_type
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

PARQUET_FILE = Path("documents.parquet")
METADATA_FILE = Path("metadata.json")
CLUSTER_SUMMARY_FILE = Path("cluster_summary.csv")
EDA_DIR = Path("eda")

SVD_COMPONENTS = 100
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3

PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embeddings(texts):
    """TF-IDF -> TruncatedSVD -> L2 normalize. Returns (embeddings, vectorizer, tfidf_matrix)."""
    vec = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
    )
    tfidf = vec.fit_transform(texts)

    n_svd = min(SVD_COMPONENTS, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    embeddings = normalize(svd.fit_transform(tfidf), norm="l2")

    return embeddings, vec, tfidf


def reduce_2d(embeddings):
    """UMAP (preferred) or PCA fallback to 2D for visualization."""
    if HAS_UMAP:
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            metric="cosine", random_state=42, low_memory=False,
        )
        return reducer.fit_transform(embeddings), "UMAP"
    xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    return xy, "PCA"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(embeddings):
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def generate_cluster_labels(df, vec, tfidf, n_terms=5):
    """Average TF-IDF scores per cluster -> top terms -> human-readable label."""
    terms = vec.get_feature_names_out()
    labels = {}
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            labels[-1] = "Unassigned"
            continue
        mask = (df["cluster_id"] == cid).values
        avg = np.asarray(tfidf[mask].mean(axis=0)).flatten()
        top_idx = avg.argsort()[-n_terms:][::-1]
        labels[cid] = " · ".join(terms[i] for i in top_idx)
    return labels


def build_cluster_summary(df, vec, tfidf):
    terms = vec.get_feature_names_out()
    rows = []
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        mask = (df["cluster_id"] == cid).values

        if cid != -1:
            avg = np.asarray(tfidf[mask].mean(axis=0)).flatten()
            top_idx = avg.argsort()[-10:][::-1]
            top_terms = ", ".join(terms[i] for i in top_idx)
        else:
            top_terms = ""

        type_counts = sub["doc_type"].value_counts()
        type_breakdown = " | ".join(f"{t} ({n})" for t, n in type_counts.items())

        # Parse dates from filenames embedded in the source URL
        date_strs = sub["source"].str.extract(r"/(\d{6})-")[0].dropna()
        if not date_strs.empty:
            parsed = pd.to_datetime(
                "20" + date_strs.str[:2] + "-" + date_strs.str[2:4] + "-" + date_strs.str[4:],
                errors="coerce",
            ).dropna()
            date_range = f"{parsed.min().date()} to {parsed.max().date()}" if not parsed.empty else ""
        else:
            date_range = ""

        rows.append({
            "cluster_id": int(cid),
            "cluster_label": sub["cluster_label"].iloc[0],
            "doc_count": len(sub),
            "doc_types": type_breakdown,
            "mean_char_count": int(sub["char_count"].mean()),
            "top_terms": top_terms,
            "date_range": date_range,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _hover_text(df):
    filename = df["source"].str.extract(r"/([^/]+\.(?:pdf|docx?|xlsx?))$")[0].fillna("")
    return filename + "<br>" + df["doc_type"] + "<br>" + df["cluster_label"]


def save_fig(fig, stem, width=1500, height=750):
    fig.write_html(str(EDA_DIR / f"{stem}.html"))
    try:
        fig.write_image(str(EDA_DIR / f"{stem}.png"), width=width, height=height, scale=1.5)
        print(f"  Saved {stem}.html + .png")
    except Exception as exc:
        print(f"  Saved {stem}.html  (PNG skipped: {exc})")


def plot_scatter_by_cluster(df, xy, method):
    hover = _hover_text(df)
    cluster_ids = sorted(df["cluster_id"].unique())
    fig = go.Figure()

    for cid in cluster_ids:
        mask = (df["cluster_id"] == cid).values
        if cid == -1:
            color, name = "#cccccc", f"Unassigned  (n={mask.sum()})"
        else:
            color = PALETTE[cid % len(PALETTE)]
            label_short = df.loc[df["cluster_id"] == cid, "cluster_label"].iloc[0][:55]
            name = f"C{cid}: {label_short}  (n={mask.sum()})"

        fig.add_trace(go.Scatter(
            x=xy[mask, 0], y=xy[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=color, size=7, opacity=0.85, line=dict(width=0.3, color="white")),
            text=hover[df["cluster_id"] == cid].values,
            hovertemplate="%{text}<extra></extra>",
        ))

    n_real = len(cluster_ids) - (1 if -1 in cluster_ids else 0)
    fig.update_layout(
        title=f"Document Clusters — {method}  ({n_real} clusters)",
        xaxis_title=f"{method}-1", yaxis_title=f"{method}-2",
        plot_bgcolor="white",
        legend=dict(x=1.01, y=1, font=dict(size=10)),
        margin=dict(r=420, t=60),
    )
    save_fig(fig, "clusters_scatter")


def plot_scatter_by_doctype(df, xy, method):
    fig = go.Figure()
    hover = (
        df["source"].str.extract(r"/([^/]+\.(?:pdf|docx?|xlsx?))$")[0].fillna("")
        + "<br>C" + df["cluster_id"].astype(str)
        + ": " + df["cluster_label"]
    )

    for i, dt in enumerate(df["doc_type"].value_counts().index):
        mask = (df["doc_type"] == dt).values
        fig.add_trace(go.Scatter(
            x=xy[mask, 0], y=xy[mask, 1],
            mode="markers",
            name=f"{dt} ({mask.sum()})",
            marker=dict(color=PALETTE[i % len(PALETTE)], size=7, opacity=0.85,
                        line=dict(width=0.3, color="white")),
            text=hover[df["doc_type"] == dt].values,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Document Types — {method}",
        xaxis_title=f"{method}-1", yaxis_title=f"{method}-2",
        plot_bgcolor="white",
        legend=dict(x=1.01, y=1),
        margin=dict(r=220, t=60),
    )
    save_fig(fig, "clusters_by_doctype")


# ---------------------------------------------------------------------------
# Dataset updates
# ---------------------------------------------------------------------------

def update_metadata(df):
    cluster_map = (
        df.set_index("source")[["cluster_id", "cluster_label"]]
        .to_dict("index")
    )
    records = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    for rec in records:
        info = cluster_map.get(rec.get("url"))
        if info:
            rec["cluster_id"] = int(info["cluster_id"])
            rec["cluster_label"] = info["cluster_label"]
        else:
            rec["cluster_id"] = None
            rec["cluster_label"] = None
    METADATA_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EDA_DIR.mkdir(exist_ok=True)
    print(f"2D projection: {'UMAP' if HAS_UMAP else 'PCA (install umap-learn for UMAP)'}")

    print("\nLoading parquet...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"  {len(df):,} documents  |  {df['doc_type'].nunique()} types")

    print("\nBuilding TF-IDF + SVD embeddings...")
    embeddings, vec, tfidf = build_embeddings(df["text"].tolist())
    print(f"  Embedding shape: {embeddings.shape}")

    print("\nClustering with HDBSCAN...")
    cluster_ids = run_hdbscan(embeddings)
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = int((cluster_ids == -1).sum())
    print(f"  {n_clusters} clusters  |  {n_noise} noise points ({n_noise / len(cluster_ids) * 100:.1f}%)")

    df["cluster_id"] = cluster_ids
    label_map = generate_cluster_labels(df, vec, tfidf)
    df["cluster_label"] = df["cluster_id"].map(label_map)

    print("\nBuilding 2D projection...")
    xy, method = reduce_2d(embeddings)

    print("\nCluster summary:")
    summary = build_cluster_summary(df, vec, tfidf)
    summary.to_csv(CLUSTER_SUMMARY_FILE, index=False)
    print(summary[["cluster_id", "cluster_label", "doc_count", "doc_types"]].to_string(index=False))

    print("\nSaving plots...")
    plot_scatter_by_cluster(df, xy, method)
    plot_scatter_by_doctype(df, xy, method)

    print("\nUpdating datasets...")
    df.to_parquet(PARQUET_FILE, index=False)
    print(f"  Updated {PARQUET_FILE}  (new columns: cluster_id, cluster_label)")
    update_metadata(df)
    print(f"  Updated {METADATA_FILE}")
    print(f"  Saved {CLUSTER_SUMMARY_FILE}")

    print(f"\nDone. {n_clusters} clusters across {len(df)} documents.")


if __name__ == "__main__":
    main()
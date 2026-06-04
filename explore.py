"""
Exploratory analysis of documents.parquet produced by ingest.py.

Outputs written to eda/:
  doc_type_distribution.{html,png}
  text_length_by_type.{html,png}
  tfidf_top_terms.{html,png}
  short_docs_flagged.csv
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

PARQUET_FILE = Path("documents.parquet")
EDA_DIR = Path("eda")
SHORT_CHAR_THRESHOLD = 500

# Plotly color palette (one per doc_type, cycling if needed)
PALETTE = [
    "#4C78A8", "#F58518", "#E45756", "#72B7B2",
    "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D",
]


def save_fig(fig: go.Figure, stem: str, width: int = 1200, height: int = 600) -> None:
    """Save a Plotly figure as both HTML and PNG."""
    html_path = EDA_DIR / f"{stem}.html"
    png_path = EDA_DIR / f"{stem}.png"

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=1.5)
        print(f"  Saved {stem}.html + .png")
    except Exception as exc:
        print(f"  Saved {stem}.html  (PNG skipped: {exc})")


# ---------------------------------------------------------------------------
# 1. Document type distribution
# ---------------------------------------------------------------------------

def plot_doc_type_distribution(df: pd.DataFrame) -> None:
    counts = df["doc_type"].value_counts().reset_index()
    counts.columns = ["doc_type", "count"]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]

    fig = go.Figure(
        go.Bar(
            x=counts["doc_type"],
            y=counts["count"],
            marker_color=colors,
            text=counts["count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Document Type Distribution",
        xaxis_title="Document Type",
        yaxis_title="Count",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E5E5"),
        margin=dict(t=60, b=60),
    )
    save_fig(fig, "doc_type_distribution", width=900, height=500)


# ---------------------------------------------------------------------------
# 2. Text length distribution by type
# ---------------------------------------------------------------------------

def plot_text_length_by_type(df: pd.DataFrame) -> None:
    types_ordered = (
        df.groupby("doc_type")["char_count"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types_ordered)}

    fig = go.Figure()
    for doc_type in types_ordered:
        subset = df[df["doc_type"] == doc_type]["char_count"]
        fig.add_trace(
            go.Box(
                y=subset,
                name=f"{doc_type} (n={len(subset)})",
                marker_color=colors[doc_type],
                boxpoints="outliers",
                line_width=1.5,
            )
        )

    # 500-char threshold line
    fig.add_hline(
        y=SHORT_CHAR_THRESHOLD,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Min useful threshold ({SHORT_CHAR_THRESHOLD:,} chars)",
        annotation_position="top right",
        annotation_font_color="crimson",
    )

    fig.update_layout(
        title="Text Length Distribution by Document Type",
        yaxis_title="Character Count",
        yaxis_type="log",
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E5E5"),
        showlegend=False,
        margin=dict(t=60, b=80),
    )
    save_fig(fig, "text_length_by_type", width=1100, height=600)


# ---------------------------------------------------------------------------
# 3. Top 30 TF-IDF terms per document type
# ---------------------------------------------------------------------------

def compute_top_tfidf(df: pd.DataFrame, n_terms: int = 30) -> dict[str, list[tuple]]:
    """
    Fit TF-IDF across all documents then average scores per doc_type.
    Returns {doc_type: [(term, avg_score), ...]} sorted descending.
    """
    usable = df[df["char_count"] >= 100].copy()

    vec = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(usable["text"])
    terms = vec.get_feature_names_out()

    top_by_type: dict[str, list[tuple]] = {}
    for doc_type in usable["doc_type"].unique():
        mask = (usable["doc_type"] == doc_type).values
        if mask.sum() == 0:
            continue
        avg = np.asarray(matrix[mask].mean(axis=0)).flatten()
        idx = avg.argsort()[-n_terms:][::-1]
        top_by_type[doc_type] = [(terms[i], float(avg[i])) for i in idx]

    return top_by_type


def plot_tfidf_top_terms(df: pd.DataFrame, n_terms: int = 30) -> None:
    top_by_type = compute_top_tfidf(df, n_terms)

    # Sort types: most documents first
    doc_counts = df["doc_type"].value_counts().to_dict()
    types_sorted = sorted(top_by_type.keys(), key=lambda t: -doc_counts.get(t, 0))

    n = len(types_sorted)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    subplot_titles = [
        f"{t}  (n={doc_counts.get(t, 0)})" for t in types_sorted
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.06,
    )

    for i, doc_type in enumerate(types_sorted):
        row = i // n_cols + 1
        col = i % n_cols + 1
        color = PALETTE[i % len(PALETTE)]

        terms_scores = top_by_type[doc_type]
        terms_list = [t for t, _ in reversed(terms_scores)]
        scores = [s for _, s in reversed(terms_scores)]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=terms_list,
                orientation="h",
                marker_color=color,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Top 30 TF-IDF Terms per Document Type",
        plot_bgcolor="white",
        height=n_rows * 560,
        width=n_cols * 520,
        margin=dict(t=80, b=40, l=40, r=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")

    save_fig(fig, "tfidf_top_terms", width=n_cols * 520, height=n_rows * 560)


# ---------------------------------------------------------------------------
# 4. Flag short documents
# ---------------------------------------------------------------------------

def flag_short_docs(df: pd.DataFrame) -> None:
    short = df[df["char_count"] < SHORT_CHAR_THRESHOLD][
        ["doc_id", "source", "doc_type", "char_count"]
    ].sort_values("char_count")

    out_path = EDA_DIR / "short_docs_flagged.csv"
    short.to_csv(str(out_path), index=False)

    if short.empty:
        print(f"  No documents below {SHORT_CHAR_THRESHOLD} chars — short_docs_flagged.csv written (empty)")
        return

    print(f"  Flagged {len(short)} documents below {SHORT_CHAR_THRESHOLD} chars -> short_docs_flagged.csv")

    counts = short["doc_type"].value_counts().reset_index()
    counts.columns = ["doc_type", "count"]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]

    fig = go.Figure(
        go.Bar(
            x=counts["count"],
            y=counts["doc_type"],
            orientation="h",
            marker_color=colors,
            text=counts["count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Short Documents (< {SHORT_CHAR_THRESHOLD} chars) by Type",
        xaxis_title="Count",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E5E5"),
        margin=dict(t=60, b=60),
    )
    save_fig(fig, "short_docs_by_type", width=800, height=400)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EDA_DIR.mkdir(exist_ok=True)

    print(f"Loading {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"  {len(df):,} documents  |  {df['doc_type'].nunique()} types  |  "
          f"total chars: {df['char_count'].sum():,}")

    print("\n[1/4] Document type distribution")
    plot_doc_type_distribution(df)

    print("\n[2/4] Text length by type")
    plot_text_length_by_type(df)

    print("\n[3/4] TF-IDF top terms")
    plot_tfidf_top_terms(df)

    print("\n[4/4] Flagging short documents")
    flag_short_docs(df)

    print(f"\nAll outputs in {EDA_DIR}/")


if __name__ == "__main__":
    main()
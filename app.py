"""Vector Database Visualized.

A tiny, fully-local Streamlit demo that turns short text chunks into TF-IDF
vectors, projects them into 2D, and lets you run a semantic search to see how a
vector database finds "nearest neighbors". No API keys, no internet, no server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Sample data: a tiny "database" of text chunks across a few themes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    title: str
    category: str
    text: str


CHUNKS: list[Chunk] = [
    # AI / machine learning
    Chunk("Neural networks", "AI / ML",
          "Neural networks learn patterns from data by adjusting weights through backpropagation."),
    Chunk("Embeddings", "AI / ML",
          "Embeddings map words and sentences into vectors so similar meanings sit close together."),
    Chunk("Training models", "AI / ML",
          "Training a machine learning model means minimizing a loss function over many examples."),
    Chunk("Transformers", "AI / ML",
          "Transformer models use attention to weigh the importance of each token in a sequence."),
    # Cooking / recipes
    Chunk("Pasta sauce", "Cooking",
          "Simmer crushed tomatoes with garlic, olive oil, and basil for a simple pasta sauce."),
    Chunk("Baking bread", "Cooking",
          "Knead flour, water, yeast, and salt, then let the dough rise before baking in the oven."),
    Chunk("Knife skills", "Cooking",
          "A sharp chef's knife and a steady grip make chopping vegetables faster and safer."),
    Chunk("Seasoning food", "Cooking",
          "Taste as you cook and balance salt, acid, fat, and heat to make flavors pop."),
    # Racing / cars
    Chunk("Engine power", "Racing / Cars",
          "Horsepower and torque determine how quickly a race car accelerates down the straight."),
    Chunk("Cornering", "Racing / Cars",
          "Braking early and hitting the apex lets a driver carry more speed through a corner."),
    Chunk("Tire grip", "Racing / Cars",
          "Tire compound and temperature control how much grip a car has on the track surface."),
    Chunk("Pit strategy", "Racing / Cars",
          "Teams plan pit stops to change tires and refuel without losing track position."),
    # Finance / investing
    Chunk("Compound interest", "Finance",
          "Compound interest grows your savings as returns are reinvested and earn returns themselves."),
    Chunk("Diversification", "Finance",
          "Spreading investments across stocks and bonds reduces the risk of any single loss."),
    Chunk("Index funds", "Finance",
          "Index funds track a market benchmark and offer low-cost, broad market exposure."),
    Chunk("Cash flow", "Finance",
          "Positive cash flow means income exceeds expenses, leaving money to save or invest."),
    # Fitness / health
    Chunk("Strength training", "Fitness",
          "Lifting weights with progressive overload builds muscle and increases overall strength."),
    Chunk("Cardio exercise", "Fitness",
          "Running and cycling raise your heart rate and improve cardiovascular endurance."),
    Chunk("Healthy diet", "Fitness",
          "A balanced diet of protein, vegetables, and whole grains supports energy and recovery."),
    Chunk("Sleep and recovery", "Fitness",
          "Good sleep lets muscles repair and is essential for recovery after hard workouts."),
]


# ---------------------------------------------------------------------------
# Modeling: TF-IDF vectors + TruncatedSVD projection to 2D. Cached so the math
# only runs once per session.
# ---------------------------------------------------------------------------


@st.cache_resource
def build_index():
    """Fit the vectorizer and 2D projection over the sample chunks."""
    texts = [f"{c.title}. {c.text}" for c in CHUNKS]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    # TruncatedSVD works directly on sparse TF-IDF matrices (PCA does not).
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(tfidf)

    df = pd.DataFrame(
        {
            "title": [c.title for c in CHUNKS],
            "category": [c.category for c in CHUNKS],
            "text": [c.text for c in CHUNKS],
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    return vectorizer, svd, tfidf, df


def wrap(text: str, width: int = 40) -> str:
    """Insert <br> tags so long hover text wraps nicely in Plotly."""
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vector Database Visualized", page_icon="🧭", layout="wide")

st.title("Vector Database Visualized")
st.markdown(
    "A tiny interactive demo that turns text chunks into vectors so you can "
    "**see how semantic search works.**"
)

# Sidebar: plain-English explanations.
with st.sidebar:
    st.header("How this works")
    st.markdown(
        """
**What is an embedding?**
An embedding turns a piece of text into a list of numbers (a *vector*). Texts
with similar meaning get similar numbers.

**What is a vector database?**
A store of those vectors that can quickly find which ones are closest to a
query vector — that's how it "searches by meaning" instead of exact keywords.

**What is nearest-neighbor search?**
Given your query's vector, it finds the stored vectors with the smallest
distance (here, highest cosine similarity) — the *nearest neighbors*.

**Why do similar ideas appear near each other?**
Because similar text produces similar vectors, related chunks land close
together in space. The clusters you see below are themes grouping themselves.
        """
    )
    st.caption(
        "Local educational demo using TF-IDF + TruncatedSVD. "
        "Not a production vector database."
    )

vectorizer, svd, tfidf, df = build_index()

# Controls.
col_query, col_k = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Search the vector database",
        placeholder="e.g. how do I get faster around a corner?",
    )
with col_k:
    top_k = st.slider("Top-k", min_value=1, max_value=8, value=3)

# Compute similarity if there's a query.
plot_df = df.copy()
plot_df["similarity"] = np.nan
top_idx: list[int] = []
query_xy = None

if query.strip():
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf)[0]
    plot_df["similarity"] = sims
    top_idx = list(np.argsort(sims)[::-1][:top_k])
    query_xy = svd.transform(q_vec)[0]

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

fig = go.Figure()

categories = sorted(plot_df["category"].unique())
palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

is_match = [i in set(top_idx) for i in range(len(plot_df))]

for cat in categories:
    cat_mask = plot_df["category"] == cat
    sub = plot_df[cat_mask]
    sub_match = [is_match[i] for i in sub.index]

    customdata = np.stack(
        [
            sub["title"],
            sub["category"],
            [wrap(t) for t in sub["text"]],
            [f"{s:.3f}" if not np.isnan(s) else "—" for s in sub["similarity"]],
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=cat,
            marker=dict(
                size=[20 if m else 12 for m in sub_match],
                color=color_map[cat],
                line=dict(
                    width=[3 if m else 0 for m in sub_match],
                    color="#111111",
                ),
                opacity=[1.0 if (m or not query.strip()) else 0.45 for m in sub_match],
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<i>%{customdata[1]}</i><br><br>"
                "%{customdata[2]}<br><br>"
                "similarity: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

# Mark the query point.
if query_xy is not None:
    fig.add_trace(
        go.Scatter(
            x=[query_xy[0]],
            y=[query_xy[1]],
            mode="markers+text",
            name="Your query",
            text=["query"],
            textposition="top center",
            marker=dict(size=18, color="#000000", symbol="star"),
            hovertemplate=f"<b>Your query</b><br>{wrap(query)}<extra></extra>",
        )
    )

fig.update_layout(
    height=560,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(title="dimension 1", zeroline=False),
    yaxis=dict(title="dimension 2", zeroline=False),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

if query.strip():
    st.subheader(f"Top {top_k} nearest chunks")
    results = (
        plot_df.loc[top_idx, ["title", "category", "similarity", "text"]]
        .rename(
            columns={
                "title": "Title",
                "category": "Category",
                "similarity": "Similarity",
                "text": "Text",
            }
        )
        .reset_index(drop=True)
    )
    results.index = results.index + 1  # rank starting at 1
    st.dataframe(
        results.style.format({"Similarity": "{:.3f}"}),
        use_container_width=True,
    )
    if results["Similarity"].max() <= 0:
        st.info(
            "No overlap with the stored chunks. Try wording your query with "
            "topics like AI, cooking, racing, finance, or fitness."
        )
else:
    st.info("Type a query above to run a semantic search and highlight the nearest chunks.")

# Vector Database Visualized

A tiny interactive demo that turns text chunks into vectors so you can **see how
semantic search works** — entirely on your machine. No API keys, no internet, no
database server.

![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## What it does

- Loads a small set of sample text chunks across a few themes (AI / ML, cooking,
  racing, finance, fitness).
- Converts each chunk into a **TF-IDF vector** with scikit-learn, then projects
  those vectors down to **2D with TruncatedSVD** so they can be plotted.
- Draws the chunks as an interactive **Plotly** scatter plot. Hover any point to
  see its title, category, and a text preview.
- Lets you type a natural-language **query**. The app vectorizes your query with
  the same vectorizer, computes **cosine similarity** against every chunk, and:
  - marks your query on the chart with a star,
  - highlights the top-k nearest chunks,
  - lists them in a table with similarity scores.
- Includes a slider for **top-k** (default 3) and a sidebar that explains
  embeddings, vector databases, and nearest-neighbor search in plain English.

## How to run

From the project folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the app in your browser (usually at http://localhost:8501).

> Using a virtual environment is recommended:
> ```bash
> python -m venv .venv
> # Windows:  .venv\Scripts\activate
> # macOS/Linux:  source .venv/bin/activate
> pip install -r requirements.txt
> streamlit run app.py
> ```

## How it works (the short version)

1. **Embed** — `TfidfVectorizer` turns each chunk of text into a numeric vector
   based on which words it uses and how distinctive those words are.
2. **Project** — real embeddings have many dimensions, so `TruncatedSVD` reduces
   them to two so we can see them. Similar chunks stay near each other.
3. **Search** — your query becomes a vector too. Cosine similarity measures the
   angle between your query vector and each chunk vector; the smallest angles
   (highest similarity) are the "nearest neighbors".

## Note

This is a **local educational visualization**, not a production vector database.
Real systems use trained neural embeddings (not TF-IDF), specialized indexes
(e.g. HNSW, IVF), and dedicated stores (FAISS, pgvector, Pinecone, Weaviate,
etc.) to search millions of vectors quickly. The goal here is to make the core
idea visible and intuitive.

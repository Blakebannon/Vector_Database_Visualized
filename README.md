# Vector Database Visualized

A tiny interactive demo that turns text chunks into vectors so you can **see how
semantic search works** — entirely on your machine. No API keys, no internet, no
database server.

![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## What it does

- Loads **~200 sample text chunks** across ten neutral, broadly relatable themes
  (Animals & Wildlife, Space & Astronomy, Food & Cooking, Weather & Climate,
  Music & Art, Books & Stories, Cities & Travel, Health & Wellness, Technology
  Basics, and Sports & Recreation), with about 20 chunks per category.
- Converts each chunk into a **TF-IDF vector** with scikit-learn, then projects
  those vectors down to **3D with TruncatedSVD** so they can be plotted.
- Draws the chunks as a large, **interactive 3D Plotly scatter plot** you can
  **click, drag, rotate, and zoom**. Each category gets its own high-contrast
  color and bold, easy-to-see markers, and hovering any point shows its title,
  category, a text preview, and (when you've searched) its similarity score.
- Lets you type a natural-language **query**. The app vectorizes your query with
  the same vectorizer, projects it into the **same 3D space**, computes **cosine
  similarity** against every chunk, and:
  - marks your query in the scene with a distinct gold-rimmed diamond marker,
  - emphasizes the top-k nearest chunks (enlarged) while fading the rest,
  - lists the matches in a table with similarity scores.
- Includes a slider for **top-k** (default 5) and a sidebar that explains
  embeddings, vector databases, and nearest-neighbor search in plain English.

Example queries to try: *planets and stars · healthy breakfast · forest animals
· rainy weather · city museums · beginner guitar · volcanoes and mountains*.

> **Note on dimensions:** real vector databases usually store *high-dimensional*
> vectors (hundreds or thousands of numbers each). This app projects them into 3D
> purely so humans can visualize the semantic space — points closer together are
> more similar in meaning.

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
   them to three so we can see them in an interactive 3D plot. Similar chunks
   stay near each other.
3. **Search** — your query becomes a vector too. Cosine similarity measures the
   angle between your query vector and each chunk vector; the smallest angles
   (highest similarity) are the "nearest neighbors".

## Note

This is a **local educational visualization**, not a production vector database.
Real systems use trained neural embeddings (not TF-IDF), specialized indexes
(e.g. HNSW, IVF), and dedicated stores (FAISS, pgvector, Pinecone, Weaviate,
etc.) to search millions of vectors quickly. The goal here is to make the core
idea visible and intuitive.

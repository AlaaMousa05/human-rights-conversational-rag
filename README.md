# Human Rights Documents – Semantic Search Engine

A semantic search engine built with LangChain that retrieves the most relevant
text passages from 163 UN human rights instruments based on the *meaning* of a
query, not keyword matching.

## Overview

The dataset is a CSV of 164 URLs pointing to international human rights documents
(UN conventions, declarations, and protocols) hosted by the University of
Minnesota Human Rights Library. The engine scrapes the text of each document,
splits it into chunks, embeds them into a vector space, and returns the closest
chunks to a user query.

## Pipeline

1. **Load** – Read the CSV of document URLs and titles.
2. **Scrape** – Fetch and clean the text of each page with `requests` + `BeautifulSoup`.
3. **Chunk** – Split documents into ~800-character overlapping chunks.
4. **Embed** – Convert each chunk into a vector using a sentence-transformer model.
5. **Index** – Store the vectors in a FAISS index saved to disk.
6. **Search** – Embed the query and retrieve the nearest chunks via cosine similarity.

## Tech Stack & Design Choices

- **LangChain** – Orchestrates the loader, text splitter, embeddings, and vector store.
- **Embedding model: `BAAI/bge-small-en-v1.5`** – An open-source, high-quality
  retrieval model that runs locally with no API key and no cost. Chosen for its
  strong performance-to-size ratio on retrieval benchmarks.
- **Vector store: FAISS** – Fast, local, and free. Persists to disk so the index
  is built once and reused for instant search.
- **UI: Streamlit** – A minimal single-page interface for the live demo.
- **Chunking: `RecursiveCharacterTextSplitter`** (size 800, overlap 120) – Keeps
  passages small enough for precise retrieval while overlap preserves context
  across chunk boundaries.
- **Normalized embeddings** – Enables cosine similarity, and L2 distance is
  converted to an intuitive 0–100% relevance score in the UI.

## Setup & Usage

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the index 
python ingest.py

# 4. Launch the search app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Example Queries

- `protections for women in prison`
- `definition of genocide`
- `rights of children in armed conflict`
- `victims right to compensation and reparation`
- `prohibition of torture`

These demonstrate semantic retrieval: results are surfaced by meaning, often
pulling relevant passages from multiple documents even when the exact query
words do not appear in the text.


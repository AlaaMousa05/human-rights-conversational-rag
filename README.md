# Human Rights Documents – RAG Assistant

A Retrieval-Augmented Generation (RAG) application built with LangChain. It
retrieves the most relevant passages from 163 UN human rights instruments and
uses a Large Language Model to generate a final answer grounded in those
passages — not in the model's own prior knowledge.

Built on top of the Week 1 semantic search engine, extended with an LLM
generation layer.

## Overview

The dataset is a CSV of 164 URLs pointing to international human rights documents
(UN conventions, declarations, and protocols) hosted by the University of
Minnesota Human Rights Library. The system scrapes the text of each document,
splits it into chunks, embeds them into a vector space, retrieves the closest
chunks to a user question, and passes them to an LLM to produce a grounded answer.

## RAG Pipeline

1. **Receive** – The user asks a question.
2. **Retrieve** – The question is embedded and the closest chunks are fetched from the FAISS index.
3. **Augment** – Retrieved chunks are combined with the question into a prompt.
4. **Generate** – The prompt is sent to an LLM, which produces an answer using only the retrieved context.
5. **Cite** – The source passages are displayed alongside the answer for traceability.

## Tech Stack & Design Choices

- **LangChain** – Orchestrates the retriever, prompt, and LLM.
- **Embedding model: `BAAI/bge-small-en-v1.5`** – Open-source, runs locally, no API key, strong retrieval performance.
- **Vector store: FAISS** – Fast, local, and persisted to disk so the index is built once and reused.
- **LLM: `llama-3.3-70b-versatile` via Groq** – A free, fast LLM API. Groq was chosen for its speed and free tier. `temperature=0` keeps answers factual and deterministic.
- **Grounded prompting** – The prompt instructs the model to answer using only the retrieved context and to say when the answer is not present, which prevents hallucination.
- **UI: Streamlit** – A minimal single-page interface for the live demo.
- **Secrets management** – The Groq API key is loaded from a local `.env` file and never committed to the repository.

## Setup & Usage

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key to a .env file
echo GROQ_API_KEY=your_key_here > .env

# 4. Build the index (run once)
python ingest.py

# 5. Launch the RAG assistant
streamlit run rag_app.py
```

Get a free Groq API key at https://console.groq.com

The app opens at `http://localhost:8501`.

## Example Questions

- `What protections exist for women prisoners?`
- `What is the definition of genocide?`
- `What rights do refugees have?`
- `What are the rights of children in armed conflict?`

Because answers are grounded in retrieved context, asking something outside the
documents (e.g. `What is the capital of France?`) makes the assistant respond
that the answer is not found in the provided documents, rather than inventing one.

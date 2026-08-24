# Human Rights RAG Assistant

A conversational Retrieval-Augmented Generation (RAG) chatbot that answers questions using United Nations human rights documents.

The application was originally built as a semantic search system and was enhanced to support:

* Conversational memory
* Multi-turn conversations
* Follow-up questions
* MMR retrieval
* Hybrid keyword + semantic retrieval
* Retrieval evaluation and comparison
* Source attribution
* LLM-generated answers using retrieved document context

---

## Project Overview

This project is a Retrieval-Augmented Generation application built around a collection of United Nations human rights documents.

The system retrieves relevant document passages and provides them to a Large Language Model (LLM), which generates an answer based only on the retrieved context.

The enhanced version makes the application conversational, allowing users to ask follow-up questions without repeating information from previous messages.

It also provides multiple retrieval strategies and an evaluation script to compare their retrieval and answer quality.

---

## Main Features

### 1. Conversational Memory

The chatbot supports multi-turn conversations.

For example:

```text
User:
What protections exist for women prisoners?

Assistant:
Women prisoners have several protections including protection from abuse,
medical support, legal aid, and protection from retaliation.

User:
What about pregnant women?

Assistant:
Pregnant women prisoners have additional protections related to healthcare,
nutrition, health monitoring, and exercise.

User:
Are there any specific protections for them?

Assistant:
Yes. Their health and diet should be monitored by a qualified health
practitioner, and they should receive adequate food, a healthy environment,
and regular exercise opportunities.
```

The chatbot uses previous conversation messages to understand follow-up questions.

---

## 2. Retrieval Methods

The application supports three retrieval approaches.

### Similarity Search

The baseline retrieval method uses vector similarity between the user query and document chunks.

The query is converted into an embedding using:

```text
BAAI/bge-small-en-v1.5
```

FAISS is then used to retrieve the most similar chunks.

Similarity Search provides a strong semantic baseline, but it can retrieve multiple chunks containing very similar information.

---

### MMR - Maximum Marginal Relevance

MMR is used to improve retrieval diversity.

Instead of selecting documents only according to similarity to the query, MMR balances:

* Relevance to the query
* Diversity among retrieved documents

This reduces redundant chunks and can provide broader context to the LLM.

MMR is useful when multiple chunks contain overlapping information.

---

### Hybrid Search

Hybrid Search combines:

* Semantic/vector search
* Keyword-based search using TF-IDF

Semantic search is useful for understanding the meaning of a query, while keyword search is useful when the query contains specific legal terminology.

For example:

```text
legal assistance
sexual abuse
pregnant women
Rule 25
gender-based violence
```

Hybrid Search can therefore improve retrieval when exact or important keywords appear in the query.

---

## Retrieval Comparison

The evaluation compares:

| Method            | Type                 | Main Strength                           |
| ----------------- | -------------------- | --------------------------------------- |
| Similarity Search | Semantic             | Strong semantic matching                |
| MMR               | Semantic + diversity | Reduces redundant results               |
| Hybrid Search     | Semantic + keyword   | Better handling of specific terminology |

The baseline Similarity Search is used as a reference point, while MMR and Hybrid Search represent retrieval enhancements.

---

## Evaluation

The project includes:

```text
evaluate.py
```

This script evaluates the retrieval methods using a set of representative human-rights questions.

The evaluation tests:

* Retrieval time
* Retrieved documents
* Retrieved document sources
* Retrieved scores
* Generated answers
* Relevance of retrieved context
* Diversity of retrieved results
* Overall answer quality

### Example evaluation questions

The evaluation includes questions such as:

```text
What protections are available for women prisoners who experience sexual abuse?

What healthcare and support should pregnant women prisoners receive?

What protections exist for women prisoners against gender-based violence and sexual harassment?

What provisions exist for mothers with children in prison?

What rights do women prisoners have regarding legal assistance?
```

---

## Evaluation Results

The evaluation demonstrated that all three retrieval methods were able to retrieve relevant United Nations human rights documents.

### Similarity Search

Similarity Search consistently retrieved the relevant United Nations Rules for the Treatment of Women Prisoners and Non-Custodial Measures for Women Offenders.

However, because it focuses primarily on semantic similarity, several retrieved chunks can contain overlapping information.

---

### MMR

MMR generally retrieved relevant documents while improving result diversity.

In some tests, MMR retrieved additional documents that were different from the top semantic matches.

This demonstrates the advantage of balancing relevance with diversity.

---

### Hybrid Search

Hybrid Search performed particularly well for questions containing specific legal terminology.

For example, for:

```text
What rights do women prisoners have regarding legal assistance?
```

Hybrid Search retrieved highly relevant results with strong retrieval scores.

It was also effective for queries containing terms such as:

```text
sexual abuse
legal assistance
gender-based violence
pregnant women
```

The evaluation therefore shows that Hybrid Search is especially useful when both semantic meaning and exact terminology are important.

---

## Example Evaluation Output

An example of the evaluation output is:

```text
Similarity Search
Retrieval time: ~0.03-0.05 seconds

MMR
Retrieval time: ~0.03-0.07 seconds

Hybrid Search
Retrieval time: ~0.02-0.05 seconds
```

The exact retrieval time may vary depending on the machine and environment.

The evaluation also showed that Hybrid Search produced highly relevant results for several terminology-heavy legal questions.

---

## Architecture

The application follows a typical RAG architecture:

```text
                 ┌──────────────────┐
                 │      User        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Streamlit     │
                 │       UI         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Conversation     │
                 │     Memory       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Retrieval Layer  │
                 ├──────────────────┤
                 │ Similarity       │
                 │ MMR              │
                 │ Hybrid Search    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   FAISS Index    │
                 │ + TF-IDF Index   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Retrieved        │
                 │ Document Chunks  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Groq LLM       │
                 │  GPT-OSS 120B    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Final Answer +   │
                 │     Sources      │
                 └──────────────────┘
```

---

## Data Pipeline

The document ingestion pipeline is implemented in:

```text
ingest.py
```

The pipeline performs the following steps:

```text
CSV document URLs
       ↓
Fetch web pages
       ↓
Clean HTML
       ↓
Create LangChain Documents
       ↓
Split documents into chunks
       ↓
Generate embeddings
       ↓
Build FAISS vector index
       ↓
Save index locally
```

---

## Document Processing

Documents are loaded from:

```text
data/human_rights_links-2.csv
```

Each document contains metadata including:

```text
title
url
```

The HTML pages are cleaned using BeautifulSoup.

Scripts and styles are removed before the content is split into chunks.

---

## Chunking

The project uses:

```text
RecursiveCharacterTextSplitter
```

with:

```text
chunk_size = 800
chunk_overlap = 120
```

This allows long documents to be divided into smaller passages suitable for retrieval.

---

## Embeddings

The application uses:

```text
BAAI/bge-small-en-v1.5
```

through Hugging Face embeddings.

Embeddings are normalized before being stored in FAISS.

---

## Vector Database

The project uses:

```text
FAISS
```

for efficient vector similarity search.

The generated index is stored in:

```text
faiss_index/
```

---

## Language Model

The application uses Groq as the LLM provider with:

```text
OpenAI GPT-OSS 120B
```

The model is instructed to answer using the retrieved document context and avoid inventing information that is not present in the provided documents.

---

## Technologies and Tools

### Programming Language

* Python

### Frontend / User Interface

* Streamlit

### RAG / LLM Framework

* LangChain

### Embeddings

* Hugging Face
* Sentence Transformers
* BAAI/bge-small-en-v1.5

### Vector Search

* FAISS

### Keyword Search

* Scikit-learn TF-IDF

### LLM

* Groq
* OpenAI GPT-OSS 120B

### Web Scraping

* Requests
* BeautifulSoup

### Data Processing

* Pandas

### Environment Management

* Python virtual environment
* python-dotenv

---

## Project Structure

```text
semantic-search/
│
├── data/
│   └── human_rights_links-2.csv
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── app.py
├── ingest.py
├── evaluate.py
├── rag_app.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AlaaMousa05/human-rights-conversational-rag
cd semantic-search
```

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Environment Variables

For local development, create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

For Streamlit Cloud deployment, add `GROQ_API_KEY` to the application's Secrets instead of committing it to the repository.

Do not commit API keys or `.env` files to GitHub.

---

## Running the Application

Start the Streamlit application:

```powershell
streamlit run rag_app.py
```

The application will open in the browser.

The user can then:

1. Enter a question.
2. Select the retrieval method.
3. Select the number of chunks.
4. Ask follow-up questions.
5. Review the generated answer.
6. Inspect the retrieved sources.

---

## Building the Index

If the FAISS index needs to be rebuilt:

```powershell
python ingest.py
```

The ingestion process will:

1. Read the document URLs.
2. Download the documents.
3. Clean the content.
4. Split the documents into chunks.
5. Generate embeddings.
6. Build the FAISS index.
7. Save the index locally.

---

## Running the Evaluation

Run:

```powershell
python evaluate.py
```

The evaluation script compares:

```text
Similarity Search
MMR
Hybrid Search
```

against the same evaluation questions.

It reports retrieval time, retrieved sources, retrieval scores, and generated answers.

---

## Conversational Example

The application supports follow-up questions.

Example:

```text
User:
What protections exist for women prisoners?

Assistant:
Women prisoners have protections against abuse, access to healthcare,
legal assistance, and protection from retaliation.

User:
What about pregnant women?

Assistant:
Pregnant women prisoners have additional protections related to healthcare,
diet, nutrition, exercise, and medical monitoring.

User:
Are there any specific protections for them?

Assistant:
Yes. Pregnant women should receive health and dietary advice monitored by
a qualified health practitioner, adequate food, a healthy environment,
and opportunities for regular exercise.
```

The second and third questions depend on the previous conversation context.

---

## RAG Prompting

The LLM receives:

```text
Conversation History
        +
Retrieved Context
        +
Current Question
```

The model is instructed to answer using the provided document context.

If the information cannot be found in the retrieved context, the assistant should indicate that it cannot find the answer in the provided documents instead of fabricating information.

---

## Why These Retrieval Enhancements?

### Why MMR?

MMR was selected because basic similarity search can return multiple highly similar chunks.

MMR improves diversity while maintaining relevance, which can provide the LLM with broader context.

### Why Hybrid Search?

Hybrid Search was selected because human-rights documents contain important legal terminology.

A query such as:

```text
legal assistance for women prisoners
```

benefits from both:

* Semantic understanding
* Exact keyword matching

Combining both approaches can improve retrieval quality for domain-specific questions.

---

## Limitations

The application has several limitations:

* Retrieval quality depends on the quality of the indexed documents.
* The evaluation dataset is relatively small.
* Retrieval scores from different retrieval algorithms are not directly comparable as absolute probabilities.
* LLM-generated answers can still vary slightly between runs.
* The application currently focuses on the indexed United Nations human-rights documents.
* The FAISS index must be rebuilt when the source documents change.

---

## Future Improvements

Possible future improvements include:

* Cross-encoder re-ranking
* Multi-query retrieval
* Contextual compression
* Better retrieval evaluation metrics
* Larger evaluation datasets
* Persistent conversation storage
* User authentication
* Conversation history management
* Streaming LLM responses
* Citation-level answer attribution
* Deployment using Docker

---

## Assignment Requirements

### Conversational Memory

* [x] Remember previous messages
* [x] Support multi-turn conversations
* [x] Answer follow-up questions using previous context
* [x] Maintain coherent conversations without requiring repetition

### Improved Retrieval

* [x] Baseline Similarity Search
* [x] MMR
* [x] Hybrid Search
* [x] Compare retrieval methods
* [x] Evaluate retrieval performance

### Submission

* [x] GitHub repository
* [x] Source code
* [x] Evaluation script
* [x] Retrieval comparison
* [x] Documentation

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Human Rights RAG", layout="wide")


@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return db


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about UN human rights documents.
Use ONLY the context below to answer the question. If the answer is not in the
context, say you cannot find it in the provided documents. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
)

st.title("Human Rights Documents - RAG Assistant")
st.caption("Ask a question. Answers are generated from retrieved document passages.")

db = load_index()
llm = load_llm()

query = st.text_input("Ask a question:", placeholder="e.g. What protections exist for women prisoners?")
k = st.slider("Number of chunks to retrieve", 1, 10, 4)

if query:
    with st.spinner("Retrieving and generating..."):
        results = db.similarity_search_with_score(query, k=k)
        context = "\n\n".join([doc.page_content for doc, _ in results])

        messages = PROMPT.format_messages(context=context, question=query)
        answer = llm.invoke(messages).content

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, (doc, score) in enumerate(results, 1):
        similarity = (1 - (score ** 2) / 2) * 100
        with st.expander(f"{i}. {doc.metadata['title']}  ({similarity:.1f}%)"):
            st.caption(f"Source: {doc.metadata['url']}")
            st.write(doc.page_content)
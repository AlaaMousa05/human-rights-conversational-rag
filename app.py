import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Human Rights Semantic Search", layout="wide")


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


st.title("Human Rights Documents - Semantic Search")
st.caption("Search across 163 UN human rights instruments by meaning, not keywords.")

db = load_index()

query = st.text_input("Enter your query:", placeholder="e.g. protections for women prisoners")
k = st.slider("Number of results", 1, 10, 5)

if query:
    results = db.similarity_search_with_score(query, k=k)
    st.subheader(f"Top {len(results)} results")
    for i, (doc, score) in enumerate(results, 1):
        similarity = (1 - (score ** 2) / 2) * 100
        with st.container():
            st.markdown(f"**{i}. {doc.metadata['title']}**")
            st.caption(f"Relevance: {similarity:.1f}%  |  Source: {doc.metadata['url']}")
            st.write(doc.page_content)
            st.divider()
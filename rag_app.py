import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Configuration

load_dotenv()

st.set_page_config(
    page_title="Human Rights RAG",
    layout="wide"
)


# Load FAISS Index

@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# Load LLM

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# Build Keyword Search Index

@st.cache_resource
def build_keyword_index(_db):
    """
    Build a TF-IDF keyword index from the same
    documents stored inside the FAISS vector store.
    """

    documents = list(
        _db.docstore._dict.values()
    )

    texts = [
        doc.page_content
        for doc in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix, documents


# Question Rewriting Prompt

QUESTION_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a question reformulation assistant.

Rewrite the user's latest question into a standalone question
that can be understood without the previous conversation.

Use the conversation history to resolve references such as:

- it
- they
- them
- this
- that
- these
- those
- the organization
- the person
- the country

If the latest question is already standalone, return it unchanged.

Do not answer the question.

Only return the rewritten standalone question.

Conversation history:
{history}

Latest question:
{question}

Standalone question:
"""
)


# Answer Prompt

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about
UN human rights documents.

Use ONLY the provided context to answer the question.

Important rules:

1. Do not use information that is not supported by
   the retrieved context.
2. Do not make up information.
3. If the answer is not contained in the context,
   say that you cannot find the answer in the
   provided documents.
4. Give a clear and concise answer.
5. When possible, mention the relevant Rule or Article
   from the provided context.

Conversation history:
{history}

Retrieved context:
{context}

Question:
{question}

Answer:
"""
)


# Session State - Conversational Memory

if "messages" not in st.session_state:
    st.session_state.messages = []


# Helper: Format Conversation History

def format_history(messages):
    """
    Convert previous chat messages into a readable
    text format for the LLM.
    """

    if not messages:
        return "No previous conversation."

    return "\n".join(
        f"{message['role'].capitalize()}: "
        f"{message['content']}"
        for message in messages
    )


# Helper: Rewrite Follow-up Question

def rewrite_question(question, history, llm):
    """
    Convert a follow-up question into a standalone
    question using the previous conversation.
    """

    if not history:
        return question

    messages = QUESTION_REWRITE_PROMPT.format_messages(
        history=history,
        question=question
    )

    response = llm.invoke(messages)

    return response.content.strip()


# Hybrid Search

def hybrid_search(
    question,
    db,
    vectorizer,
    keyword_matrix,
    documents,
    k=4,
    semantic_weight=0.5
):
    """
    Hybrid retrieval combines:

    1. Semantic search using FAISS
    2. Keyword search using TF-IDF

    The two scores are normalized and combined.
    """

    # 1. Semantic Search

    semantic_results = db.similarity_search_with_score(
        question,
        k=12
    )

    semantic_scores = {}

    for doc, distance in semantic_results:

        doc_id = id(doc)

        # Convert FAISS distance into a relevance score.
        semantic_score = 1 / (1 + distance)

        semantic_scores[doc_id] = semantic_score


    # 2. Keyword Search - TF-IDF

    query_vector = vectorizer.transform(
        [question]
    )

    keyword_scores_array = (
        keyword_matrix @ query_vector.T
    ).toarray().flatten()

    keyword_top_indices = np.argsort(
        keyword_scores_array
    )[::-1][:12]

    keyword_scores = {}

    for index in keyword_top_indices:

        document = documents[index]

        doc_id = id(document)

        keyword_scores[doc_id] = float(
            keyword_scores_array[index]
        )


    # 3. Normalize Scores

    def normalize(scores):

        if not scores:
            return {}

        max_score = max(scores.values())

        if max_score == 0:
            return {
                doc_id: 0
                for doc_id in scores
            }

        return {
            doc_id: score / max_score
            for doc_id, score in scores.items()
        }


    semantic_scores = normalize(
        semantic_scores
    )

    keyword_scores = normalize(
        keyword_scores
    )


    # 4. Combine Semantic + Keyword Scores

    combined_scores = {}

    all_document_ids = (
        set(semantic_scores)
        | set(keyword_scores)
    )

    document_lookup = {
        id(doc): doc
        for doc in documents
    }

    for doc_id in all_document_ids:

        semantic_score = semantic_scores.get(
            doc_id,
            0
        )

        keyword_score = keyword_scores.get(
            doc_id,
            0
        )

        combined_score = (
            semantic_weight * semantic_score
            + (1 - semantic_weight) * keyword_score
        )

        combined_scores[doc_id] = combined_score


    # 5. Rank Documents

    ranked_documents = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:k]


    # 6. Return Documents + Scores

    return [
        (
            document_lookup[doc_id],
            score
        )
        for doc_id, score in ranked_documents
    ]


# Retrieval Function

def retrieve_documents(
    db,
    question,
    method,
    k,
    vectorizer=None,
    keyword_matrix=None,
    documents=None
):
    """
    Select the retrieval strategy.
    """

    # Basic Similarity Search

    if method == "Similarity Search":

        results = db.similarity_search_with_score(
            question,
            k=k
        )

        return [
            (doc, score)
            for doc, score in results
        ]


    # MMR

    if method == "MMR":

        documents_result = (
            db.max_marginal_relevance_search(
                question,
                k=k,
                fetch_k=12,
                lambda_mult=0.5
            )
        )

        return [
            (doc, None)
            for doc in documents_result
        ]


    # Hybrid Search

    if method == "Hybrid Search":

        if (
            vectorizer is None
            or keyword_matrix is None
            or documents is None
        ):
            raise ValueError(
                "Keyword search index is not initialized."
            )

        return hybrid_search(
            question=question,
            db=db,
            vectorizer=vectorizer,
            keyword_matrix=keyword_matrix,
            documents=documents,
            k=k,
            semantic_weight=0.5
        )


    raise ValueError(
        f"Unknown retrieval method: {method}"
    )


# Load Resources

db = load_index()

llm = load_llm()

(
    vectorizer,
    keyword_matrix,
    all_documents
) = build_keyword_index(db)


# Application Header

st.title(
    "Human Rights RAG Assistant"
)

st.caption(
    "A conversational RAG chatbot that answers "
    "questions using UN human rights documents."
)


# Sidebar

with st.sidebar:

    st.header("Retrieval")

    retrieval_method = st.radio(
        "Choose retrieval method:",
        [
            "Similarity Search",
            "MMR",
            "Hybrid Search"
        ]
    )

    k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.write(
        f"Messages: "
        f"{len(st.session_state.messages)}"
    )


# Display Conversation History

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )


# Chat Input

query = st.chat_input(
    "Ask a question about human rights documents..."
)


# Process User Question

if query:

    # Display User Message

    with st.chat_message("user"):
        st.write(query)


    # Save User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # Previous Conversation

    previous_messages = (
        st.session_state.messages[:-1]
    )

    history = format_history(
        previous_messages
    )


    # Generate Assistant Response

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving and generating..."
        ):

            # Step 1: Rewrite Follow-up Question

            standalone_question = rewrite_question(
                question=query,
                history=history,
                llm=llm
            )


            # Step 2: Retrieve Documents

            results = retrieve_documents(
                db=db,
                question=standalone_question,
                method=retrieval_method,
                k=k,
                vectorizer=vectorizer,
                keyword_matrix=keyword_matrix,
                documents=all_documents
            )


            # Step 3: Build Context

            context = "\n\n".join(
                doc.page_content
                for doc, _ in results
            )


            # Step 4: Generate Answer

            messages = ANSWER_PROMPT.format_messages(
                history=history,
                context=context,
                question=standalone_question
            )

            response = llm.invoke(
                messages
            )

            answer = response.content


            # Display Answer

            st.write(answer)


    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# Sources

if query and results:

    st.subheader("Sources")

    for i, (doc, score) in enumerate(
        results,
        start=1
    ):

        title = doc.metadata.get(
            "title",
            "Unknown source"
        )

        url = doc.metadata.get(
            "url",
            ""
        )

        if score is not None:
            score_text = f" — score: {score:.4f}"
        else:
            score_text = ""

        with st.expander(
            f"{i}. {title}{score_text}"
        ):

            if url:
                st.caption(
                    f"Source: {url}"
                )

            st.write(
                doc.page_content
            )
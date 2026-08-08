import time
import numpy as np

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Configuration

load_dotenv()

INDEX_PATH = "faiss_index"

K = 4

RETRIEVAL_METHODS = [
    "Similarity Search",
    "MMR",
    "Hybrid Search",
]


# Evaluation Questions

QUESTIONS = [
    "What protections are available for women prisoners who experience sexual abuse?",

    "What healthcare and support should pregnant women prisoners receive?",

    "What protections exist for women prisoners against gender-based violence and sexual harassment?",

    "What provisions exist for mothers with children in prison?",

    "What rights do women prisoners have regarding legal assistance?",
]

# Load Embeddings + FAISS

def load_index():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("Loading FAISS index...")

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db

# Load LLM

def load_llm():
    print("Loading Groq LLM...")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


# Build TF-IDF Keyword Index

def build_keyword_index(db):
    print("Building TF-IDF keyword index...")

    documents = list(
        db.docstore._dict.values()
    )

    texts = [
        document.page_content
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix, documents


# Hybrid Search

def hybrid_search(
    question,
    db,
    vectorizer,
    keyword_matrix,
    documents,
    k=4,
    semantic_weight=0.5,
):
    """
    Hybrid Search combines:

    Semantic Search using FAISS
    +
    Keyword Search using TF-IDF
    """

    # Semantic Search

    semantic_results = db.similarity_search_with_score(
        question,
        k=12,
    )

    semantic_scores = {}

    for doc, distance in semantic_results:

        doc_id = id(doc)

        semantic_score = 1 / (1 + distance)

        semantic_scores[doc_id] = semantic_score


    # Keyword Search

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


    # Normalize Scores

    def normalize(scores):

        if not scores:
            return {}

        max_score = max(
            scores.values()
        )

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


    # Combine Scores

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
            0,
        )

        keyword_score = keyword_scores.get(
            doc_id,
            0,
        )

        combined_score = (
            semantic_weight * semantic_score
            + (1 - semantic_weight) * keyword_score
        )

        combined_scores[doc_id] = combined_score


    # Rank

    ranked_documents = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]


    return [
        (
            document_lookup[doc_id],
            score,
        )
        for doc_id, score in ranked_documents
    ]


# Retrieval

def retrieve_documents(
    db,
    question,
    method,
    k,
    vectorizer=None,
    keyword_matrix=None,
    documents=None,
):
    """
    Run one of the three retrieval strategies.
    """

    # Similarity Search

    if method == "Similarity Search":

        results = db.similarity_search_with_score(
            question,
            k=k,
        )

        return results


    # MMR

    if method == "MMR":

        results = db.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=12,
            lambda_mult=0.5,
        )

        return [
            (doc, None)
            for doc in results
        ]


    # Hybrid Search

    if method == "Hybrid Search":

        return hybrid_search(
            question=question,
            db=db,
            vectorizer=vectorizer,
            keyword_matrix=keyword_matrix,
            documents=documents,
            k=k,
            semantic_weight=0.5,
        )


    raise ValueError(
        f"Unknown retrieval method: {method}"
    )


# Answer Generation

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about
UN human rights documents.

Use ONLY the provided context to answer the question.

Do not use information outside the context.

If the answer is not contained in the context,
say that you cannot find the answer in the
provided documents.

Give a clear and concise answer.

Question:
{question}

Context:
{context}

Answer:
"""
)


def generate_answer(
    llm,
    question,
    results,
):
    context = "\n\n".join(
        doc.page_content
        for doc, _ in results
    )

    messages = ANSWER_PROMPT.format_messages(
        question=question,
        context=context,
    )

    response = llm.invoke(
        messages
    )

    return response.content


# Evaluate One Question

def evaluate_question(
    question,
    db,
    llm,
    vectorizer,
    keyword_matrix,
    documents,
):
    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    results_by_method = {}

    for method in RETRIEVAL_METHODS:

        print()
        print("-" * 80)
        print(method)
        print("-" * 80)

        # Retrieval timing

        start_time = time.perf_counter()

        results = retrieve_documents(
            db=db,
            question=question,
            method=method,
            k=K,
            vectorizer=vectorizer,
            keyword_matrix=keyword_matrix,
            documents=documents,
        )

        retrieval_time = (
            time.perf_counter()
            - start_time
        )

        results_by_method[method] = results

        print(
            f"Retrieval time: "
            f"{retrieval_time:.4f} seconds"
        )

        print(
            f"Documents retrieved: "
            f"{len(results)}"
        )


        # Sources

        print()
        print("Retrieved Sources:")

        for i, (doc, score) in enumerate(
            results,
            start=1,
        ):

            title = doc.metadata.get(
                "title",
                "Unknown source",
            )

            url = doc.metadata.get(
                "url",
                "",
            )

            print(
                f"\n{i}. {title}"
            )

            if url:
                print(
                    f"   URL: {url}"
                )

            if score is not None:
                print(
                    f"   Score: {score:.4f}"
                )


        # Generate Answer

        print()
        print("Generating answer...")

        answer_start = time.perf_counter()

        answer = generate_answer(
            llm=llm,
            question=question,
            results=results,
        )

        answer_time = (
            time.perf_counter()
            - answer_start
        )

        print(
            f"Answer generation time: "
            f"{answer_time:.4f} seconds"
        )

        print()
        print("ANSWER")
        print("-" * 40)
        print(answer)

    return results_by_method


# Main

def main():

    print("=" * 80)
    print("HUMAN RIGHTS RAG - RETRIEVAL EVALUATION")
    print("=" * 80)

    print()

    # Load resources

    db = load_index()

    llm = load_llm()

    vectorizer, keyword_matrix, documents = (
        build_keyword_index(db)
    )

    print()
    print(
        f"Total documents in index: "
        f"{len(documents)}"
    )

    print(
        f"Number of evaluation questions: "
        f"{len(QUESTIONS)}"
    )

    print(
        f"Chunks retrieved per method: "
        f"{K}"
    )


    # Run evaluation

    for question_number, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print()
        print()
        print("#" * 80)
        print(
            f"TEST {question_number}/{len(QUESTIONS)}"
        )
        print("#" * 80)

        evaluate_question(
            question=question,
            db=db,
            llm=llm,
            vectorizer=vectorizer,
            keyword_matrix=keyword_matrix,
            documents=documents,
        )


    # Finished

    print()
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Compared retrieval methods:"
    )

    for method in RETRIEVAL_METHODS:
        print(
            f"  - {method}"
        )

    print()
    print(
        "Use the retrieved sources and answers above "
        "to compare relevance, diversity, and answer quality."
    )


if __name__ == "__main__":
    main()

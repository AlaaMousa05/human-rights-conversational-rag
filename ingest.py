import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARCHIVE_NOTE = (
    "This site was archived on 2023-02-01 and is no longer receiving updates. "
    "Links, accessibility, and other functionality may be limited."
)


def fetch_clean_text(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = text.replace(ARCHIVE_NOTE, "")
        return text
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return ""


def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    docs = []
    for i, row in df.iterrows():
        url, title = row["URL"], row["Title"]
        print(f"[{i + 1}/{len(df)}] Fetching: {title}")
        text = fetch_clean_text(url)
        if len(text) > 200:
            docs.append(Document(
                page_content=text,
                metadata={"title": title, "url": url}
            ))
    print(f"\nFetched {len(docs)} documents")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Building index (first run downloads the model ~130MB)...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("Index saved to faiss_index/")


if __name__ == "__main__":
    docs = load_documents("data/human_rights_links-2.csv")
    chunks = split_documents(docs)
    build_index(chunks)
    print("\nDone. Ready to search.")
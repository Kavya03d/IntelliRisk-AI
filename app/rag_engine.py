import os
import chromadb
from chromadb.utils import embedding_functions
from knowledge_base import ALL_DOCUMENTS

CHROMA_PATH  = "/tmp/chroma_db"   # use /tmp on Streamlit Cloud
COLLECTION_NAME = "intellirisk_knowledge"

def get_embedding_function():
    # Use chromadb's built-in default embeddings — very lightweight
    # No sentence-transformers needed, uses onnx under the hood
    return embedding_functions.DefaultEmbeddingFunction()

def build_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = get_embedding_function()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        if collection.count() == len(ALL_DOCUMENTS):
            return collection
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids      =[doc["id"]      for doc in ALL_DOCUMENTS],
        documents=[doc["content"] for doc in ALL_DOCUMENTS],
        metadatas=[{"title": doc["title"]} for doc in ALL_DOCUMENTS],
    )
    return collection


def retrieve_relevant_docs(query: str, n_results: int = 3) -> list:
    collection = build_vector_store()
    results    = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    docs = []
    for i, doc_text in enumerate(results["documents"][0]):
        docs.append({
            "title"  : results["metadatas"][0][i]["title"],
            "content": doc_text.strip()
        })
    return docs

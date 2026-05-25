import os
import chromadb
from chromadb.utils import embedding_functions
from knowledge_base import ALL_DOCUMENTS

# ── ChromaDB setup ────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "intellirisk_knowledge"

def get_embedding_function():
    """Use sentence-transformers (free, no API key needed for embeddings)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def build_vector_store():
    """
    Build the ChromaDB vector store from ALL_DOCUMENTS.
    Called once on first run — takes about 10-20 seconds.
    After that, loads from disk instantly.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = get_embedding_function()

    # If collection already exists with all docs, skip rebuild
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        if collection.count() == len(ALL_DOCUMENTS):
            return collection
        # Wrong count — delete and rebuild
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[doc["id"] for doc in ALL_DOCUMENTS],
        documents=[doc["content"] for doc in ALL_DOCUMENTS],
        metadatas=[{"title": doc["title"]} for doc in ALL_DOCUMENTS],
    )
    return collection


def retrieve_relevant_docs(query: str, n_results: int = 3) -> list[dict]:
    """
    Search the vector store for chunks most relevant to the query.
    Returns a list of dicts with 'title' and 'content'.
    """
    collection = build_vector_store()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    docs = []
    for i, doc_text in enumerate(results["documents"][0]):
        docs.append({
            "title": results["metadatas"][0][i]["title"],
            "content": doc_text.strip()
        })
    return docs

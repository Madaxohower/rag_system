""" src/vectorstore.py - ChromaDB vector store wrapper (chunks + their embeddings to disk, and query for the top-k most wimilar chunks) """
 
import os
import sys
 
import chromadb
from chromadb.config import Settings
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, TOP_K_RESULTS
from src.embedder import embed_texts, embed_query
 
 
def _get_client() -> chromadb.PersistentClient: #Return persistent chromaDB client
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
 
 
def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection: #Get or create the document collection using cosine similarity
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
 
 
# ─------------------------------------------------------ WRITE -------------------------------------------------------
 
def add_chunks(chunks: list[dict]) -> None: #Embed and store a list of chunk dicts into ChtomaDB
    client     = _get_client()
    collection = _get_collection(client)
 
    # Filter out already-stored chunks
    existing_ids = set(collection.get(ids=[c["chunk_id"] for c in chunks])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]
 
    if not new_chunks:
        print("  ℹ All chunks already indexed — nothing to add.")
        return
 
    texts      = [c["text"]     for c in new_chunks]
    ids        = [c["chunk_id"] for c in new_chunks]
    metadatas  = [{"source": c["source"], "page": str(c["page"])} for c in new_chunks]
 
    print(f"  Embedding {len(new_chunks)} chunks ...")
    embeddings = embed_texts(texts)
 
    collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
    )
    print(f"  ✓ Stored {len(new_chunks)} chunks in ChromaDB")
 
 
def collection_size() -> int: #Return how many chunks are currently stored
 
    client     = _get_client()
    collection = _get_collection(client)
    return collection.count()
 
 
def clear_collection() -> None: #Delete all documents from the collection (full-reindex)
 
    client = _get_client()
    client.delete_collection(CHROMA_COLLECTION)
    print("  ✓ Vector store cleared.")
 
# ─------------------------------------------------------ READ -------------------------------------------------------
 
def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]: #Finder of the top-k most similar to the query. returns(chunk text, source, page, score)
 
    client     = _get_client()
    collection = _get_collection(client)
 
    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run ingest.py first to index your documents."
        )
 
    query_vector = embed_query(query)
 
    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
 
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   text,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page",   "?"),
            "score":  round(dist, 4),
        })
 
    return chunks
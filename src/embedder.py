""" src/embedder.py — Local embedding model. """
 
import os
import sys
from functools import lru_cache
 
from sentence_transformers import SentenceTransformer
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL
 
 
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:                        #Load embedding model
    print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Embedding model ready  (dim={model.get_sentence_embedding_dimension()})")
    return model
 
 
def embed_texts(texts: list[str]) -> list[list[float]]:         #Convert list of text strings to embedding vectors
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,                              # unit vectors → cosine sim = dot product
        convert_to_numpy=True,
    )
    return vectors.tolist()
 
 
def embed_query(query: str) -> list[float]:                     #Embed sing query string(Convenience wrapper around embed_texts)
    return embed_texts([query])[0]
 
 
def get_embedding_dimension() -> int:                           #Return vector dimension of the loaded model
    return _get_model().get_sentence_embedding_dimension()
""" config.py — Central configuration loader. """
 
import os
from dotenv import load_dotenv
 
 
load_dotenv()                                                                   # Load .env file from project root
 
 
def _require(key: str) -> str:                                                  # Read a required env variable or raise a clear error.
 
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your values."
        )
    return value
 
 
# ----------------------------------- Groq LLM API ----------------------------------
GROQ_API_KEY: str = _require("GROQ_API_KEY")
GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama3-70b-8192")
 
# --------------------------------- Local Vector Store ------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
CHROMA_COLLECTION:  str = "rag_documents"
 
# ---------------------------- Embedding model (runs locally) -----------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
 
# ----------------------------------- Chunking --------------------------------------
CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
 
# ----------------------------------- Retrieval -------------------------------------
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
 
# -------------------------- Supported document extensions --------------------------
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt", ".md")
 
# ------------------------------- RAG prompt template -------------------------------
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Never make up information. Always be concise and accurate.
"""
 
RAG_PROMPT_TEMPLATE = """\
Context:
{context}
 
Question: {question}
 
Answer based only on the context above:
"""
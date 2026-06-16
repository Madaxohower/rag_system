""" src/chunker.py — Text chunker. """
 
import os
import sys
from typing import Any
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHUNK_SIZE, CHUNK_OVERLAP
 
 
def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]: # Split text into chunks of ~chunk_zie characters with overlap
                                                                            
    separators = ["\n\n", "\n", ". ", " "]                              # Prefer splitting on double newlines (paragraphs), then single newlines
 
    chunks   = []
    start    = 0
    text_len = len(text)
 
    while start < text_len:
        end = min(start + chunk_size, text_len)
 
        
        if end < text_len:                                              # If not at the end, try to find a clean break point
            best_break = end    
            for sep in separators:
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    best_break = pos + len(sep)
                    break
            end = best_break
 
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
 
        
        start = end - overlap if end - overlap > start else end         # Move forward, stepping back by overlap
 
    return chunks
 
 
def chunk_documents(
    docs: list[dict],
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:                                              # Take a list of page dicts and return a list of chunk dicts.
    """
    Each chunk dict contains:
        text     — the chunk text
        source   — original file path
        page     — page number in the source document
        chunk_id — unique string ID  (used as ChromaDB document id)
    """
    chunks: list[dict] = []
 
    for doc in docs:
        text_chunks = _split_text(doc["text"], chunk_size, overlap)
        source_stem = os.path.basename(doc["source"]).replace(" ", "_")
 
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{source_stem}__p{doc['page']}__c{idx}"
            chunks.append({
                "text":     chunk_text,
                "source":   doc["source"],
                "page":     doc["page"],
                "chunk_id": chunk_id,
            })
 
    return chunks
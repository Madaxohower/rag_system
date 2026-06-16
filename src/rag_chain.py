""" src/rag_chain.py — Core RAG pipeline. Ties retrieval and generation together into a single ask() call. """
 
import os
import sys
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TOP_K_RESULTS, RAG_PROMPT_TEMPLATE
from src.vectorstore import retrieve
from src.llm import generate, generate_stream
 
 
def _build_prompt(query: str, chunks: list[dict]) -> str: # Adds retrived document content to the prompt.
 
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = os.path.basename(chunk["source"])
        page   = chunk["page"]
        context_blocks.append(
            f"[{i}] (source: {source}, page {page})\n{chunk['text']}"
        )
 
    context = "\n\n---\n\n".join(context_blocks)
 
    return RAG_PROMPT_TEMPLATE.format(
        context  = context,
        question = query,
    )
 
 
def ask(
    query:   str,
    top_k:   int  = TOP_K_RESULTS,
    stream:  bool = False,
) -> dict:                                    # run the full RAG piple line for a single query
 
    chunks = retrieve(query, top_k=top_k)     # Step 1 — Retrieve
 
    prompt = _build_prompt(query, chunks)     # Step 2 — Build prompt
 
    if stream:                                # Step 3 — Generate
        answer_parts = []
        for token in generate_stream(prompt):
            print(token, end="", flush=True)
            answer_parts.append(token)
        print()                               # newline after stream ends
        answer = "".join(answer_parts)
    else:
        answer = generate(prompt)
 
    return {
        "answer":  answer,
        "sources": chunks,
        "prompt":  prompt,
    }
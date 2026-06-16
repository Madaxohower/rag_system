""" src/llm.py — Groq LLM wrapper. """
 
import os
import sys
 
from groq import Groq
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT
 
 
def _get_client() -> Groq:                              #Return a configured Groq client.
    return Groq(api_key=GROQ_API_KEY)
 
 
def generate(
    prompt:      str,
    system:      str   = SYSTEM_PROMPT,
    model:       str   = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens:  int   = 1024,
) -> str:
    """
    Send a prompt to Groq and return the response text.
 
    Args:
        prompt:      The user-facing prompt (context + question already injected).
        system:      System instruction telling the model how to behave.
        model:       Groq model name (default from config).
        temperature: Lower = more factual/deterministic (0.0–1.0).
        max_tokens:  Max tokens in the generated response.
 
    Returns:
        The LLM's response as a plain string.
    """
    client = _get_client()
 
    chat_completion = client.chat.completions.create(
        model       = model,
        temperature = temperature,
        max_tokens  = max_tokens,
        messages    = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
 
    return chat_completion.choices[0].message.content.strip()
 
 
def generate_stream(                                       # Stream the groq response token by token   
    prompt:      str,                                       
    system:      str   = SYSTEM_PROMPT,
    model:       str   = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens:  int   = 1024,
):
    """
    Usage:
        for chunk in generate_stream(prompt):
            print(chunk, end="", flush=True)
    """
    client = _get_client()
 
    stream = client.chat.completions.create(
        model       = model,
        temperature = temperature,
        max_tokens  = max_tokens,
        stream      = True,
        messages    = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
 
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
""" src/loader.py — Document loading utility. """
 
import os
from pathlib import Path
from typing import Generator
import fitz  
from docx import Document
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SUPPORTED_EXTENSIONS
 
 
# ----------------------------------------------- individual loaders --------------------------------------------------------
 
def _load_pdf(path: Path) -> Generator[dict, None, None]:                   # One dict per page from a PDF
 
    doc = fitz.open(str(path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            yield {
                "text":   text,
                "source": str(path),
                "page":   page_num,
            }
    doc.close()
 
 
def _load_docx(path: Path) -> Generator[dict, None, None]:                  # Yield the full text of a DOCX as a single document   
 
    doc = Document(str(path))
    full_text = "\n".join(
        para.text.strip()
        for para in doc.paragraphs
        if para.text.strip()
    )
    if full_text:
        yield {
            "text":   full_text,
            "source": str(path),
            "page":   1,
        }
 
  
def _load_text(path: Path) -> Generator[dict, None, None]:                  #Yiel a TXT or MD files as a single document
 
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield {
            "text":   text,
            "source": str(path),
            "page":   1,
        }
 
 
# ---------------------------------------------- Dispatch ---------------------------------------------------------
_LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".txt":  _load_text,
    ".md":   _load_text,
}
 
 
def load_file(path: str | Path) -> list[dict]:                              #Load a single file and return a list of page dicts.
 
    path = Path(path)
    ext  = path.suffix.lower()
 
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}  (supported: {SUPPORTED_EXTENSIONS})")
 
    loader = _LOADERS[ext]
    return list(loader(path))
 
 
def load_directory(directory: str | Path) -> list[dict]:                    # Recursively load all supported documents from directory then return a flat list of page dicts
 
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")
 
    docs: list[dict] = []
    found = list(directory.rglob("*"))
    supported = [f for f in found if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
 
    if not supported:
        raise FileNotFoundError(
            f"No supported documents found in {directory}\n"
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )
 
    for file_path in supported:
        try:
            pages = load_file(file_path)
            docs.extend(pages)
            print(f"  ✓ Loaded: {file_path.name}  ({len(pages)} page(s))")
        except Exception as e:
            print(f"  ✗ Failed: {file_path.name}  → {e}")
 
    return docs
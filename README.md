# Local RAG System · Groq + ChromaDB + sentence-transformers

A fully local Retrieval-Augmented Generation system. Documents are indexed locally, embeddings run on your CPU, and only the final LLM call goes out to the Groq API.

---

## Architecture

```
![Architecture diagram](docs/Architecture.WEBP)
```

---

## Quick start

### 1. Clone & install

```bash
git clone <your-repo>
cd rag-system
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Add your documents

Drop PDF, DOCX, TXT, or MD files into the `data/` folder.

### 4. Index documents

```bash
python ingest.py                  # index ./data/
```

Other options:

```bash
python ingest.py --dir ./my_docs     # custom folder
python ingest.py --file report.pdf   # single file
python ingest.py --clear             # wipe store and re-index
```

This loads every supported file, splits it into overlapping chunks, embeds the chunks locally, and stores the embeddings in ChromaDB. Run it once, then again any time you add new documents.

### 5. Chat

```bash
python main.py
```

Other options:

```bash
python main.py --show-sources    # print retrieved chunks
python main.py --top-k 5         # retrieve 5 chunks per query
python main.py --no-stream       # disable streaming output
python main.py -q "What is X?"   # single query, then exit
```

In-chat commands:

| Command | Action |
|---|---|
| `/sources` | Toggle showing retrieved context chunks |
| `/stats`   | Show number of indexed chunks |
| `/help`    | Show all commands |
| `/exit`    | Quit |

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace model |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | Where ChromaDB saves data |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `3` | Chunks retrieved per query |

---

## Project structure

```
rag-system/
├── data/                  ← put your documents here
├── vectorstore/           ← ChromaDB auto-creates this
├── src/
│   ├── __init__.py
│   ├── loader.py          ← PDF / DOCX / TXT / MD loading
│   ├── chunker.py         ← text splitting with overlap
│   ├── embedder.py        ← sentence-transformers wrapper
│   ├── vectorstore.py     ← ChromaDB read/write
│   ├── llm.py             ← Groq API wrapper (normal + streaming)
│   └── rag_chain.py       ← retrieval + prompt building + generation
├── config.py              ← central config (reads .env)
├── ingest.py               ← indexing CLI entry point
├── main.py                ← chat CLI entry point
├── requirements.txt
├── .env.example
├── .env                   ← YOUR SECRETS (never commit)
└── .gitignore
```

---

## Supported models (Groq)

| Model | Context | Best for |
|---|---|---|
| `llama-3.3-70b-versatile` | 128k tokens | Best quality (default) |
| `llama3-8b-8192` | 8k tokens | Faster, lighter |
| `mixtral-8x7b-32768` | 32k tokens | Long documents |
| `gemma2-9b-it` | 8k tokens | Instruction following |

Change `GROQ_MODEL` in `.env` to switch.

---

## How it works

`ingest.py` runs once (or whenever you add new documents) to load all files in `./data/`, split them into chunks, embed the chunks locally, and store the embeddings in ChromaDB.

`main.py` is the interactive chat loop: it embeds your question, retrieves the most relevant chunks from ChromaDB, builds a context-grounded prompt, and streams the answer back from Groq.

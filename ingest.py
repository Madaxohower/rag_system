import traceback
 
print('=== STEP 1: LOAD ===')
try:
    from src.loader import load_directory
    docs = load_directory('./data')
    print('Pages loaded:', len(docs))
    if docs:
        print('Sample:', docs[0]['text'][:80])
except Exception as e:
    traceback.print_exc()
 
print()
print('=== STEP 2: CHUNK ===')
try:
    from src.chunker import chunk_documents
    chunks = chunk_documents(docs)
    print('Chunks created:', len(chunks))
    if chunks:
        print('First chunk_id:', chunks[0]['chunk_id'])
except Exception as e:
    traceback.print_exc()
 
print()
print('=== STEP 3: EMBED ===')
try:
    from src.embedder import embed_texts
    vecs = embed_texts([chunks[0]['text']])
    print('Vector dim:', len(vecs[0]))
    print('Embedding OK!')
except Exception as e:
    traceback.print_exc()
 
print()
print('=== STEP 4: SAVE TO CHROMADB ===')
try:
    from src.vectorstore import add_chunks, collection_size
    add_chunks(chunks)
    print('Saved! Total in store:', collection_size())
except Exception as e:
    traceback.print_exc()
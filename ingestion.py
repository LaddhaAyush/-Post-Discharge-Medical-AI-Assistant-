from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests

# Parameters
URL = 'https://nephros.gr/images/books/Brenner_and_Rectors_The_Kidney_11th_Edition-0001-0235-s.pdf'  # Use as a web page
CHUNK_SIZE = 500  # characters per chunk
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
NEPHRO_TXT_PATH = 'data/nephro.txt'
FAISS_INDEX_PATH = 'data/nephro_faiss.index'

# 1. Load text from the web (always as HTML)
def load_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = '\n'.join([doc.page_content for doc in docs])
    return text

# 2. Chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# 3. Generate embeddings
def embed_chunks(chunks, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

# 4. Store embeddings in FAISS
def save_faiss_index(embeddings, path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, path)

# 5. Save text chunks
def save_chunks(chunks, path):
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n\n')

if __name__ == '__main__':
    print(f"Loading {URL} ...")
    text = load_text(URL)
    print(f"Text length: {len(text)} characters")
    chunks = chunk_text(text)
    print(f"Number of chunks: {len(chunks)}")
    save_chunks(chunks, NEPHRO_TXT_PATH)
    print(f"Saved chunks to {NEPHRO_TXT_PATH}")
    embeddings = embed_chunks(chunks)
    save_faiss_index(embeddings, FAISS_INDEX_PATH)
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}") 
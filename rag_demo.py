import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq  # Make sure you have the groq package installed
import os
from dotenv import load_dotenv

load_dotenv()
# Paths
FAISS_INDEX_PATH = "data/nephro_faiss.index"
NEPHRO_TXT_PATH = "data/nephro.txt"
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Load text chunks
with open(NEPHRO_TXT_PATH, "r", encoding="utf-8") as f:
    chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")  # Replace with your key and model

def retrieve(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def rag_answer(question):
    context_chunks = retrieve(question, k=5)
    context = "\n".join(context_chunks)
    prompt = (
        "You are a nephrology assistant. Use the following context to answer the question.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer (with citations):"
    )
    answer = llm.invoke(prompt)
    return answer

if __name__ == "__main__":
    while True:
        user_q = input("Ask a nephrology question (or 'exit'): ")
        if user_q.lower() == "exit":
            break
        print("\nAnswer:\n", rag_answer(user_q), "\n")
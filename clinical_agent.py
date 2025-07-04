import logging
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

load_dotenv()

# Enhanced State Schema
class ClinicalState(TypedDict):
    query: str
    expanded_query: str
    context: str
    context_sources: List[Dict]
    patient_report: dict
    response: str
    search_method: str
    chat_history: List[Dict]

# Model Initialization
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
web_tool = DuckDuckGoSearchResults(output_format="list")
arxiv_tool = ArxivQueryRun()
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

# Knowledge Base Setup
faiss_index = faiss.read_index("data/nephro_faiss.index")
chunks = [chunk.strip() for chunk in open("data/nephro.txt", encoding="utf-8").read().split("\n\n") if chunk.strip()]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are Dr. Sarah, a nephrology nurse practitioner with expertise in post-discharge care.

Patient Information:
- Name: {patient_name}
- Diagnosis: {diagnosis}
- Current Medications: {medications}
- Discharge Date: {discharge_date}

Conversation History (most recent to oldest):
{chat_history}

Available Context from {search_method}:
{context}

Current Patient Query: {query}

Instructions:
1. Provide a clear, empathetic response addressing the patient's specific concern.
2. Use the conversation history to ensure continuity and avoid repeating information already provided, especially regarding medications or prior advice.
3. Reference specific details from the history (e.g., previously discussed symptoms or medications) when relevant to the current query.
4. Base your response on the provided context when applicable, prioritizing medical accuracy.
5. ALWAYS include source references in your response using the format [Source: method].
6. If recommending actions, provide specific, practical steps (e.g., dosage instructions, monitoring guidelines).
7. For serious symptoms or uncertainty, recommend consulting a physician immediately.
8. Keep responses conversational, avoiding medical jargon unless necessary, and explain terms if used.
9. Ensure responses are concise and directly address the query without redundancy.

Response Format:
- Address the patient's concern directly, referencing relevant prior conversation details.
- Provide medical information with source citations.
- Include practical next steps when appropriate.
- End with: Sources: [list sources]
""")

# Query Expansion
def expand_query(query: str) -> str:
    synonyms = {"kidney": "renal", "failure": "insufficiency", "pain": "ache"}
    words = query.split()
    expanded = [f"{word} {synonyms[word]}" if word in synonyms else word for word in words]
    return " ".join(expanded)

# Hybrid Search + Reranking
def hybrid_search(query: str) -> (List[str], List[Dict]):
    vec = embedder.encode([query])
    D, I = faiss_index.search(np.array(vec).astype("float32"), 5)
    rag_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    bm25_results = bm25.get_top_n(query.lower().split(), chunks, n=5)
    combined = list(set(rag_chunks + bm25_results))
    scores = reranker.predict([(query, c) for c in combined])
    reranked = [c for _, c in sorted(zip(scores, combined), reverse=True)][:3]
    sources = [{"type": "knowledge_base", "content_preview": c[:100]} for c in reranked]
    return reranked, sources

# Context Lookup
def run_context_lookup(state: ClinicalState) -> ClinicalState:
    query = state["query"]
    state["expanded_query"] = expand_query(query)
    chunks, sources = hybrid_search(state["expanded_query"])
    if chunks and len(" ".join(chunks)) > 100:
        state.update(context="\n\n".join(chunks), context_sources=sources, search_method="Hybrid RAG")
    else:
        try:
            web_results = web_tool.invoke(state["expanded_query"])
            web_context = [f"{r['title']}: {r['snippet']} (Source: {r['link']})" for r in web_results[:3]]
            state.update(context="\n\n".join(web_context), context_sources=web_results, search_method="Web Search")
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            state.update(context="No relevant information found.", context_sources=[], search_method="None")
    return state

# Answer Generation
def run_answer(state: ClinicalState) -> ClinicalState:
    rpt = state["patient_report"]
    # Format chat history for prompt (most recent to oldest)
    chat_history_str = "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in state["chat_history"][::-1][-5:]])
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({
        "patient_name": rpt.get("patient_name", "Patient"),
        "diagnosis": rpt.get("primary_diagnosis", ""),
        "medications": ", ".join(rpt.get("medications", [])),
        "discharge_date": rpt.get("discharge_date", ""),
        "context": state["context"],
        "query": state["query"],
        "search_method": state["search_method"],
        "chat_history": chat_history_str
    }).strip()
    citations = "\nSources: " + ", ".join([s.get("type", "Unknown") for s in state["context_sources"]])
    state["response"] = final_answer + citations
    return state

# Multi-step Reasoning
def build_graph():
    g = StateGraph(ClinicalState)
    g.add_node("ContextLookup", run_context_lookup)
    g.add_node("Answer", run_answer)
    g.add_edge(START, "ContextLookup")
    g.add_edge("ContextLookup", "Answer")
    g.add_edge("Answer", END)
    return g.compile()

# Clinical Agent Class
class ClinicalAgent:
    def __init__(self):
        self.graph = build_graph()
        self.conversation_history = []
        self.patient_report = {}

    def set_patient_report(self, report: dict):
        self.patient_report = report

    def interact(self, query: str) -> str:
        state = ClinicalState(
            query=query,
            expanded_query="",
            context="",
            context_sources=[],
            patient_report=self.patient_report,
            response="",
            search_method="",
            chat_history=self.conversation_history
        )
        final_state = self.graph.invoke(state)
        self.conversation_history.append({"query": query, "response": final_state["response"]})
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        return final_state["response"]
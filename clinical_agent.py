# import logging
# import os
# import faiss
# import numpy as np
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from duckduckgo_search import DDGS

# load_dotenv()

# class ClinicalAgent:
#     def __init__(self):
#         self.k = 3
#         self.patient_report = None
#         self.index = faiss.read_index("data/nephro_faiss.index")
#         self.chunks = [chunk.strip() for chunk in open("data/nephro.txt", encoding="utf-8").read().split("\n\n") if chunk.strip()]
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
#         self.prompt = ChatPromptTemplate.from_template(
#             """
# You are Dr. Sarah, a nephrology nurse practitioner. Use the provided patient info and context to respond accurately.

# Patient: {patient_name} | Diagnosis: {diagnosis} | Medications: {medications} | Discharge Date: {discharge_date}

# Context:
# {context}

# Patient says:
# {query}

# Reply clearly and empathetically. If using online sources, cite them. For unknowns, recommend professional consultation.
#             """
#         )

#     def set_patient_report(self, report):
#         self.patient_report = report

#     def _rag_context(self, query):
#         vec = self.embedder.encode([query])
#         D, I = self.index.search(np.array(vec).astype("float32"), self.k)
#         return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

#     def _web_search(self, query):
#         try:
#             with DDGS() as ddgs:
#                 results = list(ddgs.text(f"nephrology {query}", max_results=2))
#                 return [f"{r['title']}: {r['body'][:200]}... (Source: {r['href']})" for r in results if 'title' in r]
#         except Exception as e:
#             logging.error(f"Web search failed: {e}")
#             return ["Web search is currently unavailable."]

#     def _get_context(self, query):
#         rag_chunks = self._rag_context(query)
#         joined_context = "\n\n".join(rag_chunks)
#         if not joined_context or len(joined_context) < 100:
#             logging.info("RAG insufficient, switching to web search")
#             return "\n\n".join(self._web_search(query))
#         return joined_context

#     def interact(self, query):
#         context = self._get_context(query)
#         inputs = {
#             "patient_name": self.patient_report.get("patient_name", "Patient"),
#             "diagnosis": self.patient_report.get("primary_diagnosis", ""),
#             "medications": ", ".join(self.patient_report.get("medications", [])),
#             "discharge_date": self.patient_report.get("discharge_date", ""),
#             "context": context,
#             "query": query
#         }
#         chain = self.prompt | self.llm | StrOutputParser()
#         return chain.invoke(inputs).strip()

# import logging
# import os
# import faiss
# import numpy as np
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.tools import DuckDuckGoSearchResults

# load_dotenv()

# class ClinicalAgent:
#     def __init__(self):
#         self.k = 3
#         self.patient_report = None
#         self.index = faiss.read_index("data/nephro_faiss.index")
#         self.chunks = [chunk.strip() for chunk in open("data/nephro.txt", encoding="utf-8").read().split("\n\n") if chunk.strip()]
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
#         self.web_search = DuckDuckGoSearchResults(output_format="list")

#         self.prompt = ChatPromptTemplate.from_template(
#             """
# You are Dr. Sarah, a nephrology nurse practitioner. Use the provided patient info and context to respond accurately.

# Patient: {patient_name} | Diagnosis: {diagnosis} | Medications: {medications} | Discharge Date: {discharge_date}

# Context:
# {context}

# Patient says:
# {query}

# Reply clearly and empathetically. If using online sources, cite them. For unknowns, recommend professional consultation.
# And most important understand user query and reply according to it. Dont follow a single template everytime and provide reference for your every response.
#             """
#         )

#     def set_patient_report(self, report):
#         self.patient_report = report

#     def _rag_context(self, query):
#         vec = self.embedder.encode([query])
#         D, I = self.index.search(np.array(vec).astype("float32"), self.k)
#         return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

#     def _web_context(self, query):
#         try:
#             results = self.web_search.invoke(query)
#             return [f"{r['title']}: {r['snippet']} (Source: {r['link']})" for r in results if 'title' in r and 'snippet' in r and 'link' in r]
#         except Exception as e:
#             logging.error(f"Web search failed: {e}")
#             return ["Web search is currently unavailable."]

#     def _get_context(self, query):
#         rag_chunks = self._rag_context(query)
#         joined_context = "\n\n".join(rag_chunks)
#         if not joined_context or len(joined_context) < 100:
#             logging.info("RAG insufficient, switching to web search")
#             return "\n\n".join(self._web_context(query))
#         return joined_context

#     def interact(self, query):
#         context = self._get_context(query)
#         inputs = {
#             "patient_name": self.patient_report.get("patient_name", "Patient"),
#             "diagnosis": self.patient_report.get("primary_diagnosis", ""),
#             "medications": ", ".join(self.patient_report.get("medications", [])),
#             "discharge_date": self.patient_report.get("discharge_date", ""),
#             "context": context,
#             "query": query
#         }
#         chain = self.prompt | self.llm | StrOutputParser()
#         return chain.invoke(inputs).strip()

import logging, os, faiss, numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
import re

load_dotenv()

# 1️⃣ Define structured state with TypedDict
class ClinicalState(TypedDict):
    query: str
    context: str
    context_sources: List[Dict]  # Track sources for referencing
    patient_report: dict
    response: str
    search_method: str  # Track which method was used

# 2️⃣ Initialize tools & models
web_tool = DuckDuckGoSearchResults(output_format="list")
arxiv_tool = ArxivQueryRun()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

# 3️⃣ Enhanced prompt for LLM responses with source referencing
prompt = ChatPromptTemplate.from_template("""
You are Dr. Sarah, a nephrology nurse practitioner with expertise in post-discharge care.

Patient Information:
- Name: {patient_name}
- Diagnosis: {diagnosis}
- Current Medications: {medications}
- Discharge Date: {discharge_date}

Available Context from {search_method}:
{context}

Patient Query: {query}

Instructions:
1. Provide a clear, empathetic response addressing the patient's specific concern
2. Base your response on the provided context when relevant
3. ALWAYS include source references in your response using the format [Source: method]
4. If recommending actions, be specific and practical
5. When uncertain or for serious symptoms, recommend consulting their physician
6. Keep responses conversational and avoid medical jargon when possible
7. Don't repeat the same information if it's already been established in the conversation
8. Try to find relevant context in vectorstore first, then go for web search and don't repeat you are a nephrology nurse practitioner as user know that , say user once at starting of conversation only .

Response Format:
- Address the patient's concern directly
- Provide relevant medical information with source citations
- Include practical next steps when appropriate
- Always cite your sources at the end
""")

# 4️⃣ RAG retrieval with enhanced tracking
faiss_index = faiss.read_index("data/nephro_faiss.index")
chunks = [chunk.strip() for chunk in open("data/nephro.txt", encoding="utf-8").read().split("\n\n") if chunk.strip()]

def rag_chunks(q: str) -> tuple[List[str], List[Dict]]:
    """Return both chunks and source information"""
    vec = embedder.encode([q])
    distances, indices = faiss_index.search(np.array(vec).astype("float32"), 3)
    
    retrieved_chunks = []
    source_info = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            retrieved_chunks.append(chunk)
            source_info.append({
                "type": "knowledge_base",
                "index": idx,
                "relevance_score": float(distances[0][i]),
                "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
    
    return retrieved_chunks, source_info

def run_context_lookup(state: ClinicalState) -> ClinicalState:
    """Enhanced context lookup with source tracking"""
    query = state["query"]
    logging.info(f"Looking up context for query: {query}")
    
    # Try RAG first
    retrieved_chunks, rag_sources = rag_chunks(query)
    
    if retrieved_chunks and len(" ".join(retrieved_chunks)) >= 100:
        # RAG has sufficient content
        state["context"] = "\n\n".join(retrieved_chunks)
        state["context_sources"] = rag_sources
        state["search_method"] = "Medical Knowledge Base"
        logging.info("Using RAG context from knowledge base")
        
    else:
        # Fall back to web search and arxiv
        logging.info("RAG insufficient, using web search and arxiv")
        
        try:
            # Web search
            web_results = web_tool.invoke(f"nephrology {query}")
            web_sources = []
            web_context = []
            
            for r in web_results:
                if all(k in r for k in ("title", "snippet", "link")):
                    web_context.append(f"Title: {r['title']}\nSummary: {r['snippet']}")
                    web_sources.append({
                        "type": "web",
                        "title": r['title'],
                        "link": r['link'],
                        "snippet": r['snippet'][:200] + "..." if len(r['snippet']) > 200 else r['snippet']
                    })
            
            # ArXiv search
            try:
                arxiv_results = arxiv_tool.invoke(query)
                arxiv_sources = []
                arxiv_context = []
                
                if isinstance(arxiv_results, str):
                    # Parse arxiv results if they're in string format
                    arxiv_context.append(f"Research Reference: {arxiv_results[:500]}...")
                    arxiv_sources.append({
                        "type": "arxiv",
                        "content": arxiv_results[:200] + "..." if len(arxiv_results) > 200 else arxiv_results
                    })
                
            except Exception as e:
                logging.warning(f"ArXiv search failed: {e}")
                arxiv_context = []
                arxiv_sources = []
            
            # Combine all sources
            all_context = web_context + arxiv_context
            all_sources = web_sources + arxiv_sources
            
            state["context"] = "\n\n".join(all_context) if all_context else "Limited information available."
            state["context_sources"] = all_sources
            state["search_method"] = "Web Search and Research Papers"
            
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            state["context"] = "I'm having trouble accessing external sources right now."
            state["context_sources"] = []
            state["search_method"] = "Limited Resources"
    
    return state

def run_answer(state: ClinicalState) -> ClinicalState:
    """Generate response with proper source citations"""
    rpt = state["patient_report"]
    
    # Prepare context with source information
    context_with_sources = state["context"]
    
    # Add source summary for the LLM
    if state["context_sources"]:
        source_summary = "\n\nSource Types Used: "
        knowledge_base_count = len([s for s in state["context_sources"] if s["type"] == "knowledge_base"])
        web_count = len([s for s in state["context_sources"] if s["type"] == "web"])
        arxiv_count = len([s for s in state["context_sources"] if s["type"] == "arxiv"])
        
        if knowledge_base_count > 0:
            source_summary += f"{knowledge_base_count} medical knowledge base entries, "
        if web_count > 0:
            source_summary += f"{web_count} web sources, "
        if arxiv_count > 0:
            source_summary += f"{arxiv_count} research papers, "
        
        source_summary = source_summary.rstrip(", ")
        context_with_sources += source_summary
    
    chain = prompt | llm | StrOutputParser()
    
    raw_response = chain.invoke({
        "patient_name": rpt.get("patient_name", "Patient"),
        "diagnosis": rpt.get("primary_diagnosis", ""),
        "medications": ", ".join(rpt.get("medications", [])),
        "discharge_date": rpt.get("discharge_date", ""),
        "context": context_with_sources,
        "query": state["query"],
        "search_method": state["search_method"]
    }).strip()
    
    # Add source citations to the response
    response_with_citations = raw_response
    
    if state["context_sources"]:
        citation_text = "\n\n📚 **Sources:** "
        
        # Group sources by type
        kb_sources = [s for s in state["context_sources"] if s["type"] == "knowledge_base"]
        web_sources = [s for s in state["context_sources"] if s["type"] == "web"]
        arxiv_sources = [s for s in state["context_sources"] if s["type"] == "arxiv"]
        
        citations = []
        
        if kb_sources:
            citations.append(f"Medical Knowledge Base ({len(kb_sources)} entries)")
        
        if web_sources:
            # Show specific web sources
            for i, source in enumerate(web_sources[:2]):  # Limit to first 2 web sources
                citations.append(f"Web: {source['title']} ({source['link']})")
        
        if arxiv_sources:
            citations.append(f"Research Papers ({len(arxiv_sources)} papers)")
        
        if citations:
            citation_text += "; ".join(citations)
            response_with_citations += citation_text
    
    else:
        response_with_citations += "\n\n⚠️ **Note:** Response based on general medical knowledge. For personalized advice, please consult your healthcare provider."
    
    state["response"] = response_with_citations
    return state

def build_graph():
    """Build the clinical agent graph"""
    g = StateGraph(ClinicalState)
    g.add_node("ContextLookup", run_context_lookup)
    g.add_node("Answer", run_answer)
    g.add_edge(START, "ContextLookup")
    g.add_edge("ContextLookup", "Answer")
    g.add_edge("Answer", END)
    return g.compile()

class ClinicalAgent:
    def __init__(self):
        self.graph = build_graph()
        self.conversation_history = []  # Track conversation for context
        
    def set_patient_report(self, report: dict):
        self.patient_report = report

    def interact(self, query: str) -> str:
        """Main interaction method with conversation tracking"""
        # Create state for this interaction
        state = ClinicalState(
            query=query,
            context="",
            context_sources=[],
            patient_report=self.patient_report,
            response="",
            search_method=""
        )
        
        # Add conversation history context if available
        if self.conversation_history:
            # Add recent conversation context to help avoid repetition
            recent_context = "\n".join([
                f"Previous: {entry['query']} -> {entry['response'][:100]}..."
                for entry in self.conversation_history[-2:]  # Last 2 exchanges
            ])
            state["query"] = f"Recent conversation:\n{recent_context}\n\nCurrent query: {query}"
        
        # Run the graph
        final_state = self.graph.invoke(state)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": query,
            "response": final_state["response"],
            "sources": final_state["context_sources"]
        })
        
        # Keep only last 5 exchanges to prevent memory bloat
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        return final_state["response"]
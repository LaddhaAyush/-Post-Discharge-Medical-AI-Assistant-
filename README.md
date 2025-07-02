# Post Discharge Medical AI Assistant POC

## Overview
This project is a Proof of Concept (POC) multi-agent AI system for post-discharge patient care. It demonstrates core GenAI skills including Retrieval-Augmented Generation (RAG), multi-agent orchestration, and medical data processing in a simplified but functional system.

**Key Features:**
- Manages 25+ dummy post-discharge patient reports
- Uses RAG with nephrology reference materials
- Implements two specialized AI agents (Receptionist & Clinical)
- Provides a simple web or CLI interface for patient interactions

---

## Core Requirements & Features

### 1. Data Setup
- 25+ dummy post-discharge reports (JSON format)
- Nephrology reference book (text format)
- Simple database storage (JSON files)
- Vector embeddings for semantic search (FAISS)

### 2. Multi-Agent System
- **Receptionist Agent:**
  - Asks patient for their name
  - Uses a database retrieval tool to fetch the patient's discharge report
  - Asks follow-up questions based on the discharge information
  - Routes medical queries to the Clinical Agent
- **Clinical AI Agent:**
  - Handles medical questions and clinical advice
  - Uses RAG over nephrology reference book for answers
  - Uses web search tool for queries outside reference materials
  - Provides citations from reference materials or web
  - Logs patient interactions

### 3. RAG Implementation
- Processes and chunks nephrology reference materials
- Creates vector embeddings for semantic search (using Sentence-Transformers)
- Implements retrieval and answer generation
- Includes source citations in responses

### 4. Web Search Tool
- Integrates DuckDuckGo web search for queries outside reference materials
- Clearly indicates when information comes from web search vs. reference materials
- Handles fallback when specialized information is needed

### 5. Logging System
- Comprehensive logging throughout the system
- Logs all interactions between agents and users
- Logs agent handoffs and decision processes
- Maintains log files with timestamps showing complete system flow
- Includes information retrieval attempts and results

### 6. Patient Data Retrieval Tool
- Dedicated tool for database interaction
- Patient lookup by name (with fuzzy matching)
- Returns structured discharge report data
- Handles error cases (patient not found, multiple patients with same name)
- Logs all database access attempts

---

## Technical Specifications
- **Frontend:** Streamlit (recommended), or CLI for POC
- **Backend:** Python (LangChain, custom multi-agent logic)
- **Multi-Agent Framework:** Custom implementation (LangChain-based, can be extended to LangGraph)
- **Vector DB:** FAISS
- **Data Storage:** JSON files
- **Embeddings:** Sentence-Transformers (HuggingFace)
- **Web Search:** DuckDuckGo

---

## Sample Patient Report Structure
```json
{
  "patient_name": "John Smith",
  "discharge_date": "2024-01-15",
  "primary_diagnosis": "Chronic Kidney Disease Stage 3",
  "medications": ["Lisinopril 10mg daily", "Furosemide 20mg twice daily"],
  "dietary_restrictions": "Low sodium (2g/day), fluid restriction (1.5L/day)",
  "follow_up": "Nephrology clinic in 2 weeks",
  "warning_signs": "Swelling, shortness of breath, decreased urine output",
  "discharge_instructions": "Monitor blood pressure daily, weigh yourself daily"
}
```

---

## Expected System Workflow

**Initial Interaction:**
- System: "Hello! I'm your post-discharge care assistant. What's your name?"
- Patient: "John Smith"
- Receptionist Agent: [Fetches discharge report]
- Receptionist Agent: "Hi John! I found your discharge report from January 15th for Chronic Kidney Disease. How are you feeling today? Are you following your medication schedule?"

**Medical Query Routing:**
- Patient: "I'm having swelling in my legs. Should I be worried?"
- Receptionist Agent: "This sounds like a medical concern. Let me connect you with our Clinical AI Agent."
- Clinical Agent: "Based on your CKD diagnosis and nephrology guidelines, leg swelling can indicate fluid retention... [RAG response with citations]"

**Web Search Fallback Example:**
- Patient: "What's the latest research on SGLT2 inhibitors for kidney disease?"
- Clinical Agent: "This requires recent information. Let me search for you... According to recent medical literature [Web search results with source]..."

---

## Architecture Justification
- **LLM Selection:** Uses Groq LLM (Llama3) for fast, high-quality medical language understanding and generation.
- **Vector Database:** FAISS for efficient, scalable semantic search over reference materials.
- **RAG Implementation:** Combines vector search with LLM to provide accurate, context-aware answers with citations.
- **Multi-Agent Framework:** Modular, extensible agent design (can be extended to LangGraph for more complex workflows).
- **Web Search Integration:** DuckDuckGo search for up-to-date medical information outside the reference book.
- **Patient Data Retrieval:** Robust, fuzzy-matching tool for patient lookup and report retrieval.
- **Logging Implementation:** Comprehensive logging for all agent actions, handoffs, and retrievals.

---

## How to Run the Project
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   - Ensure `data/patients.json` contains 25+ dummy patient reports.
   - Ensure `data/nephro.txt` contains nephrology reference text.
3. **Generate vector embeddings:**
   ```bash
   python ingestion.py
   ```
4. **Run the assistant (CLI):**
   ```bash
   python receptionist_demo.py
   ```
   *(For Streamlit UI, run the appropriate Streamlit app if provided)*

---

## Disclaimers
- **This is an AI assistant for educational purposes only.**
- **Always consult healthcare professionals for medical advice.**
- **No real patient data is used.**

---

## Final Checklist
- [x] 25+ dummy patient reports created
- [x] Nephrology reference materials processed
- [x] Receptionist Agent implemented
- [x] Clinical AI Agent with RAG implemented
- [x] Patient data retrieval tool implemented
- [x] Web search tool integration
- [x] Comprehensive logging system
- [x] Simple web or CLI interface working
- [x] Agent handoff mechanism functional
- [x] GitHub repo with clean code
- [x] Brief report with architecture justification
- [x] Demo video recorded
- [x] All code commented and documented

---

## Contact
For questions or collaboration, please open an issue or contact the project maintainer.

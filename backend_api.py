from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import json
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Import your improved agents
from receptionist_agent import ReceptionistAgent
from clinical_agent import ClinicalAgent

# --- Logging Setup ---
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        f'logs/backend_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_logging()

app = FastAPI(
    title="Post-Discharge Medical AI Assistant",
    description="An AI-powered assistant for post-discharge patient care",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load patient data
try:
    with open(os.path.join("data", "patients.json"), encoding="utf-8") as f:
        PATIENTS = json.load(f)
    logging.info(f"Loaded {len(PATIENTS)} patient records")
except Exception as e:
    logging.error(f"Failed to load patient data: {e}")
    PATIENTS = []

# Initialize agents - using session-based storage for better conversation handling
agents_storage = {}

class ChatRequest(BaseModel):
    user_input: Any
    session_id: str = "user-session"
    patient_report: Optional[Dict[Any, Any]] = None

    @validator('user_input', pre=True, always=True)
    def ensure_string(cls, v):
        if not isinstance(v, str):
            return str(v)
        return v

class ChatResponse(BaseModel):
    response: str
    status: str
    patient_report: Optional[Dict[Any, Any]] = None
    agent_info: Optional[Dict[str, Any]] = None

def get_or_create_receptionist_agent(session_id: str) -> ReceptionistAgent:
    if session_id not in agents_storage:
        agents_storage[session_id] = {
            "receptionist": ReceptionistAgent(),
            "clinical": ClinicalAgent()
        }
    return agents_storage[session_id]["receptionist"]

def get_or_create_clinical_agent(session_id: str) -> ClinicalAgent:
    if session_id not in agents_storage:
        agents_storage[session_id] = {
            "receptionist": ReceptionistAgent(),
            "clinical": ClinicalAgent()
        }
    return agents_storage[session_id]["clinical"]

@app.get("/")
def root():
    return {
        "message": "Welcome to the Post-Discharge Medical AI Assistant API v2.0",
        "features": [
            "Context-aware conversations",
            "Session-based agent management",
            "Source referencing in clinical responses",
            "Natural conversation flow"
        ],
        "endpoints": {
            "receptionist": "/chat/receptionist",
            "clinical": "/chat/clinical",
            "patient_lookup": "/patients/{name}",
            "health_check": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(agents_storage),
        "patients_loaded": len(PATIENTS)
    }

@app.get("/patients/{name}")
def get_patient(name: str):
    try:
        matches = [p for p in PATIENTS if p["patient_name"].lower() == name.lower()]
        if not matches:
            raise HTTPException(status_code=404, detail="Patient not found")
        if len(matches) > 1:
            return {
                "status": "multiple_found",
                "message": f"Found {len(matches)} patients with that name",
                "patients": [{"name": p["patient_name"], "id": p.get("patient_id")} for p in matches]
            }
        return {"status": "found", "patient": matches[0]}
    except Exception as e:
        logging.error(f"Error retrieving patient {name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/receptionist", response_model=ChatResponse)
def chat_receptionist(req: ChatRequest):
    try:
        receptionist_agent = get_or_create_receptionist_agent(req.session_id)
        response, status = receptionist_agent.interact(req.user_input, session_id=req.session_id)

        chat_response = ChatResponse(
            response=response,
            status=str(status),
            patient_report=receptionist_agent.patient_report,
            agent_info={
                "agent_type": "receptionist",
                "agent_name": "Maria",
                "conversation_stage": getattr(receptionist_agent, 'conversation_stage', 'unknown'),
                "patient_identified": bool(receptionist_agent.patient_report)
            }
        )

        return chat_response

    except Exception as e:
        logging.error(f"Error in receptionist chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error in receptionist chat")

@app.post("/chat/clinical", response_model=ChatResponse)
def chat_clinical(req: ChatRequest):
    try:
        if not req.patient_report:
            raise HTTPException(status_code=400, detail="Patient report is required for clinical consultation")

        clinical_agent = get_or_create_clinical_agent(req.session_id)
        clinical_agent.set_patient_report(req.patient_report)
        response = clinical_agent.interact(req.user_input)

        chat_response = ChatResponse(
            response=response,
            status="success",
            patient_report=req.patient_report,
            agent_info={
                "agent_type": "clinical",
                "agent_name": "Dr. Sarah",
                "conversation_history_length": len(getattr(clinical_agent, 'conversation_history', [])),
                "patient_name": req.patient_report.get("patient_name", "Unknown")
            }
        )

        return chat_response

    except Exception as e:
        logging.error(f"Error in clinical chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error in clinical chat")

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in agents_storage:
        del agents_storage[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
def list_sessions():
    return {"active_sessions": list(agents_storage.keys()), "total_sessions": len(agents_storage)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import streamlit as st
import requests

API_URL = "http://localhost:8000"

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = "receptionist"
if "patient_report" not in st.session_state:
    st.session_state.patient_report = None
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# API Functions
def receptionist_chat(user_input):
    try:
        resp = requests.post(f"{API_URL}/chat/receptionist", json={"user_input": user_input, "session_id": "web-session"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("patient_report"):
            st.session_state.patient_report = data["patient_report"]
            st.session_state.patient_name = st.session_state.patient_report.get("patient_name", "")
        return data["response"], data["status"]
    except Exception:
        return "I'm experiencing a connection issue. Please try again shortly.", False

def clinical_chat(user_input):
    try:
        resp = requests.post(f"{API_URL}/chat/clinical", json={"user_input": user_input, "patient_report": st.session_state.patient_report}, timeout=30)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception:
        return "Clinical system error. Please try again."

# Page Configuration
st.set_page_config(page_title="Post-Discharge Medical AI Assistant", page_icon="🏥", layout="wide")

# Initial Greeting
if not st.session_state.conversation_started:
    st.session_state.conversation_started = True
    st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": "Hello! I'm Maria, your post-discharge care coordinator. What's your name?"})

# Header
st.title("🏥 Post-Discharge Medical AI Assistant")
st.markdown("This AI assistant is for educational use only. Please consult a healthcare professional for medical advice.")
st.divider()

# Sidebar for Patient Information
with st.sidebar:
    st.header("📋 Patient Information")
    if st.session_state.patient_report:
        rpt = st.session_state.patient_report
        st.write(f"**Name:** {rpt.get('patient_name', 'N/A')}")
        st.write(f"**Diagnosis:** {rpt.get('primary_diagnosis', 'N/A')}")
        st.write(f"**Discharge Date:** {rpt.get('discharge_date', 'N/A')}")
        if meds := rpt.get('medications'):
            st.write("**Medications:**")
            for med in meds:
                st.write(f"• {med}")
        if diet := rpt.get('dietary_restrictions'):
            st.write("**Dietary Guidelines:**")
            st.write(diet)
        if follow := rpt.get('follow_up'):
            st.write("**Follow-up:**")
            st.write(follow)
        if warnings := rpt.get('warning_signs'):
            st.warning(f"⚠️ {warnings}")
        if instructions := rpt.get('discharge_instructions'):
            st.info(instructions)
    else:
        st.write("Patient info will be displayed here after registration.")
    
    st.divider()
    
    if st.session_state.agent == "clinical":
        if st.button("🔙 Return to Receptionist"):
            st.session_state.agent = "receptionist"
            st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": "🔄 Returning to Maria for administrative assistance."})
            st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": "Hi again! I'm Maria. How can I help you?"})
            st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.clear()
        st.rerun()

# Chat Interface
st.markdown(f"**Current Agent:** {'Maria (Receptionist)' if st.session_state.agent == 'receptionist' else 'Dr. Sarah (Clinical Specialist)'}")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(message["message"])
    else:
        avatar = "👩‍💼" if message["agent"] == "maria" else "👩‍⚕️"
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(message["message"])

prompt = st.chat_input("Type your message here...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "message": prompt})
    with st.spinner("Assistant is typing..."):
        if st.session_state.agent == "receptionist":
            response, status = receptionist_chat(prompt)
            st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": response})
            if status == "route_clinical":
                st.session_state.agent = "clinical"
                st.session_state.chat_history.append({"role": "assistant", "agent": "sarah", "message": "🔄 Connecting you with Dr. Sarah, our Clinical Specialist..."})
                greeting = f"Hi {st.session_state.patient_name or 'there'}, I'm Dr. Sarah. What can I help you with today?"
                st.session_state.chat_history.append({"role": "assistant", "agent": "sarah", "message": greeting})
        else:
            response = clinical_chat(prompt)
            st.session_state.chat_history.append({"role": "assistant", "agent": "sarah", "message": response})
            if any(keyword in prompt.lower() for keyword in ["receptionist", "maria", "admin", "appointment", "schedule"]):
                st.session_state.agent = "receptionist"
                st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": "🔄 Switching back to Maria for administrative help."})
                st.session_state.chat_history.append({"role": "assistant", "agent": "maria", "message": "Hi again! I'm Maria. How can I assist you?"})
    st.rerun()

# Emergency Notice
st.error("🚨 **Emergency Notice:** If you're experiencing a medical emergency, please call emergency services immediately.")
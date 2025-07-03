import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"  # Change if backend runs elsewhere

st.set_page_config(page_title="Post-Discharge Medical AI Assistant", page_icon="🏥")

# --- Session State ---
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

# --- UI: Header & Disclaimer ---
st.title("🏥 Post-Discharge Medical AI Assistant")
st.markdown("""
**This is an AI assistant for educational purposes only.**  
**Always consult healthcare professionals for medical advice.**
---
""")

# --- Chat UI ---
def display_chat():
    """Display chat history with better formatting"""
    for entry in st.session_state.chat_history:
        role, msg = entry["role"], entry["message"]
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg)
        else:
            # Different avatars for different agents
            avatar = "👩‍⚕️" if "Dr. Sarah" in msg or st.session_state.agent == "clinical" else "👋"
            with st.chat_message("assistant", avatar=avatar):
                st.write(msg)

def add_to_history(role, message):
    """Add message to chat history with timestamp"""
    st.session_state.chat_history.append({
        "role": role, 
        "message": message,
        "timestamp": time.time()
    })

def show_typing_indicator():
    """Show typing indicator for better UX"""
    with st.chat_message("assistant", avatar="💭"):
        st.write("Typing...")

# --- API Communication ---
def receptionist_chat(user_input):
    """Handle receptionist chat with error handling"""
    try:
        if not isinstance(user_input, str):
            user_input = str(user_input)
        resp = requests.post(f"{API_URL}/chat/receptionist", json={
            "user_input": user_input,
            "session_id": "web-session"
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Save patient report if found
        if data.get("patient_report"):
            st.session_state.patient_report = data["patient_report"]
            st.session_state.patient_name = st.session_state.patient_report.get("patient_name", "")
            
        return data["response"], data["status"]
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return "I'm having trouble connecting to our system. Please try again in a moment.", False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return "Something went wrong. Please try again.", False

def clinical_chat(user_input):
    """Handle clinical chat with error handling"""
    try:
        if not isinstance(user_input, str):
            user_input = str(user_input)
        resp = requests.post(f"{API_URL}/chat/clinical", json={
            "user_input": user_input,
            "patient_report": st.session_state.patient_report
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return "I'm having trouble connecting to our clinical system. Please try again in a moment."
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return "Something went wrong with the clinical consultation. Please try again."

# --- Initial Setup ---
if not st.session_state.conversation_started:
    st.session_state.conversation_started = True
    add_to_history("assistant", "Hello! I'm Maria, your post-discharge care coordinator. What's your name?")

# --- Display Chat ---
display_chat()

# --- Agent Status Display ---
col1, col2 = st.columns([3, 1])
with col2:
    if st.session_state.agent == "receptionist":
        st.info("🏥 **Receptionist** - Maria")
    else:
        st.info("👩‍⚕️ **Clinical Specialist** - Dr. Sarah")

# --- User Input ---
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    add_to_history("user", prompt)
    
    # Show typing indicator
    typing_placeholder = st.empty()
    with typing_placeholder:
        show_typing_indicator()
    
    try:
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Process based on current agent
        if st.session_state.agent == "receptionist":
            response, status = receptionist_chat(prompt)
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Add response to history
            add_to_history("assistant", response)
            
            # Handle agent transitions
            if status == "route_clinical":
                st.session_state.agent = "clinical"
                # Add transition message
                add_to_history("assistant", "🔄 **Connecting you with Dr. Sarah, our Clinical Specialist...**")
                time.sleep(1)  # Brief pause for transition
                
                # Clinical agent greeting
                pname = st.session_state.patient_name or "there"
                clinical_greeting = f"Hi {pname}! I'm Dr. Sarah, a nephrology nurse practitioner. I understand you have some concerns. What's troubling you today?"
                add_to_history("assistant", clinical_greeting)
                
        else:  # clinical agent
            response = clinical_chat(prompt)
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Add response to history
            add_to_history("assistant", response)
            
            # Allow return to receptionist for administrative tasks
            if any(word in prompt.lower() for word in ["receptionist", "maria", "admin", "appointment", "schedule"]):
                st.session_state.agent = "receptionist"
                add_to_history("assistant", "🔄 **Connecting you back to Maria for administrative assistance...**")
                time.sleep(1)
                add_to_history("assistant", "Hi again! I'm Maria. How can I help you with your administrative needs?")
    
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"An error occurred: {e}")
        add_to_history("assistant", "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.")
    
    # Auto-scroll to bottom
    st.rerun()

# --- Sidebar: Patient Information ---
with st.sidebar:
    st.header("Patient Information")
    
    if st.session_state.patient_report:
        # Display patient info in a more organized way
        report = st.session_state.patient_report
        
        # Basic Info
        st.subheader("📋 Basic Information")
        st.write(f"**Name:** {report.get('patient_name', 'N/A')}")
        st.write(f"**Diagnosis:** {report.get('primary_diagnosis', 'N/A')}")
        st.write(f"**Discharge Date:** {report.get('discharge_date', 'N/A')}")
        
        # Medications
        if report.get('medications'):
            st.subheader("💊 Current Medications")
            for med in report['medications']:
                st.write(f"• {med}")
        
        # Dietary Restrictions
        if report.get('dietary_restrictions'):
            st.subheader("🍽️ Dietary Guidelines")
            st.write(report['dietary_restrictions'])
        
        # Follow-up
        if report.get('follow_up'):
            st.subheader("📅 Follow-up")
            st.write(report['follow_up'])
        
        # Warning Signs
        if report.get('warning_signs'):
            st.subheader("⚠️ Warning Signs")
            st.warning(report['warning_signs'])
        
        # Discharge Instructions
        if report.get('discharge_instructions'):
            st.subheader("📝 Instructions")
            st.info(report['discharge_instructions'])
            
    else:
        st.write("Patient information will appear here once you provide your name.")
    
    # Agent switching options
    st.divider()
    st.subheader("🔄 Agent Options")
    
    if st.session_state.agent == "clinical" and st.session_state.patient_report:
        if st.button("Return to Receptionist"):
            st.session_state.agent = "receptionist"
            add_to_history("assistant", "🔄 **Returning to Maria for administrative assistance...**")
            add_to_history("assistant", "Hi again! I'm Maria. How can I help you?")
            st.rerun()
    
    # Clear chat option
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.agent = "receptionist"
        st.session_state.patient_report = None
        st.session_state.patient_name = ""
        st.session_state.conversation_started = False
        st.rerun()

# --- Footer ---
st.markdown("""
---
**Emergency Notice:** If you're experiencing a medical emergency, please call emergency services immediately.
""")

# --- Debug Info (only in development) ---
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("**Current Agent:**", st.session_state.agent)
    st.write("**Patient Name:**", st.session_state.patient_name)
    st.write("**Chat History Count:**", len(st.session_state.chat_history))
    st.write("**Patient Report Available:**", bool(st.session_state.patient_report))
    if st.session_state.patient_report:
        st.json(st.session_state.patient_report)
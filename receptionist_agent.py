import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from dotenv import load_dotenv
from db import get_patient_report

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

# Improved context-aware prompt that considers conversation flow
template = ChatPromptTemplate.from_messages([
    ("system", """You are Maria, a warm and empathetic AI medical receptionist. 

Key Guidelines:
- Review the conversation history to understand the context and avoid repetition
- DON'T greet the patient again if you've already greeted them
- Be conversational and natural - avoid robotic responses
- Keep responses under 5 lines and personalized
- If symptoms like pain, fever, swelling, bleeding, or concerning changes are mentioned, suggest clinical consultation
- Use the patient's information contextually, not as a checklist
- Respond naturally to follow-up questions and acknowledgments

Patient Context Available:
- Name: {patient_name}
- Diagnosis: {diagnosis} 
- Discharge Date: {discharge_date}
- Current Medications: {medications}
- Dietary Restrictions: {diet}
- Upcoming Follow-up: {follow_up}
- Warning Signs to Watch: {warning_signs}
- Discharge Instructions: {instructions}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

class ReceptionistAgent:
    def __init__(self):
        self.patient_name = None
        self.patient_report = None
        self.state = 'ask_name'
        self.chat_history = InMemoryChatMessageHistory()
        self.conversation_stage = 'initial'  # Track conversation progression
        self.topics_covered = set()  # Track what we've already discussed

        self.contextual_chain = RunnableWithMessageHistory(
            template | llm | StrOutputParser(),
            lambda session_id: self.chat_history,
            input_messages_key="user_input",
            history_messages_key="history"
        )

    def extract_name(self, user_input):
        import re
        match = re.search(r"my name is ([a-zA-Z ]+)", user_input.lower())
        if match:
            return match.group(1).strip().title()
        return user_input.strip().title()

    def analyze_user_input(self, user_input):
        """Analyze user input to determine intent and response type needed"""
        user_lower = user_input.lower()
        
        # Check for medical concerns
        medical_keywords = ['pain', 'swelling', 'fever', 'problem', 'concern', 'help', 'advice', 
                          'shortness', 'urine', 'blood', 'dizzy', 'nausea', 'vomit', 'chest', 
                          'breathing', 'headache', 'rash', 'infection']
        
        # Check for simple acknowledgments
        simple_responses = ['yes', 'yeah', 'ok', 'okay', 'good', 'fine', 'alright', 'thanks', 
                          'thank you', 'got it', 'understood', 'sure']
        
        # Check for negative responses
        negative_responses = ['no', 'not really', 'nothing', 'nope', 'not good', 'bad', 'worse']
        
        has_medical_concern = any(keyword in user_lower for keyword in medical_keywords)
        is_simple_acknowledgment = user_lower.strip() in simple_responses
        is_negative_response = any(neg in user_lower for neg in negative_responses)
        
        return {
            'has_medical_concern': has_medical_concern,
            'is_simple_acknowledgment': is_simple_acknowledgment,
            'is_negative_response': is_negative_response,
            'is_question': '?' in user_input,
            'mentions_medication': any(word in user_lower for word in ['medicine', 'medication', 'pills', 'drug']),
            'mentions_diet': any(word in user_lower for word in ['diet', 'food', 'eat', 'drink']),
            'mentions_appointment': any(word in user_lower for word in ['appointment', 'follow-up', 'visit', 'doctor'])
        }

    def get_contextual_response_guidance(self, user_analysis):
        """Provide specific guidance based on conversation context"""
        guidance = ""
        
        if self.conversation_stage == 'post_greeting':
            if user_analysis['is_simple_acknowledgment']:
                guidance = "The patient acknowledged your greeting. Ask a follow-up question about their recovery or how they're feeling, but don't repeat the greeting or medical information already shared."
            elif user_analysis['is_negative_response']:
                guidance = "The patient indicated they're not doing well. Show empathy and ask for more details about their concerns."
        
        elif self.conversation_stage == 'ongoing':
            if user_analysis['is_simple_acknowledgment']:
                guidance = "The patient is acknowledging your previous message. Provide a natural follow-up or ask if they have any other questions/concerns."
            elif user_analysis['mentions_medication']:
                guidance = "Focus on medication-related guidance without repeating their full medical history."
            elif user_analysis['mentions_diet']:
                guidance = "Focus on dietary guidance specific to their condition."
                
        return guidance

    def interact(self, user_input, session_id="user-session"):
        logging.info(f"Receptionist - State: {self.state}, Stage: {self.conversation_stage}, Input: {user_input}")

        if self.state == 'ask_name':
            self.patient_name = self.extract_name(user_input)
            report, status = get_patient_report(self.patient_name)

            if status == 'not_found':
                logging.warning(f"Patient not found: {self.patient_name}")
                return "I'm sorry, I couldn't find your record in our system. Could you please double-check the spelling of your name?", False
            elif status == 'multiple_found':
                return "I found multiple patients with that name. Could you please provide your full name or date of birth to help me locate the correct record?", False
            else:
                self.patient_report = report
                self.state = 'follow_up'
                self.conversation_stage = 'post_greeting'
                
                # Personalized greeting based on patient info
                greeting = f"Hi {self.patient_name}! I have your discharge information from {report['discharge_date']} for {report['primary_diagnosis']}. How are you feeling today?"
                return greeting, True

        elif self.state == 'follow_up':
            user_analysis = self.analyze_user_input(user_input)
            
            # Route to clinical if medical concerns detected
            if user_analysis['has_medical_concern']:
                self.state = 'route_clinical'
                return "I understand you have some medical concerns. Let me connect you with our Clinical AI Agent who can better assist you with those symptoms.", 'route_clinical'

            if self.patient_report:
                # Update conversation stage
                if self.conversation_stage == 'post_greeting':
                    self.conversation_stage = 'ongoing'
                
                # Get contextual guidance
                guidance = self.get_contextual_response_guidance(user_analysis)
                
                inputs = {
                    "patient_name": self.patient_report.get("patient_name", "Patient"),
                    "diagnosis": self.patient_report.get("primary_diagnosis", "N/A"),
                    "discharge_date": self.patient_report.get("discharge_date", "N/A"),
                    "medications": ", ".join(self.patient_report.get("medications", [])),
                    "diet": self.patient_report.get("dietary_restrictions", "N/A"),
                    "follow_up": self.patient_report.get("follow_up", "N/A"),
                    "warning_signs": self.patient_report.get("warning_signs", "N/A"),
                    "instructions": self.patient_report.get("discharge_instructions", "N/A"),
                    "user_input": user_input + (f"\n\nContext Guidance: {guidance}" if guidance else "")
                }
                
                result = self.contextual_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})
                return result.strip(), True
            else:
                return "I'm having trouble accessing your medical records. Could you please confirm your name again?", False

        elif self.state == 'route_clinical':
            return "[Clinical Agent takes over]", 'handoff'

        else:
            return "I'm not sure how to help with that. Could you please rephrase your question?", False

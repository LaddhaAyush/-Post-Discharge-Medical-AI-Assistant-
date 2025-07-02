import re
import logging
from datetime import datetime
from db import get_patient_report
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

class ReceptionistAgent:
    def __init__(self):
        self.patient_name = None
        self.patient_report = None
        self.state = 'greeting'
        self.conversation_memory = []
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self._initialize_llm()
        self._create_prompt_templates()

    def _initialize_llm(self):
        """Initialize the language model for natural conversations."""
        try:
            self.llm = ChatGroq(
                api_key=self.GROQ_API_KEY,
                model="llama3-8b-8192",
                temperature=0.8,  # More conversational and warm
                max_tokens=300
            )
            logging.info("Receptionist Agent: LLM initialized successfully")
        except Exception as e:
            logging.error(f"Receptionist Agent: Error initializing LLM: {e}")
            self.llm = None

    def _create_prompt_templates(self):
        """Create natural conversation prompts for different scenarios."""
        
        # Greeting and name collection
        self.greeting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Maria, a friendly and professional medical receptionist at a post-discharge care center. You have a warm, welcoming personality and years of experience helping patients.

Your current task is to get the patient's name so you can look up their discharge information. Be natural and conversational - like you're talking to someone who just walked into your office.

Guidelines:
- Be warm and welcoming
- Explain briefly what you do (help with post-discharge care)
- Ask for their name naturally
- Don't be robotic or use templates
- Keep it conversational and brief"""),
            
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "{user_input}")
        ])

        # Patient information and follow-up
        self.patient_care_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Maria, a caring medical receptionist who has just found the patient's discharge information. Your role is to:

- Welcome the patient warmly using their name
- Briefly mention their discharge date and condition (to confirm identity)
- Ask caring questions about how they're doing
- Listen for any medical concerns that need clinical attention
- Provide basic information from their discharge report when asked
- Route to clinical care when needed

Patient Information:
Name: {patient_name}
Discharge Date: {discharge_date}
Condition: {primary_diagnosis}
Medications: {medications}

Be natural, caring, and professional. Show genuine interest in their wellbeing."""),
            
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "{user_input}")
        ])

        # Medical concern detection
        self.concern_routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Maria, a medical receptionist who needs to determine if a patient's concern requires clinical attention.

The patient just shared: "{user_input}"

Respond naturally and professionally. If this sounds like a medical concern that needs clinical expertise, offer to connect them with our clinical specialist. If it's just a general check-in, respond supportively.

Medical concerns include: symptoms, pain, medication issues, side effects, complications, worries about their condition, etc."""),
            ("human", "{user_input}")
        ])

    def _extract_name_intelligently(self, user_input):
        """Extract patient name using multiple strategies."""
        user_input = user_input.strip()
        
        # Direct patterns
        patterns = [
            r"my name is ([a-zA-Z .'-]+)",
            r"i'?m ([a-zA-Z .'-]+)",
            r"i am ([a-zA-Z .'-]+)",
            r"it'?s ([a-zA-Z .'-]+)",
            r"this is ([a-zA-Z .'-]+)",
            r"([a-zA-Z .'-]+) here",
            r"^([a-zA-Z .'-]+)$"  # Just a name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                name = match.group(1).strip().title()
                # Validate name (at least 2 characters, mostly letters)
                if len(name) >= 2 and sum(c.isalpha() or c.isspace() for c in name) / len(name) > 0.7:
                    return name
        
        # Fallback: if input looks like a name (mostly letters and spaces)
        if user_input.replace(' ', '').replace('.', '').replace("'", '').isalpha() and len(user_input) >= 2:
            return user_input.title()
        
        return None

    def _get_patient_context(self):
        """Get patient context for prompts."""
        if not self.patient_report:
            return {}
        
        return {
            'patient_name': self.patient_report.get('patient_name', 'Unknown'),
            'discharge_date': self.patient_report.get('discharge_date', 'Unknown'),
            'primary_diagnosis': self.patient_report.get('primary_diagnosis', 'Unknown'),
            'medications': ', '.join(self.patient_report.get('medications', [])) or 'None listed'
        }

    def _manage_conversation_memory(self, user_message, ai_response):
        """Manage conversation memory efficiently."""
        self.conversation_memory.append(HumanMessage(content=user_message))
        self.conversation_memory.append(AIMessage(content=ai_response))
        
        # Keep only last 8 messages (4 exchanges)
        if len(self.conversation_memory) > 8:
            self.conversation_memory = self.conversation_memory[-8:]

    def _answer_from_report(self, user_input):
        """Answer questions directly from the patient report."""
        if not self.patient_report:
            return None
            
        user_lower = user_input.lower()
        
        # Map keywords to report fields with natural responses
        if any(word in user_lower for word in ['medication', 'medicine', 'pills', 'drugs', 'taking']):
            meds = self.patient_report.get('medications', [])
            if meds:
                return f"According to your discharge report, you're prescribed: {', '.join(meds)}. Are you having any issues with these medications?"
            else:
                return "I don't see any medications listed in your discharge report. You might want to check with your doctor about this."
            
        elif any(word in user_lower for word in ['diet', 'food', 'eat', 'restriction', 'dietary']):
            restrictions = self.patient_report.get('dietary_restrictions', '')
            if restrictions:
                return f"Your dietary guidelines are: {restrictions}. How are you managing with these changes?"
            else:
                return "I don't see specific dietary restrictions in your report. Your doctor may have discussed this with you verbally."
            
        elif any(word in user_lower for word in ['follow', 'appointment', 'visit', 'next', 'when']):
            followup = self.patient_report.get('follow_up', '')
            if followup:
                return f"Your follow-up plan is: {followup}. Have you scheduled this appointment yet?"
            else:
                return "I don't see specific follow-up instructions in your report. You might want to contact your doctor's office directly."
            
        elif any(word in user_lower for word in ['warning', 'signs', 'watch', 'symptoms', 'look for']):
            warnings = self.patient_report.get('warning_signs', '')
            if warnings:
                return f"You should watch for these warning signs: {warnings}. Are you experiencing any of these symptoms?"
            else:
                return "I don't see specific warning signs listed in your report. Your doctor may have discussed what to watch for."
            
        elif any(word in user_lower for word in ['instructions', 'discharge', 'care', 'home']):
            instructions = self.patient_report.get('discharge_instructions', '')
            if instructions:
                return f"Your discharge instructions include: {instructions}. Do you have any questions about following these?"
            else:
                return "I don't see detailed discharge instructions in your report. You should have received these when you left the hospital."
            
        elif any(word in user_lower for word in ['diagnosis', 'condition', 'what', 'problem']):
            diagnosis = self.patient_report.get('primary_diagnosis', '')
            if diagnosis:
                return f"Your primary diagnosis is: {diagnosis}. How are you feeling about managing this condition?"
            else:
                return "I don't see a specific diagnosis listed in your report."
        
        return None

    def _detect_medical_concern(self, user_input):
        """Detect if the input contains medical concerns."""
        medical_keywords = [
            'pain', 'hurt', 'ache', 'sore', 'swelling', 'swollen', 'fever', 'temperature',
            'nausea', 'vomit', 'dizzy', 'tired', 'fatigue', 'shortness', 'breath', 'breathing',
            'urine', 'blood', 'pressure', 'chest', 'heart', 'worried', 'concern', 'problem',
            'side effect', 'reaction', 'rash', 'itching', 'headache', 'stomach', 'bowel',
            'constipation', 'diarrhea', 'sleeping', 'sleep', 'appetite', 'weight', 'emergency'
        ]
        
        return any(keyword in user_input.lower() for keyword in medical_keywords)

    def interact(self, user_input):
        """Main interaction method with natural conversation flow."""
        logging.info(f"Receptionist Agent - State: {self.state}, Input: {user_input[:50]}...")
        
        try:
            if self.state == 'greeting':
                return self._handle_greeting(user_input)
            elif self.state == 'patient_care':
                return self._handle_patient_care(user_input)
            else:
                return self._handle_fallback(user_input)
                
        except Exception as e:
            logging.error(f"Receptionist Agent error: {e}")
            return "I'm sorry, I'm having some technical difficulties. Let me try to help you anyway.", False

    def _handle_greeting(self, user_input):
        """Handle the initial greeting and name collection."""
        # Try to extract name
        extracted_name = self._extract_name_intelligently(user_input)
        
        if extracted_name:
            self.patient_name = extracted_name
            report, status = get_patient_report(self.patient_name)
            
            if status == 'found':
                self.patient_report = report
                self.state = 'patient_care'
                
                # Generate natural welcome message
                if self.llm:
                    try:
                        patient_context = self._get_patient_context()
                        welcome_chain = self.patient_care_prompt | self.llm | StrOutputParser()
                        
                        # Create a natural welcome
                        welcome_message = welcome_chain.invoke({
                            "user_input": f"Hello, I'm {self.patient_name}",
                            "conversation_history": [],
                            **patient_context
                        })
                        
                        self._manage_conversation_memory(user_input, welcome_message)
                        logging.info(f"Patient found and welcomed: {self.patient_name}")
                        return welcome_message, True
                        
                    except Exception as e:
                        logging.error(f"Error generating welcome message: {e}")
                
                # Fallback welcome
                fallback_message = f"Hi {self.patient_name}! I found your discharge information from {report.get('discharge_date', 'recently')} for {report.get('primary_diagnosis', 'your condition')}. How are you feeling today?"
                self._manage_conversation_memory(user_input, fallback_message)
                return fallback_message, True
                
            elif status == 'not_found':
                response = f"I'm sorry, I couldn't find a discharge record for {self.patient_name}. Could you double-check the spelling of your name, or perhaps you were registered under a different name?"
                self._manage_conversation_memory(user_input, response)
                return response, False
                
            elif status == 'multiple_found':
                response = f"I found multiple patients named {self.patient_name}. Could you provide your middle initial or birth date to help me find the right record?"
                self._manage_conversation_memory(user_input, response)
                return response, False
            
            else:  # error
                response = "I'm having trouble accessing the patient records right now. Could you try again in a moment?"
                self._manage_conversation_memory(user_input, response)
                return response, False
        
        else:
            # Couldn't extract name, ask again naturally
            if self.llm:
                try:
                    chain = self.greeting_prompt | self.llm | StrOutputParser()
                    response = chain.invoke({
                        "user_input": user_input,
                        "conversation_history": self.conversation_memory
                    })
                    self._manage_conversation_memory(user_input, response)
                    return response, False
                except Exception as e:
                    logging.error(f"Error generating greeting response: {e}")
            
            # Fallback
            response = "I'd be happy to help you! Could you please tell me your name so I can look up your discharge information?"
            self._manage_conversation_memory(user_input, response)
            return response, False

    def _handle_patient_care(self, user_input):
        """Handle patient care conversations."""
        # First, try to answer from report
        report_answer = self._answer_from_report(user_input)
        if report_answer:
            self._manage_conversation_memory(user_input, report_answer)
            return report_answer, True
        
        # Check for medical concerns
        if self._detect_medical_concern(user_input):
            self.state = 'routing_clinical'
            
            if self.llm:
                try:
                    chain = self.concern_routing_prompt | self.llm | StrOutputParser()
                    response = chain.invoke({"user_input": user_input})
                    
                    # Add routing indication
                    if "clinical" not in response.lower() and "specialist" not in response.lower():
                        response += " Let me connect you with our clinical specialist who can better address your concerns."
                    
                    self._manage_conversation_memory(user_input, response)
                    return response, 'route_clinical'
                    
                except Exception as e:
                    logging.error(f"Error generating concern response: {e}")
            
            # Fallback
            response = "That sounds like something our clinical specialist should address. Let me connect you with them right away."
            self._manage_conversation_memory(user_input, response)
            return response, 'route_clinical'
        
        # General conversation
        if self.llm:
            try:
                patient_context = self._get_patient_context()
                chain = self.patient_care_prompt | self.llm | StrOutputParser()
                
                response = chain.invoke({
                    "user_input": user_input,
                    "conversation_history": self.conversation_memory,
                    **patient_context
                })
                
                self._manage_conversation_memory(user_input, response)
                return response, True
                
            except Exception as e:
                logging.error(f"Error generating patient care response: {e}")
        
        # Fallback
        response = "Thank you for sharing that with me. Is there anything specific about your discharge instructions or medications that I can help clarify?"
        self._manage_conversation_memory(user_input, response)
        return response, True

    def _handle_fallback(self, user_input):
        """Handle unexpected states or errors."""
        response = "I'm here to help! Let me know if you have any questions about your discharge information or if you'd like to speak with our clinical specialist."
        self._manage_conversation_memory(user_input, response)
        return response, True

    def get_patient_summary(self):
        """Get a summary of the current patient."""
        if not self.patient_report:
            return "No patient information available."
        
        return f"Patient: {self.patient_report.get('patient_name', 'Unknown')}, Condition: {self.patient_report.get('primary_diagnosis', 'Unknown')}, Discharge: {self.patient_report.get('discharge_date', 'Unknown')}"
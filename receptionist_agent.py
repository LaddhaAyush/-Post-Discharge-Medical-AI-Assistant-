import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
from db import get_patient_report

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

# Pydantic model for structured name extraction
class NameExtraction(BaseModel):
    """Extract patient name from user input"""
    patient_name: str = Field(description="The full name of the patient extracted from the user's message. If no clear name is found, return 'NOT_FOUND'")
    confidence: str = Field(description="High, Medium, or Low confidence in the extracted name")
    reasoning: str = Field(description="Brief explanation of how the name was extracted")

# Create name extraction chain
name_extraction_parser = PydanticOutputParser(pydantic_object=NameExtraction)
name_extraction_prompt = ChatPromptTemplate.from_template(
    """You are an expert at extracting patient names from conversational text.

Extract the patient name from the following user input. Look for patterns like:
- "Hi, I am [Name]"
- "My name is [Name]"
- "I'm [Name]" 
- "This is [Name]"
- "[Name] here"
- Just a plain name like "John Smith"

IMPORTANT: Return the patient_name as a SINGLE STRING, not a list or array.

Be careful to:
- Extract ONLY the person's name, not titles, greetings, or other words
- Handle both first name only and full names
- Ignore common words like "hi", "hello", "I", "am", "is", etc.
- If the input doesn't contain a clear name, return "NOT_FOUND"
- Return the name as a single string value

Examples:
- Input: "Hi, I am John Smith" → patient_name: "John Smith"
- Input: "My name is Alice" → patient_name: "Alice"
- Input: "Hello there" → patient_name: "NOT_FOUND"

User Input: {user_input}

{format_instructions}
"""
)

name_extraction_chain = name_extraction_prompt | llm | name_extraction_parser

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
        self.reset_state()

        self.contextual_chain = RunnableWithMessageHistory(
            template | llm | StrOutputParser(),
            lambda session_id: self.chat_history,
            input_messages_key="user_input",
            history_messages_key="history"
        )
        
        # Initialize name extraction chain
        self.name_extraction_chain = name_extraction_chain

    def reset_state(self):
        """Reset agent state for a new conversation/patient"""
        self.patient_name = None
        self.patient_report = None
        self.state = 'ask_name'
        self.chat_history = InMemoryChatMessageHistory()
        self.conversation_stage = 'initial'  # Track conversation progression
        self.topics_covered = set()  # Track what we've already discussed
        logging.info("Receptionist agent state reset for new conversation")

    def extract_name(self, user_input):
        """Extract patient name using structured output with LangChain"""
        try:
            # Use the structured output chain to extract name
            result = self.name_extraction_chain.invoke({
                "user_input": user_input,
                "format_instructions": name_extraction_parser.get_format_instructions()
            })
            
            logging.info(f"Name extraction result: {result.patient_name}, Confidence: {result.confidence}, Reasoning: {result.reasoning}")
            
            # Handle case where LLM returns a list instead of string
            patient_name = result.patient_name
            if isinstance(patient_name, list) and len(patient_name) > 0:
                patient_name = patient_name[0]
            
            # If high or medium confidence and name found, return it
            if result.confidence in ["High", "Medium"] and patient_name != "NOT_FOUND":
                return str(patient_name).strip().title()
            
            # If low confidence or not found, check if it's just a plain name
            elif patient_name == "NOT_FOUND":
                # Fallback to basic regex for simple name patterns
                import re
                # Remove common greeting words and check if remaining text looks like a name
                cleaned_input = re.sub(r'\b(hi|hello|hey|good|morning|afternoon|evening|i|am|my|name|is|this)\b', '', user_input.lower(), flags=re.IGNORECASE).strip()
                
                # Check if remaining text contains probable name pattern (2-4 words, mostly letters)
                name_pattern = re.match(r'^[a-zA-Z\s]{2,50}$', cleaned_input.strip())
                if name_pattern and len(cleaned_input.strip().split()) <= 4:
                    extracted_name = cleaned_input.strip().title()
                    logging.info(f"Fallback extraction found: {extracted_name}")
                    return extracted_name
                
                # If still no luck, return the original input cleaned up
                return user_input.strip().title()
            
            else:
                return str(patient_name).strip().title()
                
        except Exception as e:
            logging.error(f"Error in structured name extraction: {e}")
            # Fallback to improved regex method
            import re
            
            # Try common patterns first
            patterns = [
                r"(?:hi|hello|hey),?\s+(?:i\s+am|i'm|my\s+name\s+is|this\s+is)\s+([a-zA-Z\s]+)",
                r"(?:my\s+name\s+is|i\s+am|i'm)\s+([a-zA-Z\s]+)",
                r"^([a-zA-Z\s]+)$"  # Just a plain name
            ]
            
            for pattern in patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    extracted_name = match.group(1).strip().title()
                    # Validate it looks like a reasonable name (not too long, not common words)
                    if len(extracted_name.split()) <= 4 and not any(word in extracted_name.lower() for word in ['help', 'info', 'discharge', 'need']):
                        logging.info(f"Regex fallback extraction: {extracted_name}")
                        return extracted_name
            
            # Final fallback
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

    def has_greeting_with_name(self, user_input):
        """Check if user input contains both greeting and name introduction"""
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        name_patterns = ['i am', 'i\'m', 'my name is', 'this is', 'here']
        
        user_lower = user_input.lower()
        has_greeting = any(pattern in user_lower for pattern in greeting_patterns)
        has_name_intro = any(pattern in user_lower for pattern in name_patterns)
        
        return has_greeting and has_name_intro

    def is_conversation_ending(self, user_input):
        """Check if the user is ending the conversation"""
        ending_keywords = ['bye', 'goodbye', 'see you', 'thank you and goodbye', 
                          'thanks, bye', 'that\'s all', 'end', 'quit', 'exit']
        user_lower = user_input.lower().strip()
        return any(keyword in user_lower for keyword in ending_keywords) or user_lower in ['bye', 'goodbye', 'thanks', 'thank you']

    def is_new_conversation_start(self, user_input):
        """Check if this looks like the start of a new conversation"""
        starting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                           'good evening', 'i need help', 'can you help', 'my name is']
        user_lower = user_input.lower().strip()
        return any(keyword in user_lower for keyword in starting_keywords) or user_lower in ['hello', 'hi', 'hey']

    def handle_conversation_ending(self):
        """Handle the end of conversation and provide closing response"""
        return "Thank you for using our service! Take care and don't hesitate to reach out if you have any questions. Goodbye!"

    def interact(self, user_input, session_id="user-session"):
        logging.info(f"Receptionist - State: {self.state}, Stage: {self.conversation_stage}, Input: {user_input}")

        # Check if user is ending the conversation
        if self.is_conversation_ending(user_input):
            response = self.handle_conversation_ending()
            self.reset_state()  # Reset for next user
            return response, 'conversation_ended'

        # If we're in follow_up state but user is starting a new conversation, reset
        if self.state != 'ask_name' and self.is_new_conversation_start(user_input):
            logging.info("Detected new conversation start while in follow_up state - resetting")
            self.reset_state()

        if self.state == 'ask_name':
            self.patient_name = self.extract_name(user_input)
            report, status = get_patient_report(self.patient_name)

            if status == 'not_found':
                logging.warning(f"Patient not found: {self.patient_name}")
                # Check if user provided a greeting without clear name
                if self.has_greeting_with_name(user_input) or any(word in user_input.lower() for word in ['hello', 'hi', 'hey']):
                    return "Hello! I'd be happy to help you with your discharge information. I couldn't find your record in our system. Could you please provide your full name as it appears in your medical records?", False
                else:
                    return "I'm sorry, I couldn't find your record in our system. Could you please double-check the spelling of your name?", False
            elif status == 'multiple_found':
                return "I found multiple patients with that name. Could you please provide your full name or date of birth to help me locate the correct record?", False
            else:
                self.patient_report = report
                self.state = 'follow_up'
                self.conversation_stage = 'post_greeting'
                
                # Create more natural greeting based on how user introduced themselves
                if self.has_greeting_with_name(user_input):
                    # User said something like "Hi, I am John Smith"
                    greeting = f"Hello {self.patient_name}! Nice to meet you. I have your discharge information from {report['discharge_date']} for {report['primary_diagnosis']}. How are you feeling today?"
                else:
                    # User just provided name
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

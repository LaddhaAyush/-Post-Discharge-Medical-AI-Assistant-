import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

class ClinicalAgent:
    def __init__(self):
        self.FAISS_INDEX_PATH = "data/nephro_faiss.index"
        self.NEPHRO_TXT_PATH = "data/nephro.txt"
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.patient_report = None
        self.k = 3
        self.conversation_memory = []
        self._load_components()

    def _load_components(self):
        """Load all necessary components for the clinical agent."""
        try:
            # Load text chunks
            with open(self.NEPHRO_TXT_PATH, "r", encoding="utf-8") as f:
                self.chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]
            
            # Load FAISS index
            self.index = faiss.read_index(self.FAISS_INDEX_PATH)
            
            # Initialize embedding model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Initialize LLM
            self.llm = ChatGroq(
                api_key=self.GROQ_API_KEY, 
                model="llama3-8b-8192",
                temperature=0.7,  # More natural responses
                max_tokens=500
            )
            
            # Create conversational prompt template
            self._create_prompt_templates()
            
            logging.info("Clinical Agent components loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading clinical components: {e}")
            raise

    def _create_prompt_templates(self):
        """Create sophisticated prompt templates for natural conversation."""
        
        # Main conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Dr. Sarah, a warm and experienced nephrology nurse practitioner. You have years of experience helping kidney patients after discharge. Your role is to:

- Provide compassionate, personalized medical guidance
- Use the patient's medical context naturally in conversation
- Reference medical literature when helpful, but explain in simple terms
- Show genuine concern for the patient's wellbeing
- Ask thoughtful follow-up questions to better understand their situation
- Always remind patients to contact their healthcare team for urgent concerns

Patient Context:
Name: {patient_name}
Diagnosis: {diagnosis}
Current Medications: {medications}
Discharge Date: {discharge_date}

Key Guidelines:
- Speak naturally and conversationally, like you're talking to a real patient
- Use the patient's name when appropriate
- Reference their specific condition and medications when relevant
- Be encouraging and supportive
- Never sound robotic or templated"""),
            
            MessagesPlaceholder(variable_name="conversation_history"),
            
            ("human", "{current_message}"),
            
            ("system", "Medical Reference Context (use naturally if relevant):\n{medical_context}")
        ])

        # Follow-up question generator
        self.followup_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the patient's message and medical context, generate ONE natural follow-up question that shows you care about their wellbeing. Make it specific to their situation.

Patient: {patient_name} - {diagnosis}
Their message: {patient_message}

Generate a caring, specific follow-up question (just the question, nothing else):"""),
            ("human", "{patient_message}")
        ])

    def set_patient_report(self, report):
        """Set patient report and log the action."""
        self.patient_report = report
        patient_name = report.get('patient_name', 'Unknown') if report else 'Unknown'
        logging.info(f"Clinical Agent: Patient context set for {patient_name}")

    def _get_patient_context(self):
        """Extract patient context for prompts."""
        if not self.patient_report:
            return {
                'patient_name': 'the patient',
                'diagnosis': 'kidney condition',
                'medications': 'medications as prescribed',
                'discharge_date': 'recent discharge'
            }
        
        return {
            'patient_name': self.patient_report.get('patient_name', 'the patient'),
            'diagnosis': self.patient_report.get('primary_diagnosis', 'kidney condition'),
            'medications': ', '.join(self.patient_report.get('medications', ['medications as prescribed'])),
            'discharge_date': self.patient_report.get('discharge_date', 'recent discharge')
        }

    def retrieve_from_rag(self, query):
        """Retrieve relevant medical information using RAG."""
        try:
            query_vec = self.model.encode([query])
            D, I = self.index.search(np.array(query_vec).astype("float32"), self.k)
            
            retrieved_chunks = []
            for i in I[0]:
                if i < len(self.chunks):
                    retrieved_chunks.append(self.chunks[i])
            
            logging.info(f"RAG retrieval: Found {len(retrieved_chunks)} relevant chunks")
            return retrieved_chunks
            
        except Exception as e:
            logging.error(f"RAG retrieval error: {e}")
            return []

    def _search_web(self, query, num_results=2):
        """Search DuckDuckGo for medical information."""
        try:
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    f"nephrology kidney {query}", 
                    region='wt-wt', 
                    safesearch='moderate', 
                    max_results=num_results
                ))
                
                for result in search_results:
                    title = result.get('title', '')
                    body = result.get('body', '')
                    source = result.get('href', '')
                    results.append(f"• {title}: {body[:200]}... (Source: {source})")
            
            web_context = "\n".join(results)
            logging.info(f"Web search completed: {len(results)} results found")
            return web_context
            
        except Exception as e:
            logging.error(f"Web search error: {e}")
            return "Web search temporarily unavailable."

    def _should_use_web_search(self, query, rag_results):
        """Determine if web search is needed."""
        # Use web search if RAG results are insufficient
        if not rag_results or len(" ".join(rag_results)) < 100:
            return True
        
        # Use web search for recent/current information
        recent_keywords = ['latest', 'recent', 'new', 'current', 'breakthrough', 'update', 'today']
        return any(keyword in query.lower() for keyword in recent_keywords)

    def _manage_conversation_memory(self, user_message, ai_response):
        """Manage conversation memory efficiently."""
        self.conversation_memory.append(HumanMessage(content=user_message))
        self.conversation_memory.append(AIMessage(content=ai_response))
        
        # Keep only last 10 messages (5 exchanges)
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]

    def _handle_conversation_references(self, user_input):
        """Handle questions about the conversation history."""
        conversation_keywords = [
            "what did i just say", "what did i tell you", "what was my last question",
            "what did i ask", "repeat my last message", "what was my previous question"
        ]
        
        if any(keyword in user_input.lower() for keyword in conversation_keywords):
            # Find the last user message
            for message in reversed(self.conversation_memory):
                if isinstance(message, HumanMessage):
                    return f"You just asked me: \"{message.content}\""
            return "I don't see any previous messages from you in our conversation."
        
        return None

    def interact(self, user_input):
        """Main interaction method with natural conversation flow."""
        logging.info(f"Clinical Agent processing: {user_input[:100]}...")
        
        try:
            # Handle conversation references
            conv_response = self._handle_conversation_references(user_input)
            if conv_response:
                self._manage_conversation_memory(user_input, conv_response)
                return conv_response

            # Get medical context through RAG
            rag_results = self.retrieve_from_rag(user_input)
            medical_context = "\n\n".join(rag_results)
            
            # Use web search if needed
            if self._should_use_web_search(user_input, rag_results):
                web_results = self._search_web(user_input)
                medical_context = f"Recent Information:\n{web_results}\n\nReference Material:\n{medical_context}"
                logging.info("Using web search results for context")

            # Get patient context
            patient_context = self._get_patient_context()

            # Prepare input for the prompt
            prompt_input = {
                "conversation_history": self.conversation_memory[-8:],  # last 4 exchanges
                "medical_context": medical_context,
                "patient_name": patient_context['patient_name'],
                "diagnosis": patient_context['diagnosis'],
                "medications": patient_context['medications'],
                "discharge_date": patient_context['discharge_date'],
                "current_message": user_input
            }

            # Create the conversation chain
            chain = self.conversation_prompt | self.llm | StrOutputParser()
            response = chain.invoke(prompt_input)

            # Generate follow-up question
            followup_chain = self.followup_prompt | self.llm | StrOutputParser()
            followup = followup_chain.invoke({
                "patient_message": user_input,
                **patient_context
            })

            # Clean up responses
            response = response.strip()
            followup = followup.strip()

            # Combine response with follow-up
            if followup and not followup.lower().startswith('based on'):
                full_response = f"{response}\n\n{followup}"
            else:
                full_response = response

            # Manage conversation memory
            self._manage_conversation_memory(user_input, full_response)
            
            logging.info("Clinical Agent: Response generated successfully")
            return full_response

        except Exception as e:
            logging.error(f"Clinical Agent error: {e}")
            error_response = "I'm having some technical difficulties right now. For any urgent medical concerns, please contact your healthcare provider immediately."
            self._manage_conversation_memory(user_input, error_response)
            return error_response

    def get_conversation_summary(self):
        """Get a summary of the current conversation."""
        if not self.conversation_memory:
            return "No conversation history available."
        
        # Create a simple summary of recent exchanges
        recent_topics = []
        for msg in self.conversation_memory[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                recent_topics.append(f"Patient asked about: {msg.content[:50]}...")
        
        return "Recent discussion topics:\n" + "\n".join(recent_topics) if recent_topics else "No recent topics to summarize."
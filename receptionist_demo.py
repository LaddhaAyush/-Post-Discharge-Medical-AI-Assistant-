from receptionist_agent import ReceptionistAgent
from clinical_agent import ClinicalAgent
import logging
from datetime import datetime
import os
import sys

def setup_logging():
    """Set up comprehensive logging system."""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler for detailed logs
    file_handler = logging.FileHandler(
        f'logs/system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Create console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def print_welcome():
    """Print professional welcome message."""
    print("\n" + "="*70)
    print("🏥  POST-DISCHARGE MEDICAL AI ASSISTANT")
    print("="*70)
    print("✨  Welcome to your personalized post-discharge care system")
    print("🤖  Powered by advanced AI agents for comprehensive support")
    print("-"*70)
    print("⚠️   IMPORTANT DISCLAIMER:")
    print("    This is an AI assistant for educational purposes only.")
    print("    Always consult healthcare professionals for medical advice.")
    print("    For emergencies, contact 911 or your emergency services.")
    print("="*70)

def print_agent_transition(from_agent, to_agent, reason=""):
    """Print professional agent transition messages."""
    transitions = {
        'receptionist_to_clinical': {
            'symbol': '🏥',
            'title': 'CLINICAL SPECIALIST ACTIVATED',
            'message': 'Connecting you with our nephrology clinical specialist...'
        },
        'clinical_to_receptionist': {
            'symbol': '📋',
            'title': 'RETURNING TO RECEPTION',
            'message': 'Transferring you back to reception services...'
        }
    }
    
    transition_key = f"{from_agent}_to_{to_agent}"
    config = transitions.get(transition_key, {
        'symbol': '🔄',
        'title': 'AGENT TRANSFER',
        'message': f'Switching from {from_agent} to {to_agent}...'
    })
    
    print(f"\n{'-'*50}")
    print(f"{config['symbol']} {config['title']}")
    print(f"{'-'*50}")
    print(f"💬 {config['message']}")
    if reason:
        print(f"📝 Reason: {reason}")
    print(f"{'-'*50}")

def get_user_input(prompt="You: "):
    """Get user input with error handling."""
    try:
        user_input = input(f"\n{prompt}").strip()
        return user_input
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Take care!")
        return None
    except EOFError:
        print("\n\n👋 Session ended. Take care!")
        return None

def handle_system_commands(user_input):
    """Handle special system commands."""
    if not user_input:
        return 'empty'
    
    lower_input = user_input.lower()
    
    # Exit commands
    if lower_input in ['exit', 'quit', 'bye', 'goodbye', 'end']:
        return 'exit'
    
    # Help commands
    if lower_input in ['help', '?', 'commands']:
        return 'help'
    
    # Status commands
    if lower_input in ['status', 'info', 'about']:
        return 'status'
    
    return 'continue'

def print_help():
    """Print help information."""
    print("\n" + "="*50)
    print("📚 HELP & COMMANDS")
    print("="*50)
    print("💬 Chat naturally with the AI agents")
    print("🏥 Clinical questions will be routed to the specialist")
    print("📋 Administrative questions handled by reception")
    print("-"*50)
    print("Commands:")
    print("  • exit, quit, bye - End the session")
    print("  • help, ? - Show this help")
    print("  • status - Show current session info")
    print("="*50)

def main():
    """Enhanced main application with professional UI and error handling."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Print welcome
        print_welcome()
        
        # Initialize agents
        logger.info("Initializing AI agents...")
        
        try:
            receptionist = ReceptionistAgent()
            logger.info("Receptionist Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Receptionist Agent: {e}")
            print("❌ Error: Could not initialize reception services. Please check your configuration.")
            return
        
        try:
            clinical = ClinicalAgent()
            logger.info("Clinical Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Clinical Agent: {e}")
            print("❌ Error: Could not initialize clinical services. Please check your configuration.")
            return
        
        # Session variables
        current_agent = receptionist
        agent_name = "receptionist"
        session_start = datetime.now()
        interaction_count = 0
        
        logger.info("Session started - All systems ready")
        
        # Initial greeting from receptionist
        print("\n💬 Maria (Reception): Hello! I'm Maria, your post-discharge care coordinator. I'm here to help you with any questions about your recovery. What's your name?")
        
        # Main conversation loop
        while True:
            # Get user input
            user_input = get_user_input()
            
            if user_input is None:  # Interrupted
                break
            
            # Handle system commands
            cmd_result = handle_system_commands(user_input)
            
            if cmd_result == 'exit':
                print("\n👋 Thank you for using our post-discharge care system.")
                print("💙 Wishing you a smooth recovery! Take care!")
                logger.info("Session ended by user command")
                break
            
            elif cmd_result == 'empty':
                print("💬 Please share what's on your mind, or type 'help' for assistance.")
                continue
            
            elif cmd_result == 'help':
                print_help()
                continue
            
            elif cmd_result == 'status':
                session_duration = datetime.now() - session_start
                print(f"\n📊 Session Status:")
                print(f"   • Duration: {session_duration}")
                print(f"   • Interactions: {interaction_count}")
                print(f"   • Currently with: {agent_name.title()} Agent")
                if receptionist.patient_name:
                    print(f"   • Patient: {receptionist.patient_name}")
                continue
            
            elif cmd_result != 'continue':
                continue
            
            # Increment interaction counter
            interaction_count += 1
            logger.info(f"Processing interaction #{interaction_count} with {agent_name} agent")
            
            try:
                # Process with current agent
                if current_agent == receptionist:
                    response, status = current_agent.interact(user_input)
                    
                    if status == 'route_clinical':
                        # Display receptionist response
                        print(f"\n💬 Maria (Reception): {response}")
                        
                        # Transition to clinical agent
                        print_agent_transition('receptionist', 'clinical', 'Medical concern detected')
                        
                        # Switch agents
                        current_agent = clinical
                        agent_name = "clinical"
                        clinical.set_patient_report(receptionist.patient_report)
                        
                        logger.info("Agent handoff: Receptionist -> Clinical")
                        
                        # Welcome message from clinical agent
                        patient_name = receptionist.patient_name or "there"
                        print(f"\n💬 Dr. Sarah (Clinical): Hi {patient_name}! I'm Dr. Sarah, a nephrology specialist. I understand you have some medical concerns. I'm here to help - what's troubling you?")
                    
                    elif status == False:
                        print(f"\n💬 Maria (Reception): {response}")
                        if "couldn't find" in response.lower() or "multiple patients" in response.lower():
                            print("💡 Tip: Please make sure to spell your name exactly as it appears on your discharge papers.")
                    
                    else:
                        print(f"\n💬 Maria (Reception): {response}")
                
                else:  # clinical agent
                    response = current_agent.interact(user_input)
                    print(f"\n💬 Dr. Sarah (Clinical): {response}")
                    
                    # Check for return to receptionist requests
                    return_keywords = ['receptionist', 'reception', 'maria', 'back to reception', 'administrative']
                    if any(keyword in user_input.lower() for keyword in return_keywords):
                        print_agent_transition('clinical', 'receptionist', 'Administrative request')
                        current_agent = receptionist
                        agent_name = "receptionist"
                        logger.info("Agent handoff: Clinical -> Receptionist")
                        print(f"\n💬 Maria (Reception): Hi again! How can I help you with the administrative side of your care?")
            
            except Exception as e:
                logger.error(f"Error during interaction #{interaction_count}: {e}")
                print(f"\n❌ I apologize, but I encountered a technical issue. Let me try to help you anyway.")
                print("💡 For urgent medical concerns, please contact your healthcare provider directly.")
        
        # Session summary
        session_duration = datetime.now() - session_start
        logger.info(f"Session completed - Duration: {session_duration}, Interactions: {interaction_count}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Take care!")
        logger.info("Session interrupted by user")
    
    except Exception as e:
        logger.error(f"Critical system error: {e}")
        print("\n❌ A critical error occurred. Please restart the application.")
        print("💡 If the problem persists, please contact technical support.")

if __name__ == "__main__":
    main()
# app.py

# Initialize logging FIRST before any other imports
import logging_config

import streamlit as st
from streamlit import session_state
import time
import base64
import os
from dotenv import load_dotenv
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbot import ChatbotManager     # Import the ChatbotManager class
from logging_config import get_logger

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)

# Disable Streamlit's caching for this session to prevent re-initialization issues
@st.cache_resource
def init_app():
    """Initialize app only once."""
    logger.info("="*50)
    logger.info("RAG-Based LLM Chatbot Application Started")
    logger.info("="*50)
    return True

# Initialize app
init_app()

# Function to display the PDF of a given file
def displayPDF(file):
    # Reading the uploaded file
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables if not already present
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Document Buddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    try:
        st.image("sct.png", width=None)
    except:
        st.markdown("### 📚 Document Buddy")
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)
    logger.info(f"User navigated to: {choice}")

# Home Page
if choice == "🏠 Home":
    st.title("📄 Document Buddy App")
    st.markdown("""
    Welcome to **Document Buddy App**! 🚀

    **Built using Modern Open Source Stack:**
    - **LLM**: Qwen2.5 (via Ollama) - Fast and efficient language model
    - **Embeddings**: Nomic Embed Text v1.5 - High-quality text embeddings
    - **Vector DB**: Chroma Cloud - Cloud-based vector storage for scalability
    - **RAG**: Retrieval-Augmented Generation for accurate document-based responses

    **Features:**
    - **Upload Documents**: Easily upload your PDF documents
    - **Smart Embeddings**: Automatic vector creation and storage in Chroma Cloud
    - **Intelligent Chat**: Ask questions about your documents with RAG technology
    - **Real-time Processing**: Fast response times with optimized retrieval

    Enhance your document management experience with Document Buddy! 📚✨
    """)

# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface🦙")
    st.markdown("---")
    
    # Initialize ChatbotManager early to check for existing embeddings
    if st.session_state['chatbot_manager'] is None:
        logger.info("Initializing ChatbotManager to check for existing embeddings")
        st.session_state['chatbot_manager'] = ChatbotManager()
    
    # Check if embeddings already exist in Chroma Cloud
    embeddings_exist = st.session_state['chatbot_manager'].has_embeddings()
    
    # Add toggle to upload new document even if embeddings exist
    if embeddings_exist:
        col_toggle1, col_toggle2 = st.columns([3, 1])
        with col_toggle1:
            st.success("✅ Existing embeddings found in Chroma Cloud! You can chat directly.")
        with col_toggle2:
            update_embeddings = st.checkbox("📄 Upload New PDF")
    else:
        update_embeddings = False
    
    st.markdown("---")
    
    if embeddings_exist and not update_embeddings:
        # Show chat interface when embeddings exist and user doesn't want to upload
        st.header("💬 Chat with Document")
        
        # Display existing messages
        for msg in st.session_state['messages']:
            st.chat_message(msg['role']).markdown(msg['content'])

        # User input
        if user_input := st.chat_input("Type your message here..."):
            logger.info(f"User input received: {user_input[:100]}...")
            # Display user message
            st.chat_message("user").markdown(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.spinner("🤖 Responding..."):
                try:
                    # Get the chatbot response using the ChatbotManager
                    logger.info("Getting chatbot response")
                    answer = st.session_state['chatbot_manager'].get_response(user_input)
                    time.sleep(1)  # Simulate processing time
                    logger.info("Chatbot response generated successfully")
                except Exception as e:
                    logger.error(f"Error in chatbot response: {str(e)}")
                    answer = f"⚠️ An error occurred while processing your request: {e}"
            
            # Display chatbot message
            st.chat_message("assistant").markdown(answer)
            st.session_state['messages'].append({"role": "assistant", "content": answer})
            logger.info("Response displayed to user")
    
    # Show upload section if no embeddings exist OR user wants to update
    if (not embeddings_exist) or update_embeddings:
        # Upload and creation workflow
        if not embeddings_exist:
            st.info("📚 No embeddings found. Please upload a PDF to get started.")
        else:
            st.info("📝 Upload a new PDF to update embeddings and clear chat history.")
        st.markdown("---")

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Column 1: File Uploader and Preview
        with col1:
            st.header("📂 Upload Document")
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
            if uploaded_file is not None:
                logger.info(f"PDF file uploaded: {uploaded_file.name} (Size: {uploaded_file.size} bytes)")
                st.success("📄 File Uploaded Successfully!")
                # Display file name and size
                st.markdown(f"**Filename:** {uploaded_file.name}")
                st.markdown(f"**File Size:** {uploaded_file.size} bytes")
                
                # Display PDF preview using displayPDF function
                st.markdown("### 📖 PDF Preview")
                displayPDF(uploaded_file)
                
                # Save the uploaded file to a temporary location
                temp_pdf_path = "temp.pdf"
                logger.debug(f"Saving uploaded file to temporary location: {temp_pdf_path}")
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Store the temp_pdf_path in session_state
                st.session_state['temp_pdf_path'] = temp_pdf_path

        # Column 2: Create Embeddings
        with col2:
            st.header("🧠 Embeddings")
            create_embeddings = st.checkbox("✅ Create Embeddings")
            if create_embeddings:
                logger.info("User initiated embedding creation")
                if st.session_state['temp_pdf_path'] is None:
                    logger.warning("Embedding creation attempted without uploaded PDF")
                    st.warning("⚠️ Please upload a PDF first.")
                else:
                    try:
                        logger.info("Creating EmbeddingsManager instance")
                        # Initialize the EmbeddingsManager
                        embeddings_manager = EmbeddingsManager()
                        
                        with st.spinner("🔄 Embeddings are in process..."):
                            # Create embeddings
                            logger.info(f"Starting embedding process for file: {st.session_state['temp_pdf_path']}")
                            result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                            time.sleep(1)  # Optional: To show spinner for a bit longer
                        st.success(result)
                        logger.info("Embeddings created successfully")
                        
                        # Force re-initialization to pick up new embeddings
                        logger.info("Reinitializing ChatbotManager to load new embeddings")
                        st.session_state['chatbot_manager'] = ChatbotManager()
                        logger.info("ChatbotManager reinitialized successfully")
                        # Clear chat history when new embeddings are created
                        st.session_state['messages'] = []
                        logger.info("Chat history cleared for new embeddings")
                        st.info("✅ Embeddings created! Please refresh the page to start chatting.")
                        
                    except FileNotFoundError as fnf_error:
                        logger.error(f"File not found error: {fnf_error}")
                        st.error(fnf_error)
                    except ValueError as val_error:
                        logger.error(f"Value error: {val_error}")
                        st.error(val_error)
                    except ConnectionError as conn_error:
                        st.error(conn_error)
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

        # Column 3: Chat placeholder
        with col3:
            st.header("💬 Chat")
            st.info("🔄 Create embeddings first to enable chat")

# Contact Page
elif choice == "📧 Contact":
    st.title("📬 Contact & Support")
    st.markdown("""
    **System Information:**
    - Frontend: Streamlit (Real-time UI)
    - Backend API: FastAPI (Port 5000 for React integration)
    - Configuration: Environment-based via .env file
    - Logging: Comprehensive logging with UTF-8 support

    **Resources:**
    - **Setup Guide**: See `API_SETUP_GUIDE.md` for deployment instructions
    - **GitHub**: [Contribute on GitHub](https://github.com/nihalkshetty2002) 🛠️

    **For Issues:**
    - Check the logs in `logs/` directory
    - Verify all services are running (Ollama, Embedding API, Chroma Cloud)
    - Ensure environment variables are properly configured in `.env`

    Your feedback and contributions help make Document Buddy better! 🚀
    """)

# Footer
st.markdown("---")
st.markdown("""<div style='text-align: center'>
© 2026 Document Buddy App | Powered by Qwen2.5 + Nomic Embeddings + Chroma Cloud | AI Anytime Project 🛡️
</div>""", unsafe_allow_html=True)

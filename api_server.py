"""
FastAPI server for RAG application
Provides REST API endpoints for document upload and chat
Runs alongside the Streamlit app
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv
import tempfile
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
from logging_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document RAG API",
    description="REST API for RAG-based document chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
embeddings_manager = None
chatbot_manager = None

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    status: str = "success"

class UploadResponse(BaseModel):
    """Upload response model"""
    message: str
    status: str = "success"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    embeddings_exist: bool
    backend_ready: bool

@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup"""
    global embeddings_manager, chatbot_manager
    
    logger.info("Initializing FastAPI server...")
    
    try:
        # Initialize ChatbotManager to check for existing embeddings
        chatbot_manager = ChatbotManager()
        logger.info("ChatbotManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChatbotManager: {str(e)}")
        chatbot_manager = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns backend status and whether embeddings exist
    """
    try:
        embeddings_exist = False
        if chatbot_manager:
            embeddings_exist = chatbot_manager.has_embeddings()
        
        logger.info(f"Health check: embeddings_exist={embeddings_exist}")
        
        return HealthResponse(
            status="healthy",
            embeddings_exist=embeddings_exist,
            backend_ready=True
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            embeddings_exist=False,
            backend_ready=False
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF and create embeddings
    
    Args:
        file: PDF file to upload
        
    Returns:
        UploadResponse with success message
    """
    if not file:
        logger.warning("Upload attempted with no file")
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    temp_pdf_path = None
    
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_pdf_path = tmp_file.name
            logger.debug(f"Saved temp file: {temp_pdf_path}")
        
        # Initialize EmbeddingsManager
        embeddings_manager = EmbeddingsManager()
        
        # Create embeddings
        logger.info(f"Creating embeddings for: {file.filename}")
        result = embeddings_manager.create_embeddings(temp_pdf_path)
        
        # Reinitialize chatbot manager to load new embeddings
        global chatbot_manager
        chatbot_manager = ChatbotManager()
        
        logger.info(f"Successfully processed: {file.filename}")
        
        return UploadResponse(
            message="✅ Vector DB Successfully Created and Stored in Chroma Cloud!",
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.debug(f"Cleaned up temp file: {temp_pdf_path}")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat query using RAG
    
    Args:
        request: ChatRequest with query string
        
    Returns:
        ChatResponse with assistant response
    """
    if not request.query or not request.query.strip():
        logger.warning("Chat request with empty query")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing chat query: {request.query[:100]}...")
        
        if not chatbot_manager:
            logger.warning("ChatbotManager not initialized")
            raise HTTPException(status_code=503, detail="Chatbot service not ready")
        
        # Get response from chatbot
        response_text = chatbot_manager.get_response(request.query)
        
        logger.info("Chat response generated successfully")
        
        return ChatResponse(
            response=response_text,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document RAG API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload (POST)",
            "chat": "/chat (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server on port 5000...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )

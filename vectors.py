# vectors.py

import os
import base64
import requests
import chromadb
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from typing import List
from logging_config import get_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class CustomEmbeddings(Embeddings):
    """Custom embedding class that calls the local embedding API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.embed_url = f"{base_url}/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        logger.info(f"Embedding {len(texts)} documents using model: {self.model}")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    self.embed_url,
                    json={"input": text, "model": self.model},
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code != 200:
                    logger.error(f"Embedding API error for document {i}: {response.text}")
                    raise Exception(f"Embedding API error: {response.text}")
                embedding = response.json()["data"][0]["embedding"]
                embeddings.append(embedding)
                logger.debug(f"Successfully embedded document {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Failed to embed document {i}: {str(e)}")
                raise
        logger.info(f"Successfully embedded all {len(texts)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        logger.info("Embedding query text using model: " + self.model)
        try:
            response = requests.post(
                self.embed_url,
                json={"input": text, "model": self.model},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                logger.error(f"Embedding API error: {response.text}")
                raise Exception(f"Embedding API error: {response.text}")
            embedding = response.json()["data"][0]["embedding"]
            logger.debug("Successfully embedded query text")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise

class EmbeddingsManager:
    def __init__(
        self,
        embedding_model: str = None,
        embedding_base_url: str = None,
        chroma_api_key: str = None,
        chroma_tenant: str = None,
        chroma_database: str = None,
    ):
        """
        Initializes the EmbeddingsManager with the specified model and Chroma Cloud settings.

        Args:
            embedding_model (str): The embedding model name.
            embedding_base_url (str): The base URL for the embedding API endpoint.
            chroma_api_key (str): Chroma Cloud API key.
            chroma_tenant (str): Chroma Cloud tenant ID.
            chroma_database (str): Chroma database name.
        """
        # Read from environment variables first, then use parameters, then fall back to defaults
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL")
        
        logger.info(f"Initializing EmbeddingsManager with model: {self.embedding_model}")
        logger.info(f"Embedding API: {self.embedding_base_url}")
        
        # Set Chroma configuration from parameters or environment variables
        self.chroma_api_key = chroma_api_key or os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = chroma_tenant or os.getenv("CHROMA_TENANT")
        self.chroma_database = chroma_database or os.getenv("CHROMA_DATABASE")
        
        self.embeddings = CustomEmbeddings(
            base_url=self.embedding_base_url,
            model=self.embedding_model
        )
        logger.info("EmbeddingsManager initialized successfully")

    def create_embeddings(self, pdf_path: str):
        """
        Processes the PDF, creates embeddings, and stores them in Chroma Cloud.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            str: Success message upon completion.
        """
        logger.info(f"Starting embedding creation for PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load and preprocess the document
        logger.info(f"Loading PDF document: {pdf_path}")
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from PDF")
        
        if not docs:
            logger.error("No documents were loaded from the PDF")
            raise ValueError("No documents were loaded from the PDF.")

        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"Created {len(splits)} text chunks from documents")
        
        if not splits:
            logger.error("No text chunks were created from documents")
            raise ValueError("No text chunks were created from the documents.")

        # Create and store embeddings in Chroma Cloud
        try:
            logger.info(f"Creating embeddings for {len(splits)} chunks")
            # Extract text content from documents
            texts = [doc.page_content for doc in splits]
            metadatas = [doc.metadata for doc in splits]
            
            logger.debug(f"Extracted {len(texts)} text contents and metadata")
            
            # Create Chroma Cloud client
            logger.info(f"Connecting to Chroma Cloud")
            client = chromadb.CloudClient(
                api_key=self.chroma_api_key,
                tenant=self.chroma_tenant,
                database=self.chroma_database
            )
            logger.debug(f"Chroma Cloud client created successfully")
            
            # Create embeddings and add to Chroma Cloud directly
            logger.info(f"Creating/updating collection '{self.chroma_database}' in Chroma Cloud")
            chroma_db = Chroma.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas,
                collection_name=self.chroma_database,
                client=client
            )
            
            logger.info(f"Successfully stored {len(texts)} embeddings in Chroma Cloud collection: {self.chroma_database}")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {e}")

        logger.info("Embedding creation completed successfully")
        return "✅ Vector DB Successfully Created and Stored in Chroma Cloud!"


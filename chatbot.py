# chatbot.py

import os
import requests
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from typing import List
from logging_config import get_logger
from dotenv import load_dotenv
import streamlit as st

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
        embeddings = []
        for text in texts:
            response = requests.post(
                self.embed_url,
                json={"input": text, "model": self.model},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise Exception(f"Embedding API error: {response.text}")
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = requests.post(
            self.embed_url,
            json={"input": text, "model": self.model},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            raise Exception(f"Embedding API error: {response.text}")
        return response.json()["data"][0]["embedding"]

class ChatbotManager:
    def __init__(
        self,
        embedding_model: str = None,
        embedding_base_url: str = None,
        llm_model: str = None,
        llm_temperature: float = None,
        llm_base_url: str = None,
        chroma_api_key: str = None,
        chroma_tenant: str = None,
        chroma_database: str = None,
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            embedding_model (str): The embedding model name.
            embedding_base_url (str): The base URL for the embedding API endpoint.
            llm_model (str): The local LLM model name.
            llm_temperature (float): Temperature setting for the LLM.
            llm_base_url (str): The base URL for the LLM API endpoint.
            chroma_api_key (str): Chroma Cloud API key.
            chroma_tenant (str): Chroma Cloud tenant ID.
            chroma_database (str): Chroma database name.
        """
        # Read from environment variables first, then use parameters, then fall back to defaults
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen2.5:32b-instruct-q8_0")
        self.llm_temperature = llm_temperature or float(os.getenv("LLM_TEMPERATURE"))
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        
        logger.info(f"Initializing ChatbotManager with LLM model: {self.llm_model}")
        logger.info(f"Embedding API: {self.embedding_base_url}")
        logger.info(f"LLM API: {self.llm_base_url}")
        
        # Set Chroma configuration from parameters or environment variables
        self.chroma_api_key = chroma_api_key or os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = chroma_tenant or os.getenv("CHROMA_TENANT")
        self.chroma_database = chroma_database or os.getenv("CHROMA_DATABASE")

        # Initialize Embeddings with custom API
        logger.info("Initializing custom embeddings")
        self.embeddings = CustomEmbeddings(
            base_url=self.embedding_base_url,
            model=self.embedding_model
        )

        # Initialize Local LLM with OpenAI-compatible API
        logger.info(f"Initializing ChatOpenAI with model: {llm_model}")
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            api_key="sk-no-key-required",  # Dummy key for local API
            base_url=self.llm_base_url,
        )

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Store format function for later use
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.format_docs = format_docs
        
        # Initialize Chroma vector store (will be loaded when available)
        self.db = None
        self._load_chroma_db()

    def _load_chroma_db(self):
        """
        Connect to Chroma Cloud vector database.
        """
        logger.info("Attempting to connect to Chroma Cloud")
        if not self.chroma_api_key or not self.chroma_tenant or not self.chroma_database:
            logger.warning("Chroma connection parameters not configured. Database will be initialized when embeddings are created.")
            self.db = None
            return
        
        try:
            logger.info(f"Creating Chroma Cloud client")
            # Create Chroma Cloud client
            client = chromadb.CloudClient(
                api_key=self.chroma_api_key,
                tenant=self.chroma_tenant,
                database=self.chroma_database
            )
            logger.debug("Chroma Cloud client created successfully")
            
            # Try to connect to the collection directly (will return empty if doesn't exist)
            logger.info(f"Attempting to connect to collection '{self.chroma_database}'")
            self.db = Chroma(
                collection_name=self.chroma_database,
                embedding_function=self.embeddings,
                client=client
            )
            logger.info(f"✅ Successfully connected to Chroma Cloud collection: {self.chroma_database}")
        except Exception as e:
            logger.warning(f"Could not connect to Chroma Cloud database: {str(e)}")
            logger.debug(f"Full error: {type(e).__name__}: {e}")
            logger.info("Collection will be created when embeddings are generated.")
            self.db = None

    def has_embeddings(self) -> bool:
        """
        Check if the collection has existing embeddings.
        
        Returns:
            bool: True if collection has documents, False otherwise.
        """
        if self.db is None:
            logger.info("No database connection available")
            return False
        
        try:
            # Try to count documents in the collection
            collection = self.db._collection
            count = collection.count()
            logger.info(f"Collection has {count} documents")
            return count > 0
        except Exception as e:
            logger.warning(f"Could not check collection status: {str(e)}")
            return False

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        logger.info(f"Processing user query: {query[:100]}...")  # Log first 100 chars
        
        if self.db is None:
            logger.warning("Attempt to get response without loaded Chroma database")
            return "⚠️ No documents have been uploaded yet. Please upload a PDF and create embeddings first."

        try:
            logger.info("Retrieving relevant documents from Chroma")
            retriever = self.db.as_retriever(search_kwargs={"k": 1})
            
            logger.info("Building RAG chain")
            qa = (
                {
                    "context": retriever | self.format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("Invoking RAG chain with query")
            response = qa.invoke(query)
            logger.info(f"Successfully generated response: {response[:100]}...")  # Log first 100 chars
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."

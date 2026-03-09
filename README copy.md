# RAG Based LLM AI Chatbot 🤖

A robust, production-ready RAG (Retrieval-Augmented Generation) chatbot built with open-source technologies. Upload PDFs, generate embeddings, and interact with your documents using an intelligent AI chatbot.

**Tech Stack:** Qwen2.5 LLM • Nomic Embed Text • Chroma Cloud • FastAPI • Streamlit • LangChain

![RAG Based LLM AI Chatbot](sct.png)

**Document Buddy** is a powerful application designed to simplify document management and intelligent retrieval. Upload your PDF documents, create embeddings for efficient search, and interact with them through an intuitive chatbot interface powered by advanced language models. 🚀

## 🛠️ Features

- **📂 Upload Documents**: Easily upload and preview PDF documents with drag-and-drop support
- **🧠 Smart Embeddings**: Generate high-quality embeddings using Nomic Embed Text v1.5
- **☁️ Cloud Storage**: Store embeddings in Chroma Cloud for scalability and reliability
- **🤖 AI Chatbot**: Interact with your documents using Qwen2.5 powered RAG responses
- **⚡ Dual Interface**: Choose between Streamlit UI or React frontend with FastAPI backend
- **📊 Comprehensive Logging**: Full traceability with UTF-8 encoded logging
- **🔧 Environment Configuration**: Easy setup with .env file for different environments
- **🌟 Production Ready**: Scalable architecture with REST API endpoints

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                             │
├─────────────────────────────┬───────────────────────────────────────┤
│   Streamlit UI (Port 8501)  │   React Frontend (Port 3000)          │
│   ├─ Upload Document        │   ├─ Material-UI Components           │
│   ├─ View Chat              │   ├─ Zustand State Management         │
│   └─ Manage Embeddings      │   └─ Axios HTTP Client               │
└─────────────────────────────┴───────────────────────────────────────┘
                                      ↓
                    ┌──────────────────────────────┐
                    │   Next.js API Routes         │
                    ├──────────────────────────────┤
                    │  /api/upload   → POST        │
                    │  /api/chat     → POST        │
                    │  /api/health   → GET         │
                    └──────────────────────────────┘
                                      ↓
        ┌─────────────────────────────────────────────────────┐
        │         FastAPI Backend (Port 5000)                 │
        ├─────────────────────────────────────────────────────┤
        │  ├─ POST /upload     → PDF Processing & Embedding   │
        │  ├─ POST /chat       → RAG Query Processing         │
        │  └─ GET /health      → Health Check                 │
        └─────────────────────────────────────────────────────┘
                         ↓              ↓              ↓
        ┌────────────┬───────────┬──────────────┐
        │            │           │              │
        ↓            ↓           ↓              ↓
    ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Unstr-  │  │ LangChain│  │ ChatBot  │  │ Vectors  │
    │uctured │  │ & LCEL   │  │ Manager  │  │ Manager  │
    └────────┘  └──────────┘  └──────────┘  └──────────┘
        ↓            ↓             ↓            ↓
    ┌──────────────────────────────────────────────────┐
    │         External Services                        │
    ├──────────────────────────────────────────────────┤
    │  • Ollama (LLM)           - Port 11434          │
    │    → qwen2.5:32b-instruct-q8_0                  │
    │                                                 │
    │  • Embedding Service      - Port 3333           │
    │    → nomic-embed-text-v1.5                      │
    │                                                 │
    │  • Chroma Cloud                                 │
    │    → Vector Database (Cloud Managed)            │
    └──────────────────────────────────────────────────┘
```

### Data Flow: Upload & Chat

**1. Document Upload Flow:**
```
User Uploads PDF
     ↓
Frontend → FastAPI /upload
     ↓
UnstructuredPDFLoader (Text Extraction)
     ↓
RecursiveCharacterTextSplitter (Chunking)
     ↓
CustomEmbeddings (API Call to Port 3333)
     ↓
Chroma Cloud (Vector Storage)
     ↓
Success Response to User
```

**2. Chat Query Flow:**
```
User Types Question
     ↓
Frontend → FastAPI /chat
     ↓
CustomEmbeddings (Embed Query)
     ↓
Chroma Cloud (Similarity Search)
     ↓
LangChain RAG Chain
     ↓
ChatOpenAI (Ollama) → Port 11434
     ↓
Formatted Response to User
```

## 🖥️ Tech Stack

The Document Buddy App is built with a modern, scalable technology stack:

### Core Components

- **[Qwen2.5](https://qwenlm.github.io/)** - Advanced language model (via Ollama) for intelligent responses
- **[Nomic Embed Text v1.5](https://www.nomic.ai/)** - High-quality text embeddings for semantic search
- **[Chroma Cloud](https://www.trychroma.com/)** - Managed vector database for scalable embedding storage
- **[LangChain](https://langchain.readthedocs.io/)** - Framework for orchestrating RAG pipelines
- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** - PDF processing and text extraction

### Application Frameworks

- **[Streamlit](https://streamlit.io/)** - Interactive web interface for direct Streamlit access
- **[FastAPI](https://fastapi.tiangolo.com/)** - Production-grade REST API for backend services
- **[React/Next.js](https://nextjs.org/)** - Modern frontend for professional UI (optional)
- **[Material-UI](https://mui.com/)** - Professional component library for React frontend

### Infrastructure

- **[Ollama](https://ollama.com/)** - Local LLM serving (Qwen2.5:32b-instruct-q8_0)
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server for FastAPI
- **Docker Support** - Optional containerization for deployment

## 📁 Directory Structure

```
RAG-Based-LLM-Chatbot/
├── app.py                    # Main Streamlit application
├── api_server.py             # FastAPI backend server (Port 5000)
├── chatbot.py                # RAG chain orchestration
├── vectors.py                # Embedding and vector DB management
├── logging_config.py         # Centralized logging configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration (create this file)
├── logs/                     # Application logs directory
├── README.md                 # Documentation
└── LICENSE                   # MIT License
```

## 🚀 Getting Started

Follow these steps to set up and run the Document Buddy App.

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) running with Qwen2.5 model
- Embedding service running (Nomic Embed Text at port 3333)
- Chroma Cloud account with API credentials

### 1. Clone the Repository

```bash
git clone https://github.com/nihalkshetty2002/RAG-Based-LLM-Chatbot.git
cd RAG-Based-LLM-Chatbot
```

### 2. Create Virtual Environment

**Using venv (Windows):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Using venv (macOS/Linux):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using Conda:**
```bash
conda create --name rag-chatbot python=3.10
conda activate rag-chatbot
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Chroma Cloud Configuration
CHROMA_API_KEY=your_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=RAG

# Embedding Model Configuration
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_BASE_URL=http://your-embedding-host:3333/v1  # Change to your embedding service address

# LLM Configuration
LLM_MODEL=qwen2.5:32b-instruct-q8_0
LLM_BASE_URL=http://your-ollama-host:11434/v1  # Change to your Ollama server address
LLM_TEMPERATURE=0.3
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

**Option A: Streamlit UI (Interactive)**
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

**Option B: FastAPI Backend + React Frontend**

Terminal 1 - Start FastAPI:
```bash
python api_server.py
```

Terminal 2 - Start React Frontend (from document-RAG/ folder):
```bash
cd ../document-RAG
npm install
npm run dev
```
Access at: `http://localhost:3000`

## 🔧 Configuration

All configuration is stored in the `.env` file. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_API_KEY` | Chroma Cloud API key | - |
| `CHROMA_TENANT` | Chroma Cloud tenant ID | - |
| `CHROMA_DATABASE` | Database name in Chroma | RAG |
| `EMBEDDING_MODEL` | Embedding model name | nomic-embed-text-v1.5 |
| `EMBEDDING_BASE_URL` | Embedding service URL | http://localhost:3333/v1 |
| `LLM_MODEL` | LLM model identifier | qwen2.5:32b-instruct-q8_0 |
| `LLM_BASE_URL` | LLM service URL (Ollama) | http://localhost:11434/v1 |
| `LLM_TEMPERATURE` | LLM temperature (0-2) for response creativity | 0.3 |

For detailed configuration guide, see [ENVIRONMENT_CONFIG.md](../ENVIRONMENT_CONFIG.md)

### Temperature Settings for RAG

The `LLM_TEMPERATURE` parameter controls response creativity and randomness:

- **0.0-0.2** (Recommended for RAG) - Deterministic, factual responses. Perfect for document-based Q&A where accuracy matters most
- **0.3-0.5** - Balanced approach with minor variation while maintaining factuality
- **0.7+** - Highly creative, suitable for creative writing but increases hallucination risk in RAG

**Default: 0.3** - Optimal for RAG applications balancing factuality with natural responses

## 📡 API Endpoints

When running FastAPI backend on port 5000:

- `GET /health` - Health check and embeddings status
- `POST /upload` - Upload PDF and create embeddings
- `POST /chat` - Query the RAG system

## 🏗️ Project Structure

- **app.py** - Streamlit UI for direct interaction
- **api_server.py** - FastAPI REST API server
- **chatbot.py** - RAG chain implementation
- **vectors.py** - Embedding and vector DB operations
- **logging_config.py** - Centralized logging setup

## 🧪 Troubleshooting

### API Connection Failed
- Verify FastAPI is running: `python api_server.py`
- Check `.env` file has correct URLs
- Ensure embedding and LLM services are accessible

### Chroma Authentication Error
- Verify `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`
- Visit https://console.trychroma.com to manage credentials

### Embedding Creation Failed
- Ensure embedding service is running on configured port
- Check network connectivity to embedding service
- Review logs in `logs/` directory

### Model Not Found
- For Ollama: Run `ollama pull qwen2.5:32b-instruct-q8_0`
- Verify LLM service is running: `ollama serve`

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the Repository**: Click "Fork" at the top-right corner
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Your Changes**: Implement your feature or fix
4. **Commit with Clear Messages**:
   ```bash
   git commit -m "Add: your feature description"
   ```
5. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**: Submit PR with detailed description

## 📖 Development Workflow

1. Update dependencies: `pip install -r requirements.txt`
2. Run linting: Check code style
3. Test locally: Run both Streamlit and FastAPI
4. Check logs: Verify logging output
5. Update documentation: Keep README and configs synced

## 🔗 Resources

- [LangChain Docs](https://langchain.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Chroma Docs](https://docs.trychroma.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)

## 📚 Project Documentation

- [API Setup Guide](../API_SETUP_GUIDE.md)
- [Environment Configuration](../ENVIRONMENT_CONFIG.md)
- [Architecture Overview](../document-RAG/ARCHITECTURE.md)

## ©️ License 🪪

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## 🌟 Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

Connect with me:

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nihalkshetty2002)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/nihal-shetty)

---

**Happy coding! 🚀✨**

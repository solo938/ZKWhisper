# ZK RAG Bot - Zero-Knowledge Contextual Retrieval-Augmented Generation

# 🛡️ ZK Contextual RAG Bot

> A Production-Grade Retrieval-Augmented Generation Chatbot for Zero-Knowledge Proofs, Noir, and Tornado Cash

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-0.1+-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Overview

**ZK Contextual RAG Bot** is a production-grade Retrieval-Augmented Generation (RAG) chatbot that combines cutting-edge LLM technology with vector-based document retrieval. It provides instant, accurate answers to questions about Zero-Knowledge Proofs, Noir programming language, and Tornado Cash privacy protocols.

The bot uses **LangChain Expression Language (LCEL)** with modern imports and **ChromaDB** for efficient semantic search, making it ideal for knowledge-intensive applications.

## ✨ Key Features

- **🎯 Intelligent RAG Pipeline**: Combines LLM reasoning with document retrieval for accurate, contextual answers
- **⚡ Production-Ready LCEL**: Modern LangChain Expression Language with simplified, stable imports
- **🔐 Privacy-Focused Content**: Specialized knowledge base on ZKPs, Noir, and Tornado Cash
- **💾 Vector Database**: ChromaDB for fast semantic similarity search
- **💬 Conversational Memory**: Maintains chat history for coherent multi-turn conversations
- **🌐 Cloud-Ready**: Easily deployable on Google Colab with ngrok tunneling
- **🎨 User-Friendly UI**: Clean Streamlit interface with real-time feedback
- **📊 Efficient Retrieval**: Top-3 document retrieval with configurable parameters

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              User Query (Streamlit UI)               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Query Processing & Context │
        │  (with Chat History)        │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Generate Embeddings       │
        │  (OpenAI Embeddings)       │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Vector Similarity Search  │
        │  (ChromaDB Retrieval)      │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  LCEL Chain Execution      │
        │  (RunnablePassthrough)     │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Format Prompt with Context│
        │  + Chat History            │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  GPT-4 Response Generation │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Display Response (UI)     │
        │  Store in Memory           │
        └────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key
- ngrok token (for cloud deployment)
- Google Colab account (optional, for cloud hosting)

### Installation (Local)

```bash
# Clone the repository
git clone https://github.com/solo938/ZKWhisper.git
cd ZKWhisper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run app.py
```

### Installation (Google Colab)

#### Cell 1: Install Dependencies
```python
!pip install streamlit langchain langchain-community langchain-openai chromadb openai tiktoken pyngrok pysqlite3-binary -q
```

#### Cell 2: Create Knowledge Base
```python
knowledge_base_content = """
# Zero-Knowledge Proofs, Noir, and Tornado Cash Knowledge Base
# [Your knowledge base content here]
"""

with open("knowledge_base.md", "w") as f:
    f.write(knowledge_base_content)

print("✅ Knowledge base created successfully!")
```

#### Cell 3: Run the App
```python
!streamlit run app.py &
```

#### Cell 4: Setup ngrok Tunnel
```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
ngrok.kill()

public_url = ngrok.connect(8501)
print(f"🚀 View your RAG Bot here: {public_url}")

!streamlit run app.py &
```


## 🔧 Configuration

### Environment Variables (.env)

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here

# Database Configuration
CHROMA_DB_PATH=./chroma_db

# Application Settings
MODEL_NAME=gpt-4
TEMPERATURE=0
MAX_TOKENS=2048

# Retrieval Settings
TOP_K_RESULTS=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuration File (config.py)

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Database
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # LLM Settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
    
    # Retrieval Settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
```

## 💻 Core Components

### 1. Document Loading & Processing

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("knowledge_base.md")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = splitter.split_documents(docs)
```

### 2. Vector Embeddings & Storage

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    persist_directory="./chroma_db"
)
vectorstore.persist()
```

### 3. Modern LCEL Chain (Fixed Imports)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.get_relevant_documents(x["input"]))
    )
    | qa_prompt
    | llm
)
```

### 4. Conversational Memory

```python
from langchain_core.messages import HumanMessage, AIMessage

formatted_chat_history = []
for msg in st.session_state.messages[:-1]:
    if msg["role"] == "user":
        formatted_chat_history.append(HumanMessage(content=msg["content"]))
    elif msg["role"] == "assistant":
        formatted_chat_history.append(AIMessage(content=msg["content"]))
```

## 🐍 Usage Examples

### Basic Query

```python
# User asks a question
question = "What are Zero-Knowledge Proofs?"

# The bot retrieves relevant documents and generates a response
response = rag_chain.invoke({
    "input": question,
    "chat_history": chat_history
})
```

### Multi-turn Conversation

```python
# Question 1
user_input_1 = "Explain Noir programming language"
response_1 = rag_chain.invoke({
    "input": user_input_1,
    "chat_history": []
})

# Question 2 (with context from previous)
user_input_2 = "How is it used for ZKPs?"
response_2 = rag_chain.invoke({
    "input": user_input_2,
    "chat_history": [
        HumanMessage(content=user_input_1),
        AIMessage(content=response_1.content)
    ]
})
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rag.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run integration tests
pytest tests/test_integration.py -v
```

## 🌐 Deployment Options

### Option 1: Google Colab (Free, Recommended for Getting Started)
- Uses ngrok for public tunneling
- No infrastructure setup needed
- Free tier available
- See Colab section above

### Option 2: Streamlit Cloud (Free Tier Available)
```bash
# Push to GitHub
git push origin main

# Connect repo to Streamlit Cloud
# Visit: https://share.streamlit.io
```

### Option 3: Docker (Production)
```bash
# Build Docker image
docker build -t zkwhisper:latest .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  zkwhisper:latest
```

### Option 4: AWS/GCP/Azure
Deploy via cloud provider's container or serverless services.

## 📊 Performance & Optimization

| Metric | Value | Notes |
|--------|-------|-------|
| Document Load Time | ~500ms | Cached after first run |
| Embedding Generation | ~1-2s | Per query |
| Retrieval Time | ~100ms | ChromaDB similarity search |
| LLM Response Time | ~5-15s | Depends on GPT-4 load |
| Total Response Time | ~8-20s | End-to-end |

### Optimization Tips

- **Cache Results**: Use `@st.cache_resource` for expensive operations
- **Reduce Chunk Size**: Smaller chunks = faster retrieval
- **Limit Top-K**: Reduce `TOP_K_RESULTS` from 3 to 1-2
- **Use gpt-3.5-turbo**: Faster & cheaper than gpt-4
- **Batch Requests**: Group multiple queries together

## 🔧 Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'langchain.chains'`
**Solution**: This project uses modern LCEL with `RunnablePassthrough`. The old chain imports don't work in newer LangChain versions. Ensure you're using the latest code.

### ChromaDB Error or SQLite Issues (Colab)
**Solution**: The `pysqlite3` fix at the top of `app.py` resolves this:
```python
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### API Key Not Recognized
**Solution**: Verify your OpenAI API key is valid and has sufficient credits. Check `.env` file format.

### Slow Response Times
**Solution**: 
- Reduce `CHUNK_SIZE` in config
- Switch to `gpt-3.5-turbo` 
- Reduce `TOP_K_RESULTS` to 1-2
- Enable caching

### ngrok Tunnel Not Working
**Solution**: 
- Verify ngrok token is correct
- Run `ngrok.kill()` to close existing tunnels
- Wait 10-15 seconds for new tunnel to establish

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## 🚧 Roadmap

- [ ] Support for PDF document uploads
- [ ] Multi-language support
- [ ] Fine-tuned models for domain-specific QA
- [ ] Real-time collaboration features
- [ ] Advanced filtering and faceted search
- [ ] Response caching and optimization
- [ ] Web3 integration (wallet authentication)
- [ ] Mobile app (React Native)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code quality checks
black . && flake8 . && isort .

# Run tests before committing
pytest tests/
```

## 📚 Documentation

- [Setup Guide](docs/SETUP_GUIDE.md) - Detailed installation instructions
- [Architecture](docs/ARCHITECTURE.md) - Technical deep dive
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues & solutions
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - The orchestration framework
- [Streamlit](https://streamlit.io/) - The web UI framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - LLM provider
- Zero-Knowledge Proof research community

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/solo938/ZKWhisper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/solo938/ZKWhisper/discussions)
- **Twitter**: [@solo938](https://twitter.com/solo938)

## 🎯 Use Cases

✅ **Educational**: Learn about ZKPs, Noir, and privacy protocols  
✅ **Research**: Quick reference for cryptographic concepts  
✅ **Development**: Q&A assistant for builders  
✅ **Documentation**: Always-available knowledge base  
✅ **Integration**: Embed RAG capabilities in your apps  

---


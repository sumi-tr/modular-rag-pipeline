### Modular RAG Pipeline with Local Lightweight LLM (phi3:mini)

This repository implements a **modular, LLM-agnostic Retrieval-Augmented Generation (RAG) pipeline** designed to work under **real-world resource constraints** such as low system memory.

The complete RAG pipeline was successfully tested using a **local lightweight LLM (`phi3:mini`) via Ollama**, making it suitable for low-RAM systems.
---
### Local LLM Setup (Optional)

This project supports local LLM-based generation using Ollama.

### Install Ollama
Download and install Ollama from:
https://ollama.com

### Verify installation:

```bash

ollama --version

ollama pull phi3:mini

---

### How to Run the Project

### Add Your Documents

Place one or more PDF files inside the `data/` directory before building the vector store.

Example:
data/
├── document1.pdf
├── document2.pdf

---

### Step 1: Build the Vector Store (One-time)
# Run this step whenever documents or embedding settings change.

python vector_store.py

# Step: 2

python rag_chat.py

---
## Project Motivation

Large documents such as PDFs, reports, and research papers are difficult to query efficiently.  
RAG systems address this by retrieving relevant document segments and using them as context for generation.

This project focuses on:
- retrieval quality
- modular system design
- LLM-agnostic architecture
- practical execution on low-resource machines

---

## What This Project Implements

### 1. Document Ingestion
- Loads multiple PDF documents from a directory
- Converts them into structured `Document` objects (text + metadata)

**File:** `load_docs.py`

---

### 2. Chunking Strategy
- Uses overlapping semantic chunks
- Preserves document metadata (file name, page number)

**File:** `chunk_docs.py`

Why this matters:
- Improves semantic retrieval
- Prevents context fragmentation
- Enables traceability

---

### 3. Semantic Embeddings
- Uses HuggingFace sentence-transformers
- Model: `all-MiniLM-L6-v2`
- Normalized embeddings for cosine similarity

**File:** `embeddings.py`

Embeddings were validated using semantic similarity tests.

---

### 4. Vector Store (FAISS)
- Builds a FAISS index from real document chunks
- Stores:
  - embedding vectors
  - chunk text
  - metadata
- Saves index to disk for reuse

**File:** `vector_store.py`

Generated artifacts:
faiss_index/
├── index.faiss
└── index.pkl

---

### 5. Semantic Retrieval (Core of RAG)
- Embeds user queries
- Retrieves top-k relevant chunks using FAISS
- Enables inspection of retrieved context

**File:** `retrieval.py`

This step demonstrates that **retrieval quality, not generation, is the foundation of RAG**.

---

## 6. Generation (LLM) – Design Status

**File** `rag_chat.py`

The system is completed with ollam phi3:mini

The system is intentionally designed so that **retrieval and generation are decoupled**.

---

## Project Structure
modular-rag-pipeline/
│
├── data/ # Sample PDF documents
├── load_docs.py
├── chunk_docs.py
├── embeddings.py
├── vector_store.py
├── retrieval.py
└── rag_chat.py 

├── faiss_index/ # Ignored in git
├── README.md
├── requirements.txt
└── .gitignore



Author: Sumathy T. 
Built with Python, LangChain, FAISS, HuggingFace  
Focus: Retrieval quality & system-aware RAG design

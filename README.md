# 📜 Sanskrit RAG System

A modern, **CPU-optimized** Retrieval-Augmented Generation (RAG) system for Sanskrit documents built with Streamlit and LangChain.

## 🚀 Quick Start

### Step 1: Clone the Repository
```bash
git clone https://github.com/rudrakshseth01/RAG_Sanskrit_Rudraksh_seth1.git
cd RAG_Sanskrit_Rudraksh_seth
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment
**On Windows:**
```bash
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
streamlit run test_final.py
```

The app will open at `http://localhost:8501`

### Step 6: Upload Documents & Ask Questions
- Upload Sanskrit documents (DOCX, PDF, TXT)
- Click "Process and Index Documents"
- Ask questions in Sanskrit (Devanagari) or English
- Get context-based Sanskrit answers

**Note:** First run will download ~ 4-5GB of models (one-time only)

---

## 🎯 Project Overview

### What It Does
- **Document Indexing**: Extracts and chunks DOCX, PDF, and TXT files
- **Semantic Search**: Uses FAISS vector store for fast similarity matching
- **Sanskrit Generation**: Produces answers in pure Sanskrit using LLMs
- **CPU-First Design**: Optimized for CPU-only inference (no GPU required)

### 🔧 Key Tech Achievements

#### 1. **ChatPromptTemplate with Sanskrit Rules**
- Uses LangChain's `ChatPromptTemplate` for structured prompts
- Enforces 7 Sanskrit-specific rules:
  - Answers grounded in provided context
  - No explanation or repetition
  - Single sentence responses
  - Pure Sanskrit output only
  - Devanagari script compliance
  - Fallback message for unanswered queries
  - Strict language enforcement (no English)

#### 2. **CPU-Optimized Architecture**
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
  - Lightweight multilingual model
  - 50+ language support including Sanskrit
  - ~420MB total size
- **Vector Search**: FAISS IndexFlatL2 on CPU
  - Exact L2 distance matching
  - No approximation (100% accuracy)
  - Supports dynamic indexing
- **Generation**: Sarvam-1 LLM
  - Sanskrit-first model
  - CPU-compatible inference
  - 50-token max generation

#### 3. **Sanskrit-Aware Text Processing**
- `RecursiveCharacterTextSplitter` with Sanskrit separators:
  - Double danda (`॥`) - verse boundaries
  - Single danda (`।`) - phrase boundaries
  - Paragraph breaks (`\n\n`)
  - Preserves semantic structure

#### 4. **Multi-Script Query Support**
- `detect_query_type()`: Identifies Devanagari vs Latin vs English
- `is_valid_query()`: Flexible validation for multiple input formats
- Supports mixed-script queries

---

## 📦 Architecture

```
User Query (Sanskrit/English)
         ↓
   Query Validation
         ↓
  Multilingual Embeddings
         ↓
  FAISS Vector Search (Top-K)
         ↓
  Context Retrieval
         ↓
  ChatPromptTemplate Formatting
         ↓
  Sarvam-1 Generation (CPU)
         ↓
  Sanskrit Answer (Devanagari)
```

---

## ⚙️ Configuration

Default settings in `test_final.py`:
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 150 characters
- **Retrieved Documents**: Top 4 matches
- **Max Generation**: 50 tokens
- **Embeddings**: Multilingual MiniLM


## 🌟 Features

✅ Multi-format document support (PDF, DOCX, TXT)  
✅ Real-time document indexing  
✅ Semantic similarity search  
✅ Pure Sanskrit output generation  
✅ CPU-optimized inference  
✅ Chat-style prompting with LangChain  
✅ Persistent FAISS indexes  
✅ Flexible query validation  

---

---

## 📊 Performance

- **First Run**: ~2-3 minutes (model downloads)
- **Document Processing**: ~1 second per page
- **Query Response**: 10-20 seconds
- **Memory Usage**: ~2-3GB
- **CPU**: Multi-core optimized

---


## 📝 Notes

- All generation happens in **pure Sanskrit (Devanagari script)**
- System uses **CPU only** (no CUDA required)
- Embeddings support 50+ languages natively

---

**Built with ❤️ for Sanskrit Language Preservation**

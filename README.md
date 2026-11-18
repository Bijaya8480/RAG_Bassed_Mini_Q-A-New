# RAG-Based Mini Q&A System

A complete, production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF/TXT) and ask natural-language questions with answers grounded in the document content.

## Problem Statement

Traditional LLMs hallucinate when asked about specific documents or proprietary information. This system solves that by:
1. **Retrieving** relevant document chunks using semantic search
2. **Generating** answers strictly from retrieved context
3. **Preventing** hallucinations through strict prompt engineering and similarity thresholds

## Features

- 📄 **Document Upload**: Support for PDF and TXT files
- ✂️ **Smart Chunking**: Overlapping word-based chunks (configurable size/overlap)
- 🧠 **Local Embeddings**: Free Sentence Transformers (all-MiniLM-L6-v2) - no API costs
- 🔍 **Vector Search**: FAISS with cosine similarity (inner product on normalized vectors)
- 🤖 **LLM Integration**: Optional OpenAI (gpt-4o-mini) with extractive fallback
- 🛡️ **Anti-Hallucination**: Strict system prompt + similarity threshold
- 📚 **Source References**: Every answer includes source chunks with similarity scores
- 🖥️ **Streamlit UI**: Clean, beginner-friendly interface

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Document Processing | PyPDF |
| Text Chunking | Custom Python splitter |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS (IndexFlatIP) |
| LLM | OpenAI API (gpt-4o-mini) - optional |
| Config | python-dotenv |
| Testing | Pytest |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Upload    │────▶│  Text Split  │────▶│  Embeddings    │
│  (PDF/TXT)  │     │  (300 words, │     │  (384-dim)     │
└─────────────┘     │   50 overlap)│     └───────┬────────┘
                    └──────────────┘             │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Answer    │◀────│  LLM/Extract │◀────│  FAISS Index   │
│  + Sources  │     │  ive Fallback│     │  (Cosine Sim)  │
└─────────────┘     └──────────────┘     └───────┬────────┘
                    ▲                             │
                    │              ┌──────────────┘
                    └──────────────┤
                                   ▼
                            ┌──────────────┐
                            │   Query      │
                            │  Embedding   │
                            └──────────────┘
```

## Folder Structure

```
mini-rag-qa/
├── app.py                 # Streamlit UI
├── document_loader.py     # PDF/TXT loading
├── text_splitter.py       # Word-based chunking
├── embeddings.py          # Sentence Transformers
├── vector_store.py        # FAISS wrapper
├── rag_pipeline.py        # Complete RAG pipeline
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (local)
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
├── README.md              # This file
│
├── data/
│   └── company_policy.txt # Sample document
│
└── tests/
    └── test_rag_pipeline.py # Unit & integration tests
```

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd mini-rag-qa
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env to add your OpenAI API key (optional)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (optional) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CHUNK_SIZE` | `300` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `3` | Chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score |

## Running the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload Document**: Click "Browse files" and select a PDF or TXT file
2. **Process Document**: Click "Process Document" - wait for indexing to complete
3. **Ask Questions**: Type your question and click "Get Answer"
4. **View Results**: See the answer with source chunk references and similarity scores

## Sample Questions (using company_policy.txt)

| Question | Expected Answer |
|----------|-----------------|
| "How many days of annual leave?" | "20 days per calendar year" |
| "What are the working hours?" | "Monday to Friday, 9:30 AM to 6:00 PM" |
| "Can I work remotely?" | "Up to 2 days per week with manager approval" |
| "How to report lost equipment?" | "Report to IT immediately" |
| "Travel expense deadline?" | "Within 15 calendar days of trip completion" |

## Testing

```bash
# Run all tests
python -m pytest tests/test_rag_pipeline.py -v

# Run specific test class
python -m pytest tests/test_rag_pipeline.py::TestDocumentLoader -v
python -m pytest tests/test_rag_pipeline.py::TestTextSplitter -v
python -m pytest tests/test_rag_pipeline.py::TestEmbeddings -v
python -m pytest tests/test_rag_pipeline.py::TestVectorStore -v
python -m pytest tests/test_rag_pipeline.py::TestRAGPipeline -v
python -m pytest tests/test_rag_pipeline.py::TestIntegration -v
```

> **Note**: First test run downloads the embedding model (~90MB). Subsequent runs are faster.

## GitHub Publishing

```bash
git init
git add .
git commit -m "Initial commit: RAG Mini Q&A System"
git branch -M main
git remote add origin https://github.com/yourusername/mini-rag-qa.git
git push -u origin main
```

> **Important**: Never commit `.env` with real API keys! Use `.env.example` as template.

## Anti-Hallucination Design

### System Prompt
```
You are a document question-answering assistant.
Answer using ONLY the supplied document context.
Never use external knowledge.
Do not invent, assume, or fabricate information.
If the answer is not available in the supplied context, respond:
'I could not find this information in the provided document.'
Keep answers concise and provide source chunk references.
```

### Similarity Threshold
- Default: 0.3 (configurable via `SIMILARITY_THRESHOLD`)
- Chunks below threshold are filtered out
- If no chunks pass threshold → "I could not find this information..."

### Limitations
- Prompt instructions and thresholds **reduce but cannot mathematically guarantee** zero hallucinations
- LLM may still misinterpret context
- Extractive fallback is more conservative but less fluent

## Future Improvements

- [ ] Persistent FAISS index (save/load)
- [ ] Multi-document support
- [ ] Hybrid search (BM25 + semantic)
- [ ] Streaming responses
- [ ] Conversation history
- [ ] Evaluation metrics (RAGAS)
- [ ] Docker deployment
- [ ] Authentication/authorization

## License

MIT License - feel free to use for learning and production.

## Acknowledgments

- Sentence Transformers by UKPLab
- FAISS by Facebook Research
- Streamlit for the amazing UI framework
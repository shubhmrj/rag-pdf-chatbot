---
title: Medical Research Assistant
emoji: 🩺
sdk: streamlit
app_file: app.py
---

# Medical Research Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Integrated-4B0082)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21F)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A67D8)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/shubhmrj/rag-pdf-chatbot?style=social)](https://github.com/shubhmrj/rag-pdf-chatbot)
[![GitHub forks](https://img.shields.io/github/forks/shubhmrj/rag-pdf-chatbot?style=social)](https://github.com/shubhmrj/rag-pdf-chatbot)

A modern Retrieval-Augmented Generation (RAG) application for chatting with medical research PDFs using a Streamlit frontend, FastAPI backend, Groq LLM, and Chroma vector search.

## Overview

This project allows users to upload one or more PDF documents, extract their text, split it into meaningful chunks, generate embeddings, store them in a local vector database, and then ask natural-language questions grounded in the uploaded content. It is designed for research and document exploration, especially in the medical domain where accurate source-backed answers matter.

## Features

- PDF upload through a user-friendly Streamlit interface
- PDF parsing and text extraction from uploaded documents
- Text cleaning and chunking for better retrieval quality
- Embedding generation using Hugging Face sentence-transformers
- Persistent vector storage using ChromaDB
- Semantic retrieval for relevant document chunks
- LLM-based answer generation with Groq
- Source citations with document and page references
- REST API endpoints for indexing and chatting
- Configurable chunk size, overlap, and retrieval depth
- Graceful error handling for missing API keys, empty uploads, and failed parsing
- Persistent local document index for repeat use

## Architecture

The application follows a standard RAG workflow:

```text
User
  │
  ▼
Upload PDF(s)
  │
  ▼
PDF Loader / Extractor
  │
  ▼
Text Cleaner + Chunker
  │
  ▼
Embedding Model
  │
  ▼
Chroma Vector Database
  │
  ▼
Retriever
  │
  ▼
Groq LLM
  │
  ▼
Answer with Source Citations
```

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq (default: llama-3.1-8b-instant) |
| Embedding Model | Hugging Face sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Document Parsing | PyPDF / PyPDFLoader |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| API Layer | FastAPI + Uvicorn |
| Environment Management | python-dotenv |
| Packaging | pip / requirements.txt |

## Folder Structure

```text
rag-pdf-chatbot/
├── api.py
├── app.py
├── requirements.txt
├── Readme.md
├── .env
├── data/
├── chroma_db/
└── rag/
    ├── __init__.py
    ├── config.py
    ├── embeddings.py
    ├── pdfs.py
    ├── service.py
    └── vector_store.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shubhmrj/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_SEARCH_K=4
```

## Running the Application

### Start the backend API

```bash
uvicorn api:app --reload
```

The API will be available at:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/status

### Start the Streamlit frontend

```bash
streamlit run app.py
```

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| GROQ_API_KEY | API key for Groq access | None |
| GROQ_MODEL | Groq model identifier | llama-3.1-8b-instant |
| EMBEDDING_MODEL | Embedding model for semantic search | all-MiniLM-L6-v2 |
| RAG_CHUNK_SIZE | Size of each text chunk | 500 |
| RAG_CHUNK_OVERLAP | Overlap between adjacent chunks | 50 |
| RAG_SEARCH_K | Number of retrieved chunks for each query | 4 |

## Usage

1. Launch the Streamlit app.
2. Upload one or more PDF files.
3. Click the index button to extract and store the content.
4. Ask questions about the uploaded documents.
5. Review the answer along with source citations.

You can also use the REST API directly for automation or integration into other systems.

## API Endpoints

| Method | Route | Description | Example Request | Example Response |
| --- | --- | --- | --- | --- |
| GET | /status | Checks whether the document index is ready | `curl http://127.0.0.1:8000/status` | `{ "ready": true }` |
| POST | /index | Uploads PDF files and builds the index | `curl -X POST -F "files=@paper.pdf" http://127.0.0.1:8000/index` | `{ "ok": true, "message": "Indexed 12 chunks from 1 PDF(s)." }` |
| POST | /chat | Sends a question to the RAG pipeline | `curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"question":"What are the main findings?"}'` | `{ "ok": true, "answer": "...", "sources": ["paper.pdf p.3"] }` |

## Screenshots

> Screenshots will be added here once the UI is captured for documentation.

- Streamlit upload and indexing interface
- Chat view with source references
- FastAPI Swagger documentation

## Example Conversation

**User:** What does this research paper say about treatment efficacy?  
**Assistant:** Based on the provided context, the paper reports improved efficacy in the treatment group compared with the control group, with the strongest effect observed in the later phase of the study.

## How RAG Works

This project uses a Retrieval-Augmented Generation pipeline to answer questions from uploaded PDFs:

1. PDF files are uploaded and stored locally.
2. The text is extracted from each page.
3. The text is cleaned and split into smaller chunks.
4. Each chunk is converted into an embedding vector.
5. Those vectors are stored in ChromaDB for semantic search.
6. When a question is asked, the system retrieves the most relevant chunks.
7. The retrieved context is sent to the Groq LLM, which generates a grounded answer.
8. The response is returned with source references for transparency.

## Project Workflow

- Upload Phase: User provides one or more PDF files.
- Indexing Phase: PDFs are parsed, cleaned, chunked, and embedded.
- Storage Phase: Embeddings are persisted in ChromaDB.
- Retrieval Phase: The system searches for context relevant to the user query.
- Generation Phase: The LLM uses the retrieved context to generate a grounded response.
- Presentation Phase: The UI displays the answer and source citations.

## Performance

The implementation is designed for practical local use and includes a few lightweight optimizations:

- Cached embedding model initialization to avoid repeated loading
- Persistent vector storage so the index does not need to be rebuilt every session
- Configurable chunking and retrieval depth for tuning quality and speed
- Atomic replacement of the Chroma index to reduce corruption risk during updates

## Error Handling

The application includes validation and error handling for common failure cases:

- Missing or empty PDF uploads
- PDFs with no readable text
- Failed chunk generation
- Missing `GROQ_API_KEY`
- Failed LLM API calls or network issues
- No matching documents found during retrieval

These errors are surfaced as clear user-facing messages in both the Streamlit frontend and the FastAPI API.

## Future Improvements

Recommended enhancements for the next iteration:

- Add user authentication and multi-user support
- Support additional document formats such as DOCX and TXT
- Add streaming responses for a more interactive chat experience
- Containerize the app with Docker and Docker Compose
- Add automated tests and evaluation datasets for RAG quality
- Improve citation formatting and provenance tracking
- Deploy to a cloud platform with persistent storage

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and commit them.
4. Open a pull request with a clear description of the improvement.

Please keep changes focused, documented, and compatible with the existing architecture.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

This project makes use of excellent open-source libraries and services, including:

- Streamlit
- FastAPI
- LangChain
- Hugging Face Transformers / sentence-transformers
- ChromaDB
- Groq
- PyPDF / pypdf
- Uvicorn
- python-dotenv
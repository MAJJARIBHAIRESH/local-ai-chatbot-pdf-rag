Local AI Chatbot with PDF RAG
A locally running AI chatbot that answers questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG).
Features

Chat interface built with React and Vite
Python backend powered by FastAPI
Local language model integration using Ollama and Llama 3.2
PDF upload and text extraction
Embedding-based document retrieval using Sentence Transformers and FAISS
Question answering based on retrieved document content
Chat history and Markdown-rendered responses

Technology Stack

Frontend: React, Vite, JavaScript
Backend: Python, FastAPI
LLM: Ollama, Llama 3.2
Retrieval: Sentence Transformers, FAISS
PDF Processing: PyMuPDF

Architecture

The user uploads a PDF through the React interface.
FastAPI extracts and processes the document text.
The text is divided into chunks and converted into embeddings.
FAISS retrieves relevant chunks for a user's question.
The retrieved context is passed to the local language model.
The chatbot returns an answer grounded in the retrieved document content.

Prerequisites

Python installed
Node.js and npm installed
Ollama installed
The llama3.2:3b model available locally

Run Locally
1. Start Ollama
Make sure Ollama is running and the model is available:
ollama pull llama3.2:3b

2. Start the backend
From the project root, activate your Python virtual environment:
.\venv\Scripts\Activate.ps1

Install the backend dependencies from your project's dependency file, then start FastAPI:
python -m uvicorn main:app --app-dir . --host 127.0.0.1 --port 8001

3. Start the frontend
Open a second terminal:
cd frontend
npm install
npm run dev

Open the local URL printed by Vite in your browser.
API Endpoints

GET /health — backend health check
POST /chat — general chat
POST /pdf/upload — upload and process a PDF
POST /pdf/chat — ask questions about an uploaded PDF

Limitations

The language model runs locally and depends on available system resources.
Uploaded document processing and storage depend on the backend implementation.
Generated answers should be checked against the original document.

Future Improvements

Add automated RAG evaluation
Improve retrieval quality and chunking
Add tests for API endpoints
Improve document source citations
Add deployment documentation


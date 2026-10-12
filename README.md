Local AI Chatbot with PDF RAG
Project README / Portfolio Guide
Project Overview
A locally running AI chatbot that supports general conversation and question answering from an uploaded PDF.
The backend uses FastAPI, the language model runs through Ollama, and the frontend is built with React and
Vite. The PDF workflow extracts document text, creates embeddings, indexes vectors with FAISS, retrieves
relevant passages, and supplies those passages to the language model to produce grounded answers.
Key Features
• Local chat powered by Ollama and the Llama 3.2 model.
• Upload a PDF and ask questions about its contents.
• Retrieval-Augmented Generation (RAG) using text embeddings and FAISS similarity search.
• Answers can include page references from the source document.
• Chat history and a dark-themed React user interface.
Technology Stack
• Frontend: React, Vite, JavaScript, CSS
• Backend: Python, FastAPI, Uvicorn
• LLM runtime: Ollama with Llama 3.2 (3B)
• PDF extraction: PyMuPDF
• Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
• Vector search: FAISS
Architecture
Browser (React/Vite) → FastAPI endpoints → (a) Ollama for general chat, or (b) PDF text extraction →
chunking/embeddings → FAISS retrieval → Ollama answer using retrieved context.
Project Folder Structure
AI-Chatbot/
■■■ main.py
■■■ .gitignore
■■■ venv/ # local Python environment; do not upload
■■■ node_modules/ # local dependencies; do not upload
■■■ package.json # if present at project root
■■■ frontend/
 ■■■ src/
 ■ ■■■ App.jsx
 ■ ■■■ App.css
 ■ ■■■ main.jsx
 ■■■ package.json
 ■■■ vite.config.js
 ■■■ README.md
Run Locally on Windows 1. Start the backend from the project root in a terminal. Use the Python environment where the backend
dependencies are installed:
python -m uvicorn main:app --app-dir . --host 127.0.0.1 --port 8001
Open http://127.0.0.1:8001/docs to inspect the FastAPI API. The backend must show the PDF endpoints
/pdf/upload and /pdf/chat.
2. Start the frontend in a second terminal:
cd frontend
npm install
npm run dev
Open the local URL printed by Vite, commonly http://localhost:5173. Ensure the frontend API base URL
matches the backend port (8001 in this example).
API Endpoints
GET / — Basic backend response
GET /health — Backend health check
POST /chat — General chat
POST /pdf/upload — Upload and process a PDF
POST /pdf/chat — Ask questions about an uploaded PDF
GitHub Safety Checklist
• Commit source code, README, and a requirements.txt file.
• Do not commit venv/, node_modules/, .env files, private PDFs, API keys, or large model files.
• Check that .gitignore is located at the project root (AI-Chatbot/.gitignore), not only inside __pycache__ or
frontend.
• Keep the frontend package.json inside frontend/; run npm commands from that folder.
• Verify the repository after pushing and test the setup instructions on a clean environment if possible.
Note: Update the backend port and frontend API base URL to match your actual configuration before publishing
this guide. Only claim features and performance metrics that you have tested.

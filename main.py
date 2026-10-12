
import json
import logging
import uuid
from collections.abc import Iterator
from threading import Lock

import faiss
import numpy as np
import requests
import pymupdf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("local-chatbot")

app = FastAPI(title="Local AI Chatbot with PDF RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

MODEL_NAME = "llama3.2:3b"
KEEP_ALIVE = "30m"

MAX_HISTORY_MESSAGES = 8
MAX_OUTPUT_TOKENS = 1536
CONTEXT_SIZE = 2048

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 300

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 300
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVAL_COUNT = 4

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Uploaded documents and indexes stay in server memory.
# They are cleared when the backend restarts.
documents = {}
documents_lock = Lock()
embedding_model = None
embedding_lock = Lock()


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)


class PDFQuestion(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=4000)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_embedding_model():
    global embedding_model

    with embedding_lock:
        if embedding_model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")

    return embedding_model


def split_text(text: str):
    """Split extracted text into overlapping character chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def ollama_stream(messages: list[dict[str, str]]) -> Iterator[str]:
    """Stream response text from Ollama."""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": 0.3,
                "num_predict": MAX_OUTPUT_TOKENS,
                "num_ctx": CONTEXT_SIZE,
            },
        }

        logger.info("Starting Ollama generation.")

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as response:
            response.raise_for_status()

            finished = False

            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("Skipping invalid Ollama JSON.")
                    continue

                if data.get("error"):
                    logger.error("Ollama error: %s", data["error"])
                    yield "\n\n**Ollama error:** " + str(data["error"])
                    return

                token = (data.get("message") or {}).get("content", "")

                if token:
                    yield token

                if data.get("done") is True:
                    finished = True
                    logger.info(
                        "OLLAMA COMPLETED | reason=%s | output_tokens=%s",
                        data.get("done_reason", "unknown"),
                        data.get("eval_count", "unknown"),
                    )
                    break

            if not finished:
                logger.warning("Ollama stream ended without done=true")
                yield "\n\n[Generation ended before completion was confirmed.]"

    except requests.Timeout:
        logger.exception("Ollama timed out")
        yield "\n\n[Ollama timed out. Try a shorter question.]"

    except requests.RequestException:
        logger.exception("Could not connect to Ollama")
        yield "\n\n[Cannot connect to Ollama. Check that it is running.]"

    except Exception:
        logger.exception("Unexpected generation error")
        yield "\n\n[An unexpected generation error occurred.]"


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Local AI Chatbot with PDF RAG is running!"
    }


@app.get("/health")
def health():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()

        model_names = [
            item.get("name", "")
            for item in response.json().get("models", [])
        ]

        available = any(
            name == MODEL_NAME or name.startswith(MODEL_NAME + "-")
            for name in model_names
        )

        return {
            "api": "running",
            "ollama": "connected",
            "model": MODEL_NAME,
            "model_available": available,
            "pdf_documents_loaded": len(documents),
        }

    except (requests.RequestException, ValueError):
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=503,
            detail="Ollama is not responding. Start Ollama and retry.",
        )


# --------------------------------------------------
# Existing text chat endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    messages = []

    for item in request.messages[-MAX_HISTORY_MESSAGES:]:
        content = item.content.strip()

        if item.role in ("user", "assistant") and content:
            messages.append({
                "role": item.role,
                "content": content,
            })

    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=400,
            detail="The latest message must be a user question.",
        )

    logger.info("Received normal chat request.")

    return StreamingResponse(
        ollama_stream(messages),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------
# PDF upload and indexing
# --------------------------------------------------

@app.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    try:
        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    finally:
        await file.close()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The PDF is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF is too large. Maximum size is 15 MB.",
        )

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.page_count > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds the {MAX_PDF_PAGES}-page limit.",
                )

            page_texts = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    page_texts.append({
                        "page": page_number,
                        "text": text,
                    })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not read PDF.")
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Check that it is a valid PDF.",
        ) from exc

    if not page_texts:
        raise HTTPException(
            status_code=422,
            detail=(
                "No selectable text was found. This may be a scanned PDF "
                "that requires OCR."
            ),
        )

    chunks = []

    for page in page_texts:
        for chunk in split_text(page["text"]):
            chunks.append({
                "page": page["page"],
                "text": chunk,
            })

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No usable text chunks could be created from this PDF.",
        )

    try:
        model = get_embedding_model()

        vectors = model.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

    except Exception as exc:
        logger.exception("PDF embedding/index creation failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not process the PDF embeddings. "
                "Check the backend terminal."
            ),
        ) from exc

    document_id = str(uuid.uuid4())

    document_data = {
        "filename": filename,
        "pages": len(page_texts),
        "chunks": chunks,
        "index": index,
    }

    with documents_lock:
        documents[document_id] = document_data

    logger.info(
        "PDF indexed | file=%s | pages=%d | chunks=%d | id=%s",
        filename,
        len(page_texts),
        len(chunks),
        document_id,
    )

    return {
        "message": "PDF uploaded and indexed successfully.",
        "document_id": document_id,
        "filename": filename,
        "pages_with_text": len(page_texts),
        "chunks": len(chunks),
    }


# --------------------------------------------------
# Ask questions about an uploaded PDF
# --------------------------------------------------

@app.post("/pdf/chat")
def chat_with_pdf(request: PDFQuestion):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Enter a question about your PDF.",
        )

    with documents_lock:
        document = documents.get(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="PDF not found. Upload the PDF again.",
        )

    try:
        model = get_embedding_model()

        query_vector = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(query_vector, dtype=np.float32)

        result_count = min(
            RETRIEVAL_COUNT,
            len(document["chunks"]),
        )

        scores, indices = document["index"].search(
            query_vector,
            result_count,
        )

        retrieved = []

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0:
                continue

            chunk = document["chunks"][int(chunk_index)]

            retrieved.append({
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            })

    except Exception as exc:
        logger.exception("PDF retrieval failed.")
        raise HTTPException(
            status_code=500,
            detail="Could not search the PDF. Check the backend terminal.",
        ) from exc

    if not retrieved:
        raise HTTPException(
            status_code=422,
            detail="No relevant text was found for this question.",
        )

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = f"""
Answer the user's question using the PDF excerpts below.

Rules:
- Use the provided excerpts as the main source of information.
- Do not invent facts that are not supported by the excerpts.
- If the excerpts do not contain the answer, clearly say so.
- Cite relevant source pages using [Page N].
- Use simple, clear language.

PDF filename: {document['filename']}

PDF excerpts:
{context}

User question:
{question}
""".strip()

    logger.info(
        "PDF question received | file=%s | retrieved_chunks=%d",
        document["filename"],
        len(retrieved),
    )

    return StreamingResponse(
        ollama_stream([
            {
                "role": "user",
                "content": prompt,
            }
        ]),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )

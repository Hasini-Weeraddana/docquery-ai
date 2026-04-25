"""
Dialog RAG APP

FastAPI backend for document Question Answering using:
- BM25 retrieval
- Groq LLM
- PDF/TXT ingestion
"""

import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
from groq import Groq
from rank_bm25 import BM25Okapi

# Config 
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GEN_MODEL = "llama-3.3-70b-versatile" 
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4

# In-memory store
chunks:               list[str]        = []
bm25:                 Optional[BM25Okapi] = None
conversation_history: list[dict]       = []

# App 
app = FastAPI(
    title="Dialog RAG API",
    description="Context-aware document question answering using BM25 retrieval and Groq LLM",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers 
def split_text(text: str) -> list[str]:
    result, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return result


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def rebuild_bm25():
    global bm25
    if chunks:
        bm25 = BM25Okapi([tokenize(c) for c in chunks])
    else:
        bm25 = None


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    if bm25 is None or not chunks:
        return []
    scores = bm25.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices if scores[i] > 0]


class QuestionPayload(BaseModel):
    question: str
    reset_memory: bool = False


# Endpoints
@app.get("/")
def root():
    return {
        "service": "Dialog RAG APP",
        "status": "running",
        "chunks_indexed": len(chunks),
        "endpoints": ["/ingest", "/ask", "/reset", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.post("/ingest")
async def ingest(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Accepts raw text or uploaded TXT/PDF file,
    chunks content, and indexes using BM25.
    """
    global chunks

    raw_text = ""
    if file:
        content = await file.read()
        fname = (file.filename or "").lower()
        if fname.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif fname.endswith(".txt"):
            raw_text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(400, "Only .txt and .pdf files are supported.")
    elif text:
        raw_text = text
    else:
        raise HTTPException(400, "Provide a `text` field or a `file` upload.")

    if not raw_text.strip():
        raise HTTPException(400, "Document appears to be empty.")

    new_chunks = split_text(raw_text)
    chunks.extend(new_chunks)
    rebuild_bm25()

    return {
        "message": "Document ingested successfully.",
        "new_chunks": len(new_chunks),
        "total_chunks": len(chunks),
    }


@app.post("/ask")
def ask(payload: QuestionPayload):
    
    """
    Retrieve relevant chunks and generate answer using Groq LLM.
    Supports conversational memory.
    """

    global conversation_history

    if payload.reset_memory:
        conversation_history = []

    if not chunks:
        raise HTTPException(400, "No documents ingested yet. Call /ingest first.")

    ctx_chunks = retrieve(payload.question)

    if not ctx_chunks:
        ctx_chunks = chunks[:TOP_K]

    context = "\n\n---\n\n".join(ctx_chunks)

    history_str = ""
    if conversation_history:
        history_str = "\nConversation so far:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation_history[-6:]
        ) + "\n"

    prompt = f"""You are a document question-answering assistant.

Instructions:
1. Use only the provided context.
2. If the answer is not in the context, reply: I don't know
3. Do not make assumptions.
4. Keep the answer clear and concise.
{history_str}
CONTEXT:
{context}

QUESTION: {payload.question}

ANSWER:"""

    try:
        response = groq_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"LLM service error: {str(e)}"

    conversation_history.append({"role": "user",      "content": payload.question})
    conversation_history.append({"role": "assistant", "content": answer})

    return {
        "question": payload.question,
        "answer":   answer,
        "context_chunks_used": len(ctx_chunks),
    }


@app.post("/reset")
def reset():
    """Clear all ingested documents and conversation history."""
    global chunks, bm25, conversation_history
    chunks = []
    bm25   = None
    conversation_history = []
    return {"message": "All data cleared."}

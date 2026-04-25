# 📄 Dialog RAG APP — Context-Aware Document Q&A System

A lightweight **Retrieval-Augmented Generation (RAG)** web application built with **FastAPI**, **BM25**, and **Groq LLM**. Upload any **TXT** or **PDF** document and ask questions based only on the uploaded content.

The assistant answers strictly from your document context or responds with:

```text
I don't know.

---

## 🏗️ Architecture

```
User → Upload Document → Text Extraction → Chunking → BM25 Index
User → Ask Question → BM25 Retrieval → Top-K Chunks → Groq LLM → Answer
```

| Component | Technology | Purpose |
|-----------|--------|-----|
| Backend API | FastAPI | REST API with auto docs |
| Retrieval Engine | BM25Okapi | Lightweight keyword-based retrieval |
| LLM Provider | Groq API | Fast cloud inference |
| Model  | Llama 3.3 70B Versatile | Answer generation |
| PDF Parsing  | pdfplumber  | Extract text from PDF |
| Frontend  | HTML | User Interface |

---

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker & Docker Compose installed
- Get a free [Groq API key](https://console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/dialog-rag.git
cd dialog-rag
```

### 2. Set your API key
```bash
cp .env.example .env
```

**.env.example**
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```

API is now live at **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Endpoints

### `POST /ingest` — Upload a document
```bash
# Upload a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"

# Upload a TXT
curl -X POST http://localhost:8000/ingest \
  -F "file=@notes.txt"

# Paste raw text
curl -X POST http://localhost:8000/ingest \
  -F "text=The capital of France is Paris. It is known for the Eiffel Tower."
```

**Response:**
```json
{
  "message": "Document ingested successfully.",
  "new_chunks": 12,
  "total_chunks": 12
}
```

---

### `POST /ask` — Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

**Response:**
```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris.",
  "context_chunks_used": 4
}
```

If the answer is not in the document:
```json
{
  "answer": "I don't know"
}
```

---

### `POST /reset` — Clear all data
```bash
curl -X POST http://localhost:8000/reset
```

### `GET /health` — Health check
```bash
curl http://localhost:8000/health
```

---

## 🖥️ Streamlit UI (Bonus)

```bash
pip install streamlit requests
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

---

## 🧠 Prompt Strategy

The LLM is instructed with strict constraints to prevent hallucination:

```
You are a precise document assistant. Your ONLY job is to answer questions
based on the CONTEXT provided below.

Rules:
1. Answer ONLY using information found in the CONTEXT.
2. If the answer cannot be found in the CONTEXT, respond with exactly: "I don't know"
3. Do NOT use any external knowledge or make assumptions.
4. Be concise and factual.
```

**Retrieval:** The user's question is embedded and compared against all stored chunk vectors via FAISS L2 similarity search. The top-4 most relevant chunks are injected as context.

**Conversational memory:** The last 3 conversation turns are appended to the prompt, enabling follow-up questions.

---

## ☁️ Deployment Guide

### Option A: Render (Free, Recommended)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Runtime:** Docker
   - **Environment Variable:** `GROQ_API_KEY=your_groq_api_key_here`
5. Deploy — you get a public URL instantly.

### Option B: Koyeb (Free)

1. Go to [koyeb.com](https://koyeb.com) → Create App
2. Select Docker → connect your GitHub repo
3. Add env var `GROQ_API_KEY`
4. Deploy

### Option C: Google Cloud Run (Free tier)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<PROJECT_ID>/dialog-rag

# Deploy
gcloud run deploy dialog-rag \
  --image gcr.io/<PROJECT_ID>/dialog-rag \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your_groq_api_key_here \
  --memory 512Mi
```

---

## ✅ CI/CD

GitHub Actions automatically builds and pushes a Docker image to Docker Hub on every push to `main`. See `.github/workflows/ci.yml`.

Setup secrets in your GitHub repo:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## 📦 Resource Footprint

| Resource | Usage |
|----------|-------|
| RAM | Low |
| CPU | Low |
| Storage | Minimal |

Suitable for student projects and free hosting tiers.

---

## 🛠️ Local Development (without Docker)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_api_key_here
uvicorn app.main:app --reload --port 8000
```
---

## 👨‍💻 Author

**Hasini Weeraddana**
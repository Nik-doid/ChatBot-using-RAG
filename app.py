from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from transformers import pipeline
import os
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Initialize Jinja2Templates for serving HTML files
templates = Jinja2Templates(directory="templates")

# Load embedding model for retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Load Google API credentials (optional)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
google_service = None
if GOOGLE_API_KEY and GOOGLE_CSE_ID:
    google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

# Load Hugging Face model for generation
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Pydantic model for JSON API requests
class QueryRequest(BaseModel):
    query: str

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle POST requests from the HTML form
@app.post("/chat")
async def chat(query: str = Form(...)):
    # Step 1: Retrieve relevant documents using ChromaDB
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_docs = results.get("documents", [[]])[0] if results.get("documents") else []

    # Step 2: Use Google API for additional context (optional)
    google_snippets = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            google_results = google_service.cse().list(q=query, cx=GOOGLE_CSE_ID).execute()
            google_snippets = [item["snippet"] for item in google_results.get("items", [])]
        except Exception as e:
            google_snippets = [f"Google Search failed: {str(e)}"]

    # Combine retrieved docs and Google snippets
    context = " ".join(retrieved_docs + google_snippets)

    # Step 3: Generate response using Hugging Face model (RAG)
    prompt = f"Given the following context, answer the question accurately:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = generator(prompt, max_length=150, temperature=0.7)
    answer = response[0]["generated_text"].strip()

    return {"response": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

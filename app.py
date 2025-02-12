from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import time
import google
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Function to load PDFs from the 'data' folder
def load_documents():
    document_loader = PyPDFDirectoryLoader("./data")
    documents = document_loader.load()
    
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:\n{doc.page_content[:500]}...\n")  # Print first 500 chars
    
    return documents

# Function to split documents into chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  
    chunk_overlap=200,  
    length_function=len,
    is_separator_regex=False,
)

    chunks = text_splitter.split_documents(documents)
    
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:\n{chunk.page_content[:500]}...\n")  # Print first 500 chars
    
    return chunks

# Function to get embedding model
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

# Function to store documents in ChromaDB
def add_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory="./db"
    )


# Function to retrieve similar documents from ChromaDB
def retrieve_relevant_documents(query: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory="./db", embedding_function=embedding_function)
    relevant_docs = db.similarity_search(query, k=5)
    
   
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}: {doc.page_content[:500]}...\n") 
    
    return relevant_docs

# Function to generate an answer using Gemini API
def generate_answer(retrieved_docs, query):
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])

    print("\n--- 🔥 Final Context Sent to LLM 🔥 ---")
    print(context)
    print("\n--- 🔥 End of Context 🔥 ---")

    prompt = f"""
You are a helpful AI that only uses the provided context to answer questions. 
If the answer is missing from the context, do NOT say "The provided context does not have information." 
Instead, do your best to infer the answer based on what is provided. 
Do NOT use external knowledge. 

Context:
{context}

Question: {query}
Answer:
"""
    model = genai.GenerativeModel("gemini-pro")
    
    try:
        response = model.generate_content(prompt)
        time.sleep(2)
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        return "Error: API quota exceeded. Try again later."


# Function to run RAG system
def run_rag_system(query):
    retrieved_docs = retrieve_relevant_documents(query)
    return generate_answer(retrieved_docs, query)

# FastAPI Routes
@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/ask")
async def ask_question(request: Request, query: str = Form(...)):
    answer = run_rag_system(query)
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "query": query})

# Load and prepare documents
documents = load_documents()
split_chunks = split_documents(documents)
add_to_chroma(split_chunks)

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

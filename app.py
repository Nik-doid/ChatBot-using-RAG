import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import google.generativeai as genai 
import time
import google

load_dotenv()

# Load Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini model
genai.configure(api_key=google_api_key)

# Function to load documents from PDF files
def load_documents():
    document_loader = PyPDFDirectoryLoader("./data")
    return document_loader.load()

# Function to split documents into chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Function to get embedding function
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# Function to add chunks to Chroma DB
def add_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory="./db"
    )

# Function to embed query and retrieve documents from Chroma
def retrieve_relevant_documents(query: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory="./db", embedding_function=embedding_function)

    print(f"Query being passed: {query}")  # Debugging step
    relevant_docs = db.similarity_search(query, k=3)
    
    return relevant_docs

# Function to generate an answer using Google Gemini API
def generate_answer(retrieved_docs, query):
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    model = genai.GenerativeModel("gemini-pro")

    try:
        response = model.generate_content(prompt)
        time.sleep(2)  # Delay of 2 seconds before the next request
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        print("Resource exhausted. Please check your API quota.")
        return "Error: API quota exceeded. Try again later."

# Main function to run the RAG system
def run_rag_system(query):
    retrieved_docs = retrieve_relevant_documents(query)
    answer = generate_answer(retrieved_docs, query)
    return answer

# Load, split, add to Chroma, and test the RAG system
documents = load_documents()
split_chunks = split_documents(documents)
add_to_chroma(split_chunks)

# Example query
query = "how to cook pizza"
answer = run_rag_system(query)
print(answer)

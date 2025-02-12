# FastAPI Chatbot with Retrieval-Augmented Generation (RAG)

This project is a chatbot built using FastAPI, which integrates Retrieval-Augmented Generation (RAG) using ChromaDB, Google Search API, and Hugging Face transformers. The chatbot answers questions by retrieving relevant information from a set of documents and generating answers based on the retrieved content.

## Features

- **Document Loading**: Load PDF documents from a directory (`./data`) for use in the RAG system.
- **Text Splitting**: The documents are split into smaller chunks for easier processing and retrieval.
- **ChromaDB Integration**: Store document chunks in ChromaDB for fast similarity search.
- **Google Gemini API**: Generate answers based on the retrieved documents using the Gemini API.
- **FastAPI Web Interface**: Provides a web interface to ask questions and get answers.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Langchain
- Chroma
- Google Generative AI
- Langchain Ollama Embeddings
- PyPDF2 (for loading PDFs)
- dotenv (for environment variables)



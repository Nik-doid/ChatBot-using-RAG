# FastAPI RAG Chatbot

## Overview
This is a FastAPI-based chatbot that leverages Retrieval-Augmented Generation (RAG) to provide accurate responses. The system integrates ChromaDB for knowledge retrieval, Google Search API for additional context, and a Hugging Face transformer model for response generation.

## Features
- **FastAPI Framework**: Provides API endpoints for interaction.
- **Jinja2 Templates**: Serves an HTML-based chat interface.
- **ChromaDB**: Retrieves relevant knowledge from an embedded database.
- **Google Search API** (Optional): Fetches additional context from the web.
- **Hugging Face Transformers**: Generates responses using `google/flan-t5-large`.

## Installation
### Prerequisites
- Python 3.8+
- `pip` package manager

### Clone the Repository
```sh
git clone https://github.com/your-repo/fastapi-rag-chatbot.git
cd fastapi-rag-chatbot


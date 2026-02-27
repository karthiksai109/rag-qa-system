# RAG Q&A System

Document Q&A system built from scratch without LangChain. Upload PDFs, TXT, or MD files, ask questions, and get answers with source citations.

## How it works

1. Documents are split into chunks using recursive token-based chunking
2. Each chunk is embedded using OpenAI text-embedding-ada-002
3. Embeddings are indexed in a FAISS vector store for fast similarity search
4. Questions are matched against chunks using cosine similarity
5. Top matching chunks are sent to GPT-3.5-turbo for answer generation with source citations

## Tech

- Python
- OpenAI API (embeddings + chat completions)
- FAISS (vector search)
- Streamlit (UI)
- PyPDF2 (PDF parsing)
- tiktoken (token counting)

## Features

- Recursive chunking with configurable chunk size and overlap
- Batch embedding processing
- Cosine similarity search with FAISS
- Source citations in every answer (document name + chunk number)
- Supports PDF, TXT, and MD file uploads
- No LangChain, no abstractions, built from raw OpenAI SDK

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need an OpenAI API key. Enter it in the sidebar when the app loads.

## Live demo

https://rag-qa-system-karthiksai109.streamlit.app

## Built by

Karthik Ramadugu - [Portfolio](https://karthikramadugu.vercel.app) | [LinkedIn](https://linkedin.com/in/ramadugukarthik)

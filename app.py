import streamlit as st
import os
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from PyPDF2 import PdfReader
import hashlib
import json
import time

st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📄",
    layout="wide"
)

# --- chunking ---

def count_tokens(text, model="text-embedding-ada-002"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def recursive_chunk(text, max_tokens=500, overlap=50):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

# --- embedding ---

def get_embeddings(texts, client):
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

# --- search ---

def search(query, index, chunks, client, top_k=5):
    query_embedding = get_embeddings([query], client)[0]
    query_vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "score": float(score)
            })
    return results

# --- answer ---

def generate_answer(query, context_chunks, client):
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Chunk {c['chunk_id']}]\n{c['chunk']}"
        for c in context_chunks
    ])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided document context. Always cite your sources using [Source: filename, Chunk N] format. If the context doesn't contain enough information to answer, say so clearly."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based on the context above. Include source citations."
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- file reading ---

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_text_file(file):
    return file.read().decode("utf-8")

# --- UI ---

st.title("RAG Q&A System")
st.caption("Upload documents, ask questions, get cited answers. Built from scratch without LangChain.")

# sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your key stays in your browser session only.")
    
    st.divider()
    st.header("Parameters")
    chunk_size = st.slider("Chunk size (tokens)", 100, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 200, 50, 10)
    top_k = st.slider("Top K results", 1, 10, 5)
    
    st.divider()
    st.caption("Built by Karthik Ramadugu")
    st.caption("No LangChain. OpenAI + FAISS + Streamlit.")

# main area
uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT, MD)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

# process documents
if uploaded_files and api_key:
    client = OpenAI(api_key=api_key)
    
    # check if files changed
    file_hash = hashlib.md5("".join([f.name for f in uploaded_files]).encode()).hexdigest()
    
    if "file_hash" not in st.session_state or st.session_state.file_hash != file_hash or "index" not in st.session_state:
        with st.spinner("Processing documents..."):
            all_chunks = []
            
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = read_pdf(file)
                else:
                    text = read_text_file(file)
                
                chunks = recursive_chunk(text, max_tokens=chunk_size, overlap=chunk_overlap)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "source": file.name,
                        "chunk_id": i + 1
                    })
            
            st.session_state.chunks = all_chunks
            st.session_state.file_hash = file_hash
            
            # build embeddings and index
            texts = [c["text"] for c in all_chunks]
            embeddings = get_embeddings(texts, client)
            st.session_state.index = build_index(embeddings)
            
            st.success(f"Processed {len(uploaded_files)} file(s) into {len(all_chunks)} chunks.")
    
    # query
    st.divider()
    query = st.text_input("Ask a question about your documents")
    
    if query:
        with st.spinner("Searching and generating answer..."):
            results = search(query, st.session_state.index, st.session_state.chunks, client, top_k=top_k)
            answer = generate_answer(query, results, client)
        
        st.subheader("Answer")
        st.write(answer)
        
        with st.expander("Source chunks used"):
            for i, r in enumerate(results):
                st.markdown(f"**Chunk {i+1}** (Score: {r['score']:.3f}) - *{r['source']}, Chunk {r['chunk_id']}*")
                st.text(r["chunk"][:500])
                st.divider()

elif uploaded_files and not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to process documents.")
else:
    st.info("Upload PDF, TXT, or MD files to get started. Your documents are processed locally in your browser session.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunking", "Recursive")
    with col2:
        st.metric("Embeddings", "ada-002")
    with col3:
        st.metric("Vector Search", "FAISS")
    
    st.divider()
    st.subheader("How it works")
    st.markdown("""
1. Upload your documents (PDF, TXT, or MD files)
2. Documents are split into chunks using recursive token-based chunking
3. Each chunk is embedded using OpenAI's text-embedding-ada-002
4. Embeddings are indexed in a FAISS vector store for fast similarity search
5. Ask a question and get an answer with source citations
    """)
    st.caption("No LangChain. No abstractions. Built from scratch with OpenAI SDK, FAISS, and Streamlit.")

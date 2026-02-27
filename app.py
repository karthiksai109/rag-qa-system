import streamlit as st
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from PyPDF2 import PdfReader
import hashlib

st.set_page_config(page_title="RAG Q&A", layout="wide")


def chunk_text(text, max_tokens=500, overlap=50):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        end = min(i + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[i:end]))
        i += max_tokens - overlap
    return chunks


def embed(texts, client):
    out = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i+20]
        resp = client.embeddings.create(input=batch, model="text-embedding-ada-002")
        out.extend([d.embedding for d in resp.data])
    return out


def make_index(embeddings):
    dim = len(embeddings[0])
    idx = faiss.IndexFlatIP(dim)
    vecs = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    return idx


def find_relevant(query, index, chunks, client, k=5):
    qvec = np.array([embed([query], client)[0]], dtype="float32")
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < len(chunks):
            hits.append({
                "text": chunks[idx]["text"],
                "file": chunks[idx]["file"],
                "cid": chunks[idx]["cid"],
                "score": float(score)
            })
    return hits


def answer_question(query, hits, client):
    ctx = "\n\n---\n\n".join(
        [f"[{h['file']}, chunk {h['cid']}]\n{h['text']}" for h in hits]
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the user's question using only the provided context. Cite sources as [filename, chunk N]. If the context is not enough, say you don't have enough info."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}
        ],
        temperature=0.1,
        max_tokens=800
    )
    return resp.choices[0].message.content


def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return file.read().decode("utf-8")


st.title("RAG Q&A")
st.write("Upload docs, ask questions, get answers with citations. No LangChain, just raw OpenAI + FAISS.")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    chunk_size = st.slider("Chunk size (tokens)", 200, 800, 500, 50)
    chunk_overlap = st.slider("Overlap (tokens)", 0, 150, 50, 10)
    top_k = st.slider("Results to retrieve", 1, 10, 5)

files = st.file_uploader("Drop your files here", type=["pdf", "txt", "md"], accept_multiple_files=True)

if files and api_key:
    client = OpenAI(api_key=api_key)
    fhash = hashlib.md5("|".join(sorted([f.name for f in files])).encode()).hexdigest()

    if st.session_state.get("fhash") != fhash or "idx" not in st.session_state:
        with st.spinner("Chunking and embedding..."):
            all_chunks = []
            for f in files:
                text = extract_text(f)
                pieces = chunk_text(text, max_tokens=chunk_size, overlap=chunk_overlap)
                for j, p in enumerate(pieces):
                    all_chunks.append({"text": p, "file": f.name, "cid": j + 1})

            st.session_state.chunks = all_chunks
            st.session_state.fhash = fhash

            vecs = embed([c["text"] for c in all_chunks], client)
            st.session_state.idx = make_index(vecs)

        st.success(f"Done. {len(files)} file(s), {len(all_chunks)} chunks indexed.")

    st.markdown("---")
    q = st.text_input("Ask something about your docs")

    if q:
        with st.spinner("Searching..."):
            hits = find_relevant(q, st.session_state.idx, st.session_state.chunks, client, k=top_k)
            ans = answer_question(q, hits, client)

        st.markdown("### Answer")
        st.write(ans)

        with st.expander("Retrieved chunks"):
            for i, h in enumerate(hits):
                st.markdown(f"**{i+1}.** `{h['file']}` chunk {h['cid']} (score: {h['score']:.3f})")
                st.code(h["text"][:400], language=None)

elif files and not api_key:
    st.warning("Paste your OpenAI API key in the sidebar.")
else:
    st.markdown("Upload a PDF, TXT, or MD file to get started. Everything runs in your session, nothing is stored.")

"""
Advanced RAG Document Q&A System
--------------------------------
Retrieves answers from a pre-built FAISS index + metadata (produced by the
companion Colab notebook) and answers questions using Groq's
`openai/gpt-oss-120b` model, grounded strictly in the retrieved context.

Run locally:   streamlit run app.py
Deploy:        Streamlit Cloud (see README.md)
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index" / "index.faiss"
METADATA_PATH = BASE_DIR / "data" / "metadata" / "metadata.pkl"

# Must match the model used in the Colab indexing notebook exactly.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
TOP_K = 6

def category_display_label(category: str) -> str:
    if category.lower() == "all":
        return "All Categories"
    if category.lower() == "darc":
        return "DARC"
    return category.replace("_", " ").title()

# --------------------------------------------------------------------------
# Page setup + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f8fa; }
    .app-title {
        font-size: 2.1rem; font-weight: 700; color: #1a2b4c; margin-bottom: 0.1rem;
    }
    .app-subtitle { color: #5a6472; font-size: 1rem; margin-bottom: 1.5rem; }
    .answer-box {
        background: #ffffff; border: 1px solid #e3e6eb; border-radius: 10px;
        padding: 1.4rem 1.6rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-top: 0.5rem;
    }
    .source-chip {
        display: inline-block; background: #eef2ff; color: #33418f;
        border-radius: 6px; padding: 4px 10px; margin: 3px 6px 3px 0;
        font-size: 0.85rem; border: 1px solid #d7ddfb;
    }
    div.stButton > button {
        background-color: #2f4bdb; color: white; border-radius: 8px;
        padding: 0.55rem 1.4rem; font-weight: 600; border: none;
    }
    div.stButton > button:hover { background-color: #24399f; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        return None
    return faiss.read_index(str(FAISS_INDEX_PATH))


@st.cache_resource(show_spinner=False)
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def embed_query(model, query: str) -> np.ndarray:
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def search(index, metadata_payload, model, query: str, category: str,
           relevance_threshold: float = 0.0, top_k: int = TOP_K):
    """FAISS similarity search, then metadata category filtering, then a
    user-controlled minimum relevance (cosine similarity) cutoff.

    relevance_threshold is 0.0-1.0: only chunks with score >= threshold are
    kept. Since embeddings are normalized and the index is IndexFlatIP,
    `score` is the cosine similarity between the query and the chunk.
    """
    chunks_meta = metadata_payload["chunks"]
    query_vec = embed_query(model, query)

    # Over-fetch generously so category + threshold filtering still leaves
    # enough candidates to choose top_k from.
    fetch_k = min(len(chunks_meta), max(top_k * 12, 80))
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        score = float(score)
        if score < relevance_threshold:
            continue
        meta = chunks_meta[idx]
        if category != "all" and meta["category"].lower() != category.lower():
            continue
        results.append({**meta, "score": score})
        if len(results) >= top_k:
            break
    return results


def build_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Source {i} | {r['category']} / {r['file_name']}]\n{r['chunk_text']}")
    return "\n\n---\n\n".join(blocks)


def format_sources(results):
    """Deduplicate by (category, subdirectory, file_name)."""
    seen = []
    for r in results:
        key = (r["category"], r.get("subdirectory", ""), r["file_name"])
        if key not in [k for k, _ in seen]:
            seen.append((key, r))
    lines = []
    for (category, subdir, fname), _ in seen:
        path_parts = [category]
        if subdir:
            path_parts.append(subdir)
        path_parts.append(fname)
        lines.append(" → ".join(path_parts))
    return lines


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question \
using ONLY the information contained in the provided context below. \
Do not use outside knowledge and do not guess.

If the context does not contain enough information to answer the question, \
respond exactly with:
"I could not find this information in the provided documents."

Be concise, accurate, and cite facts only from the given context."""


def generate_answer(client, question: str, context: str) -> str:
    if not context.strip():
        return "I could not find this information in the provided documents."

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">📚 Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions about committees, counsellors, '
    'curriculum, and DARC documents — answers are grounded in your indexed files.</div>',
    unsafe_allow_html=True,
)

index = load_faiss_index()
metadata_payload = load_metadata()

if index is None or metadata_payload is None:
    st.error(
        "FAISS index or metadata not found. Make sure `data/faiss_index/index.faiss` "
        "and `data/metadata/metadata.pkl` (generated by the Colab notebook) are present "
        "in this repository."
    )
    st.stop()

if metadata_payload.get("embedding_model") != EMBEDDING_MODEL_NAME:
    st.warning(
        f"Metadata was built with '{metadata_payload.get('embedding_model')}', "
        f"but this app is configured for '{EMBEDDING_MODEL_NAME}'. Results may be inaccurate."
    )

with st.sidebar:
    st.header("🔎 Search Settings")
    categories_present = ["all"] + sorted(metadata_payload.get("categories", []))
    category_labels = [category_display_label(c) for c in categories_present]
    selected_label = st.selectbox("Category", category_labels, index=0)
    selected_category = categories_present[category_labels.index(selected_label)]

    relevance_threshold = st.slider(
        "Minimum relevance",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
        help=(
            "Only chunks with a cosine similarity to your question at or "
            "above this value are used. Higher = stricter/more precise "
            "(fewer, more relevant results). Lower = broader recall "
            "(more results, possibly less relevant)."
        ),
    )

    st.markdown("---")
    st.caption(f"Indexed chunks: **{len(metadata_payload['chunks'])}**")
    st.caption(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
    st.caption(f"LLM: `{GROQ_MODEL_NAME}`")

client = get_groq_client()
if client is None:
    st.warning(
        "GROQ_API_KEY not found in Streamlit Secrets. Add it under "
        "**Settings → Secrets** to enable answer generation."
    )

st.markdown("### Ask a question")
question = st.text_area(
    "Your question",
    placeholder="e.g. What are the responsibilities of the Academic Committee?",
    height=110,
    label_visibility="collapsed",
)
ask_clicked = st.button("🔍 Ask", use_container_width=False)

if ask_clicked:
    if not question.strip():
        st.error("Please enter a question.")
    elif client is None:
        st.error("Cannot generate an answer: GROQ_API_KEY is not configured.")
    else:
        with st.spinner("Searching documents..."):
            try:
                model = load_embedding_model()
                results = search(
                    index, metadata_payload, model, question, selected_category,
                    relevance_threshold=relevance_threshold,
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                results = []

        if not results:
            st.markdown(
                '<div class="answer-box">I could not find this information in the '
                'provided documents.</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"No chunks met the minimum relevance of {relevance_threshold:.2f}. "
                "Try lowering the 'Minimum relevance' slider in the sidebar."
            )
        else:
            context = build_context(results)
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(client, question, context)
                except Exception as e:
                    answer = f"Error calling Groq API: {e}"

            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("### Sources")
            source_lines = format_sources(results)
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in source_lines)
            st.markdown(chips, unsafe_allow_html=True)

            with st.expander("View retrieved chunks (debug)"):
                st.caption(f"Minimum relevance in effect: {relevance_threshold:.2f}")
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r['category']} / {r['file_name']}** — relevance: {r['score']:.3f}")
                    st.text(r["chunk_text"][:600] + ("..." if len(r["chunk_text"]) > 600 else ""))

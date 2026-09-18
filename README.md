# Document Assistant — RAG Q&A System

An enterprise-style document Q&A app. Documents are indexed **once** in Google
Colab; the deployed Streamlit app only ever reads the resulting FAISS index
and metadata — it never touches the original Google Drive files at runtime.

## Architecture

```
Google Drive Documents
        │  (Colab, one-time)
        ▼
Recursive Download → Text Extraction → Cleaning → Chunking
        │
        ▼
Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
        │
        ▼
FAISS Index (index.faiss) + Metadata (metadata.pkl)
        │  (download to local machine)
        ▼
GitHub Repository
        │
        ▼
Streamlit Cloud App (app.py)
   Category Selection → Query Embedding → FAISS Search →
   Metadata Filtering → Context → Groq `openai/gpt-oss-120b` → Answer + Sources
```

## 1. Run the Colab indexing notebook

1. Open `rag_indexing_colab.ipynb` in Google Colab.
2. Run all cells top to bottom (Runtime → Run all).
   - Cell 2 uses `gdown --folder` to recursively download your public Drive
     folder (`committees/`, `counsellors/`, `curriculum/`, `DARC/`), keeping
     the subfolder structure.
   - Cells 3–6 extract text (PDF/DOCX/PPTX/TXT), clean it, and split it into
     ~350-word overlapping chunks.
   - Cell 7 embeds every chunk with **`all-MiniLM-L6-v2`** (open-source,
     384-dim, normalized for cosine similarity).
   - Cell 8 builds a FAISS `IndexFlatIP` index.
   - Cell 9 saves `index.faiss` and `metadata.pkl`.
   - Cell 10 zips both files and triggers a browser download of
     `rag_output.zip`.
3. Unzip `rag_output.zip` on your computer. You now have:
   - `index.faiss`
   - `metadata.pkl`

**Important:** `app.py` uses the exact same embedding model
(`all-MiniLM-L6-v2`). Do not change one without the other, or query and
document vectors will no longer be comparable.

## 2. Files to upload to GitHub

Upload **only**:

```
index.faiss     →  data/faiss_index/index.faiss
metadata.pkl    →  data/metadata/metadata.pkl
```

plus the app code (`app.py`, `requirements.txt`, `README.md`).

**Do NOT upload:**
- The original `committees/`, `counsellors/`, `curriculum/`, `DARC/`
  document folders — all their content, chunk text, and source info is
  already embedded inside `metadata.pkl`, so the app never needs the
  originals at runtime.
- The Colab notebook's raw download cache (`/content/raw_docs`).
- Any API keys.

## 3. Exact repository structure

```
rag-app/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── faiss_index/
    │   └── index.faiss
    │
    └── metadata/
        └── metadata.pkl
```

- `data/faiss_index/index.faiss` — the FAISS vector index.
- `data/metadata/metadata.pkl` — a pickled dict:
  `{"embedding_model": ..., "categories": [...], "chunks": [ {chunk_id, category, subdirectory, file_name, file_path, chunk_text, ...}, ... ]}`.
  List index `i` in `chunks` corresponds exactly to FAISS vector `i`.
- No extra config files are required — `GROQ_API_KEY` lives in Streamlit
  Secrets, not in a file in the repo.

## 4. Configure Streamlit Secrets

In your Streamlit Cloud app: **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

The key is never hard-coded in `app.py` — it's read via `st.secrets`.

## 5. Deploy on Streamlit Cloud

1. Push the repository (structure above) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file to `app.py`.
4. Add `GROQ_API_KEY` in Secrets (step 4 above).
5. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

## 6. How retrieval works

1. User selects a category (`All`, `Committees`, `Counsellors`,
   `Curriculum`, `DARC`) and types a question.
2. The question is embedded with `all-MiniLM-L6-v2` (same model/preprocessing
   as indexing).
3. FAISS performs cosine-similarity search (`IndexFlatIP` on normalized
   vectors) to find the most semantically relevant chunks, over-fetching a
   larger candidate pool.
4. Results are filtered by the `category` field in metadata (skipped when
   "All Categories" is selected) down to the top-K relevant chunks.
5. The matching chunk texts are concatenated into a context block, tagged
   with their source.
6. The context + question are sent to Groq's `openai/gpt-oss-120b`, with a
   system prompt instructing it to answer **only** from the given context,
   and to reply *"I could not find this information in the provided
   documents."* when the context is insufficient.
7. The answer and a de-duplicated, human-readable list of sources
   (`Category → Subdirectory → File name`) are displayed.

## 7. Testing

- Ask a question with **All Categories** selected — it should be able to
  pull from any of the four folders.
- Switch to a single category (e.g. **Curriculum**) and ask a question only
  answerable from another category (e.g. Committees) — it should correctly
  respond that it could not find the information.
- Ask something entirely unrelated to your documents — same graceful
  fallback response should appear, not a hallucinated answer.

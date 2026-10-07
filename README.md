# RAG vs No-RAG — Complete Demo Project

The project that builds the same question-answering application in two ways:

1. **Without RAG:** question -> LLM -> answer.
2. **With RAG:** question -> embedding -> retrieve relevant company-document chunks -> LLM + context -> answer.

The implementation uses **Ollama** for the local chat and embedding models, and a small NumPy index so the mechanics of RAG are visible. It is deliberately simpler than a production stack.

## Architecture

### Without RAG
```text
Question -> LLM -> Answer
```

### With RAG
```text
Company documents -> Chunking -> Embeddings -> Local vector index
                                              ^
Question -> Embedding -> Similarity search ----|
                           |
                           v
                    Relevant chunks
                           |
                           v
                    LLM + context
                           |
                           v
                    Answer + sources
```

## Requirements

- Python 3.10+
- Ollama installed and running locally
- A chat model, e.g. `llama3.2`
- An embedding model, e.g. `nomic-embed-text`

Pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Start Ollama if needed:

```bash
ollama serve
```

## 1. Create a Python environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install Python packages

```bash
pip install -r requirements.txt
```

## 3. Configure models

Copy `.env.example` to `.env` if you want to change the defaults.

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
TOP_K=3
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

## 4. Build the RAG index

```bash
python ingest.py
```

This reads `documents/*.txt`, chunks them, creates embeddings, and writes:

```text
data/chunks.json
data/embeddings.npy
```

## 5. Run without RAG

```bash
python without_rag.py
```

Or provide the question directly:

```bash
python without_rag.py "How many days of parental leave are available?"
```

The model receives **only the question**. It does not receive the company documents.

## 6. Run with RAG

```bash
python rag.py
```

Or:

```bash
python rag.py "How many days of parental leave are available?"
```

The program prints the retrieved chunks, similarity scores, answer, and sources.

## 7. Compare both

```bash
python compare.py
```

Or:

```bash
python compare.py "What is the travel meal reimbursement limit?"
```

This sends the same question through both pipelines.

## Suggested KT demo

Run these questions:

```text
What is the parental leave policy?
How many days of parental leave are available?
What is the travel meal reimbursement limit?
How much can an employee claim for a hotel per night?
What is the health insurance deductible?
What should I do if I lose my company laptop?
How often must employees change their passwords?
```

Then ask something that is not in the documents:

```text
What is the company's policy for working from Mars?
```

This is useful for explaining grounded answers and why RAG should refuse to invent company-specific facts when the context does not contain the answer.

## What each file teaches

- `without_rag.py` — direct LLM baseline.
- `ingest.py` — indexing/ingestion stage.
- `rag.py` — retrieval + generation.
- `compare.py` — side-by-side demo.
- `rag_utils.py` — chunking, embeddings, cosine similarity, retrieval, prompt construction.
- `documents/` — sample company knowledge base.
- `evaluation_questions.json` — starter evaluation set.

## Mapping to your 9 RAG seminar topics

1. **Problem RAG solves** — compare the two pipelines.
2. **Embeddings** — `embed_text()` converts text into vectors.
3. **Vector databases** — this demo uses a local NumPy index; production can use PostgreSQL/pgvector, Qdrant, Pinecone, etc.
4. **RAG pipeline** — `ingest.py` + `rag.py`.
5. **Chunking** — `chunk_text()`.
6. **Retrieval quality** — top-k, similarity scores, chunk size/overlap.
7. **Debugging** — inspect retrieved chunks and scores before blaming generation.
8. **Evaluation** — use `evaluation_questions.json` as a starting test set.
9. **Security** — authorization and prompt-injection defenses must be added before production.

## Important production notes

This is a teaching/demo implementation, not a production security architecture. Production systems should enforce authorization before retrieval, isolate tenants, treat retrieved documents as untrusted data, defend against prompt injection, log access, evaluate systematically, track document versions/freshness, and protect credentials.

## Suggested seminar explanation

Ask: **“If I ask the LLM about our private company policy, where does it get the answer?”** Run the no-RAG version.

Then ask: **“What changes if we retrieve the relevant policy and provide it to the model?”** Run the RAG version.

The key takeaway:

```text
No RAG:  Question -> LLM -> Answer

RAG:     Question -> Retrieve evidence -> LLM + evidence -> Grounded answer
```

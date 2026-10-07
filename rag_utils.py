import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "chunks.json"

def ollama_health_check():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Start it with `ollama serve`.") from exc

def embed_text(text: str) -> List[float]:
    r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data:
        return data["embeddings"][0]
    if "embedding" in data:
        return data["embedding"]
    raise RuntimeError(f"Unexpected embedding response: {data}")

def generate(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "stream":False}, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def cosine_similarity(query_vector, matrix):
    q = np.asarray(query_vector, dtype=np.float32)
    qnorm = np.linalg.norm(q)
    if qnorm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    q = q / qnorm
    return normalize(matrix.astype(np.float32)) @ q

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            candidates = [text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end)]
            best = max(candidates)
            if best > start + int(chunk_size * 0.55):
                end = best + (2 if text[best:best+2] == "\n\n" else 1)
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

def load_documents(documents_dir: Path):
    records = []
    for path in sorted(documents_dir.glob("*.txt")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            records.append({"id":f"{path.stem}-{i}", "source":path.name, "chunk_index":i, "text":chunk})
    return records

def save_index(chunks, embeddings):
    DATA_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

def load_index() -> Tuple[List[Dict], np.ndarray]:
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("RAG index not found. Run `python ingest.py` first.")
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8")), np.load(EMBEDDINGS_FILE)

def retrieve(question: str, top_k=TOP_K):
    chunks, embeddings = load_index()
    scores = cosine_similarity(embed_text(question), embeddings)
    indices = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in indices:
        item = dict(chunks[int(idx)]); item["score"] = float(scores[int(idx)]); out.append(item)
    return out

def build_rag_prompt(question, results):
    context = "\n\n".join(f"[Source {i}: {x['source']} | chunk {x['chunk_index']}]\n{x['text']}" for i,x in enumerate(results,1))
    return f"""You are a company policy assistant.\nAnswer the user's question using ONLY the supplied context.\nIf the context does not contain enough information, say: \"I don't have enough information in the provided company documents.\"\nDo not invent company policies. Keep the answer concise and mention the source filename when useful.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n"""

def print_retrieval(results):
    print("\n--- RETRIEVED CONTEXT ---")
    for rank,x in enumerate(results,1):
        print(f"\n[{rank}] {x['source']} | chunk={x['chunk_index']} | score={x['score']:.4f}\n{x['text'][:700]}")
    print("--- END CONTEXT ---\n")

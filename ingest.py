from pathlib import Path
import numpy as np
from rag_utils import embed_text, load_documents, ollama_health_check, save_index
ROOT=Path(__file__).resolve().parent
if __name__ == "__main__":
    ollama_health_check()
    chunks=load_documents(ROOT/"documents")
    if not chunks: raise RuntimeError("No .txt files found in documents/")
    print(f"Loaded {len(chunks)} chunks.")
    vectors=[]
    for i,item in enumerate(chunks,1):
        print(f"Embedding {i}/{len(chunks)}: {item['source']}")
        vectors.append(embed_text(item["text"]))
    arr=np.asarray(vectors,dtype=np.float32)
    save_index(chunks,arr)
    print(f"\nIndex created: {len(chunks)} chunks, embedding shape {arr.shape}")

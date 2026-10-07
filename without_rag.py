import sys
from rag_utils import generate, ollama_health_check

def answer_without_rag(question):
    prompt=f"""You are a helpful assistant. You do NOT have access to the company's internal documents. If you do not know a company-specific fact, say you do not have access to it.\n\nUSER QUESTION:\n{question}"""
    return generate(prompt)

if __name__ == "__main__":
    ollama_health_check(); q=" ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    print("\n===== WITHOUT RAG =====\n" + answer_without_rag(q))

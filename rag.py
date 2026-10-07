import sys
from rag_utils import build_rag_prompt, generate, ollama_health_check, print_retrieval, retrieve

def answer_with_rag(question):
    results=retrieve(question); print_retrieval(results)
    return generate(build_rag_prompt(question,results)), results

if __name__ == "__main__":
    ollama_health_check(); q=" ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    ans,res=answer_with_rag(q)
    print("ANSWER:\n"+ans+"\n\nSOURCES:")
    for x in res: print(f"- {x['source']} (score={x['score']:.4f})")

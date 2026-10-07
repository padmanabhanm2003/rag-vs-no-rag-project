import sys
from rag import answer_with_rag
from rag_utils import ollama_health_check
from without_rag import answer_without_rag

if __name__ == "__main__":
    ollama_health_check(); q=" ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    print("="*80+"\nRAG VS NO-RAG COMPARISON\n"+"="*80+f"\n\nQUESTION:\n{q}")
    print("\n"+"-"*80+"\n1. WITHOUT RAG\n"+"-"*80); print(answer_without_rag(q))
    print("\n"+"-"*80+"\n2. WITH RAG\n"+"-"*80); ans,res=answer_with_rag(q); print(ans)
    print("\n"+"-"*80+"\nCOMPARISON\n"+"-"*80); print("No-RAG: question only.\nRAG: question + retrieved document context.\n\nRetrieved sources:")
    for x in res: print(f"- {x['source']} | similarity={x['score']:.4f}")

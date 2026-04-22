# Technical Documentation: RAG Customer Support Assistant

## 1. Introduction
**RAG**: Retrieval-Augmented Generation combines vector search + LLM for grounded answers.
**Need**: Hallucination-free support from docs.
**Use Case**: E-commerce FAQ bot with escalation.

## 2. System Architecture Explanation
See HLD diagram. Data flows: ingest offline, query online via graph.

## 3. Design Decisions
- Chunk size: 500 chars (balance context/granularity)
- Embeddings: MiniLM (fast, 384d, good for support text)
- Top-k=4: Optimal recall/precision
- Prompt: Few-shot with sources
- Local stack: No cloud costs/API keys

## 4. Workflow Explanation
**LangGraph**: Stateful graph vs chains.
- retrieve_node(state): `retriever.invoke(state['messages'][-1].content)`
- generate_node: Ollama(prompt with context)
- router_node: LLM extracts confidence
- State propagates messages/context.

## 5. Conditional Logic
Prompt detects escalation:
```
Analyze: {answer}
Confidence score (0-1): ?
Escalate? (complex topics like fraud/bulk)
```

## 6. HITL Implementation
Simulation: `input()` for human. Prod: API webhook.
Benefits: Accuracy boost. Limits: Agent availability.

## 7. Challenges & Trade-offs
- Retrieval: Relevance vs speed (hybrid search future)
- Chunking: Overlap for context continuity
- Cost: Local = free, but slower gen
- Eval: Manual QA on 20 queries

## 8. Testing Strategy
Unit:
- ingest: len(db.get() ) >0
- graph: Mock nodes

E2E:
- \"Return policy?\" → Correct chunk
- \"Bulk order\" → Escalates

## 9. Future Enhancements
- Multi-doc RAG
- User feedback → rerank
- Agent memory (chat history)
- Deploy: Docker + FastAPI
- Reranker (cross-encoder)

Export to PDF: `pandoc tech_doc.md -o tech_doc.pdf`


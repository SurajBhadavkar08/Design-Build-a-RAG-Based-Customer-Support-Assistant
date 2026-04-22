# Low-Level Design (LLD): RAG Customer Support Assistant

## 1. Module-Level Design
```
rag_customer_support/
├── src/
│   ├── ingest.py        # loader, splitter, embedder, chroma
│   ├── rag_graph.py     # StateGraph, nodes, edges
│   ├── hitl.py          # get_human_input()
│   └── __init__.py
├── app.py               # Streamlit app
├── chroma_db/           # Persistent DB (auto)
```

## 2. Data Structures
```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str  # Retrieved docs
    confidence: float  # 0-1
    needs_escalation: bool
    human_response: str  # Optional
```

Chunk format: `{"page_content": str, "metadata": {"source": str}}`

Query-response: `{"answer": str, "sources": list[str], "escalated": bool}`

## 3. Workflow Design (LangGraph)
```python
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("router", router_node)
workflow.add_node("hitl", hitl_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "router")
workflow.add_conditional_edges(
    "router",
    should_escalate,
    {"escalate": "hitl", "respond": END}
)
workflow.add_edge("hitl", END)
```

## 4. Conditional Routing Logic
Router prompt: "Rate confidence in answer (0-1). Escalate if <0.7 or complex (bulk, dispute)."
Criteria:
- Confidence < 0.7
- Keywords: 'bulk', 'wholesale', 'dispute', 'legal'

## 5. HITL Design
- Triggered by router
- CLI: input("Human agent response: ")
- In prod: Queue to agent dashboard, callback
- State update: `state["human_response"] = input`

## 6. API/Interface Design
Input: `{"messages": [("human", "query")] }`
Output: `state["messages"][-1]["content"]`
Invoke: `app.invoke(input)`

## 7. Error Handling
- No chunks: Default msg "No info found, escalating."
- LLM fail: Retry 3x or escalate
- DB error: Recreate index


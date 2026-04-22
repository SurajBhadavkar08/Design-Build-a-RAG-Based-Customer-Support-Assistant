"""LangGraph workflow for RAG with HITL routing."""
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser

persist_directory = "./chromadb"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.8})

llm = OllamaLLM(model="llama3.2")

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    confidence: float
    needs_escalation: bool

# Node 1: Retrieve
def retrieve_node(state: GraphState) -> GraphState:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# Node 2: Generate
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. Use context to answer accurately. Sources: {context}"),
    MessagesPlaceholder(variable_name="messages"),
])

def generate_node(state: GraphState) -> GraphState:
    chain = rag_prompt | llm
    response = chain.invoke({
        "context": state["context"],
        "messages": state["messages"],
    })
    return {"messages": [HumanMessage(content=response)]}

# Router Prompt
router_prompt = ChatPromptTemplate.from_template("""
Review this answer for customer query.

Answer: {answer}

Respond JSON:
{{"confidence": number 0-1.0, "needs_escalation": boolean, "reason": "brief reason"}}

Escalate if low confidence (<0.7), complex (bulk/fraud), or missing context.
""")

parser = JsonOutputParser()

def router_node(state: GraphState) -> GraphState:
    answer = state["messages"][-1].content
    chain = router_prompt | llm | parser
    result = chain.invoke({"answer": answer})
    return {
        "confidence": result["confidence"],
        "needs_escalation": result["needs_escalation"]
    }

# HITL Node
def hitl_node(state: GraphState) -> GraphState:
    print(f"\n🤖 Escalation triggered (confidence: {state['confidence']}). Reason: Complex query.")
    human_input = input("👤 Human agent response: ")
    return {"messages": [HumanMessage(content=human_input)], "needs_escalation": False}

# Router function for edges
def should_escalate(state: GraphState) -> str:
    if state["needs_escalation"]:
        return "hitl"
    return END

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("router", router_node)
workflow.add_node("hitl", hitl_node)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "router")
workflow.add_conditional_edges("router", should_escalate, {"hitl": "hitl", END: END})
workflow.add_edge("hitl", END)

app = workflow.compile()

if __name__ == "__main__":
    while True:
        query = input("Query: ")
        if query.lower() == "exit":
            break
        result = app.invoke({"messages": [HumanMessage(content=query)]})
        print("Answer:", result["messages"][-1].content)


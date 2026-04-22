"""Streamlit UI for RAG Customer Support Assistant."""
import streamlit as st
from src.rag_graph import app, GraphState

st.set_page_config(page_title="Customer Support Bot", page_icon="🤖")

st.title("🤖 RAG Customer Support Assistant")
st.markdown("**LangGraph + HITL enabled. Run `python src/ingest.py` first.**")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about shipping, returns, or test escalation with 'bulk order'..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        try:
            result = app.invoke({
                "messages": st.session_state.messages[-1:]  # Last message
            })
            response = result["messages"][-1].content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Check escalation
            if result.get("needs_escalation", False):
                st.warning("🚨 Escalated to human agent! Check terminal for input.")
        except Exception as e:
            st.error(f"Error: {e}. Ensure Ollama running.")

# Sidebar instructions
with st.sidebar:
    st.header("Quick Start")
    st.code("""
cd rag_customer_support
python src/ingest.py  # Build index
streamlit run app.py
    """, language="bash")
    st.info("Test escalation: 'I want 100 units'")
    st.info("Docs: hld.md, lld.md, tech_doc.md")
    if st.button("Convert to PDFs"):
        st.code("pandoc hld.md -o hld.pdf\\npandoc lld.md -o lld.pdf\\npandoc tech_doc.md -o tech_doc.pdf")


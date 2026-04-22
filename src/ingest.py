"""Ingest customer support FAQ into ChromaDB vector store."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setup
persist_dir = Path("./chromadb")
persist_dir.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("Loading document...")
loader = TextLoader("data/customer_support_faq.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} docs.")

print("Chunking...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir.as_posix()
)

print("Ingest complete! Vector store persisted to chromadb/")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Retriever ready for use in graph.")

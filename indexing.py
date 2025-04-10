# indexing.py
import os
import datetime
import traceback
import streamlit as st # Keep for feedback/errors
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config variables
from config import POLICY_INDEX_DIR

def create_and_save_policy_index(policy_file_path: str, index_name: str, embeddings):
    """Loads policy, chunks, creates and saves FAISS index."""
    # Use st.write for feedback during the process
    st.write(f"Processing Policy: {os.path.basename(policy_file_path)} for index '{index_name}'...")
    try:
        loader = PyPDFLoader(policy_file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"PDF Load Error: {e}")
        return False, f"PDF Load Error: {e}"

    if not documents:
        st.error(f"Could not load content from {policy_file_path}.")
        return False, f"Could not load content from {policy_file_path}."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
    policy_chunks = text_splitter.split_documents(documents)

    if not policy_chunks:
        st.error("Text splitting yielded zero chunks.")
        return False, "Text splitting yielded zero chunks."

    try:
        # st.write("Creating FAISS index...") # Less verbose
        vector_store = FAISS.from_documents(policy_chunks, embeddings)
        # st.write("FAISS index created in memory.")
    except Exception as e:
        st.error(f"FAISS Create Error: {e}")
        st.error(traceback.format_exc())
        return False, f"FAISS Create Error: {e}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_folder_name = "".join(c if c.isalnum() else "_" for c in index_name)
    index_folder_path = os.path.join(POLICY_INDEX_DIR, f"{safe_folder_name}_{timestamp}")

    try:
        vector_store.save_local(index_folder_path)
        st.write(f"FAISS index saved: {os.path.basename(index_folder_path)}") # Show relative path
        return True, index_folder_path
    except Exception as e:
        st.error(f"FAISS Save Error: {e}")
        st.error(traceback.format_exc())
        return False, f"FAISS Save Error: {e}"
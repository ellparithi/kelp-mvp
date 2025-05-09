
import streamlit as st
from datetime import datetime
import chromadb
import os
from dotenv import load_dotenv

from backend import (
    get_or_create_collection,
    add_document_to_kelp,
    retrieve_relevant_memories,
    delete_documents_from_kelp,
    delete_entire_kelp,
    kelp_kbase_reasoning,
    kelp_kawl_reasoning,
    extract_text_from_file,
)
from chat_manager import (
    list_chat_sessions,
    save_chat_session,
    load_chat_session,
    delete_chat_session,
)

import re

def sanitize_kelp_name(name):
    # Replace invalid characters with underscores and trim
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())
    return sanitized[:512] or "kelp_default"

load_dotenv()
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ---------- Page Config ----------
st.set_page_config(page_title="Kelp ", layout="wide")
st.sidebar.image("klwtwobg1.png", use_container_width=True)
st.sidebar.title("Kelp Manager")

# ---------- Sidebar: Kelp Selector ----------
def list_existing_kelps():
    try:
        collections = chroma_client.list_collections()
        print("Collections:", collections)
        return [col.name if hasattr(col, "name") else str(col) for col in collections]
    except Exception as e:
        st.error(f"Error listing Kelps: {e}")
        return []

kelp_names = list_existing_kelps()
if "active_kelp" not in st.session_state:
    st.session_state.active_kelp = None
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

selected_kelp = st.sidebar.selectbox(
    "Select a Kelp:", ["Create a New Kelp"] + kelp_names
)

if selected_kelp == "Create a New Kelp":
    new_kelp_name = st.sidebar.text_input("Enter New Kelp Name:")
    st.sidebar.caption("Use only letters, numbers, hyphens and underscores")
    if st.sidebar.button("Create Kelp"):
        if new_kelp_name:
            get_or_create_collection(new_kelp_name)
            st.session_state.active_kelp = new_kelp_name
            st.rerun()

else:
    st.session_state.active_kelp = selected_kelp

if st.session_state.active_kelp:
    st.sidebar.markdown(f"**Active Kelp:** `{st.session_state.active_kelp}`")

# ---------- Delete Kelp ----------
st.sidebar.header("Data Management")
if st.sidebar.button("Delete Current Kelp"):
    if st.session_state.active_kelp:
        success = delete_entire_kelp(st.session_state.active_kelp)
        if success:
            st.success(f"Kelp '{st.session_state.active_kelp}' deleted.")
            st.session_state.active_kelp = None
            st.session_state.chat_session_id = None
            st.session_state.chat_history = []
            st.rerun()

# ---------- Main Area ----------
st.title("Chat with your Kelp!")

if not st.session_state.active_kelp:
    st.warning("Please select or create a Kelp to begin.")
    st.stop()

# ---------- Upload Documents ----------
uploaded_files = st.file_uploader(
    f"Upload Documents for `{st.session_state.active_kelp}`",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text.strip():
            metadata = {"source": file.name}
            add_document_to_kelp(st.session_state.active_kelp, text, metadata)
    st.success(f"Uploaded {len(uploaded_files)} document(s).")

# ---------- Manage Documents ----------
try:
    collection = get_or_create_collection(st.session_state.active_kelp)
    docs = collection.get(include=["documents", "metadatas"])
    doc_ids = docs.get("ids", [])
    doc_texts = docs.get("documents", [])
    metas = docs.get("metadatas", [])

    if doc_ids:
        doc_options = [f"{i+1}. {meta.get('source_title', '')}" for i, meta in enumerate(metas)]
        doc_id_map = {opt: doc_ids[i] for i, opt in enumerate(doc_options)}
        to_delete = st.multiselect("Select documents to delete:", doc_options)
        if st.button("Delete Selected Document(s)"):
            ids = [doc_id_map[label] for label in to_delete]
            delete_documents_from_kelp(st.session_state.active_kelp, ids)
            st.success(f"Deleted {len(ids)} document(s).")
            st.rerun()
    else:
        st.info("No documents uploaded yet.")
except Exception as e:
    st.error(f"Error loading documents: {e}")

# ---------- Chat Manager ----------
st.subheader("Chats")

# Handle start new chat
if st.button("Start a New Chat"):
    st.session_state.chat_session_id = None
    st.session_state.chat_history = []
    st.session_state["prompt_input"] = ""

# Load chats
chats = list_chat_sessions(st.session_state.active_kelp)
chat_display = ["New Chat"] + chats
selected_chat = st.selectbox("📁 Select a Chat:", options=chat_display)

# If switching to saved chat
if selected_chat != "New Chat" and selected_chat != st.session_state.chat_session_id:
    st.session_state.chat_session_id = selected_chat
    st.session_state.chat_history = load_chat_session(
        st.session_state.active_kelp, selected_chat
    )

if selected_chat != "New Chat":
    if st.button("Delete Selected Chat"):
        delete_chat_session(st.session_state.active_kelp, selected_chat)
        st.session_state.chat_session_id = None
        st.session_state.chat_history = []
        st.rerun()

# ---------- Chat Logic ----------
st.subheader("Chat with your Kelp!")

reasoning_mode = st.radio(
    "Choose Reasoning Mode:", ["KBase (Strict)", "Kawl (Enhanced)"], horizontal=True
)

user_input = st.text_input("Enter your prompt:", key="prompt_input")

if st.button("Ask") and user_input:
    # Create a new session if needed
    if st.session_state.chat_session_id is None:
        chat_name = user_input.strip().replace("_", " ").strip()[:30]
        st.session_state.chat_session_id = chat_name
        st.session_state.chat_history = []
        st.rerun()

    st.session_state.chat_history.append(("user", user_input))

    result = kelp_kbase_reasoning(st.session_state.active_kelp, user_input)
    base_answer = result["answer"]
    memory_context = result["context"]

    if reasoning_mode.startswith("Kawl"):
        final_answer = kelp_kawl_reasoning(user_input, base_answer, memory_context)
    else:
        final_answer = base_answer

    st.session_state.chat_history.append(("kelp", final_answer))
    save_chat_session(
        st.session_state.active_kelp,
        st.session_state.chat_session_id,
        st.session_state.chat_history,
    )

# ---------- Display Chat ----------
for role, text in st.session_state.chat_history:
    st.markdown(f"**{'You' if role == 'user' else 'Kelp'}:** {text}")

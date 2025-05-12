
import streamlit as st
import chromadb
import os
from dotenv import load_dotenv
from backend import build_doc_corpus
from backend import store_doc_corpus_to_chroma, load_doc_corpus_from_chroma


from backend import (
    get_or_create_collection,
    delete_documents_from_kelp,
    delete_entire_kelp,
    kelp_kbase_reasoning,
    kelp_kawl_reasoning,
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

def clear_prompt_input():
    st.session_state["user_prompt"] = ""

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
st.title("Welcome!")
st.subheader("File Manager")


if not st.session_state.active_kelp:
    st.warning("Please select or create a Kelp to begin.")
    st.stop()

# ---------- Upload Documents ----------
uploaded_files = st.file_uploader(
    f"Upload Documents for `{st.session_state.active_kelp}` (Max 10 files)",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
)

# Enforce file upload limit
if uploaded_files and len(uploaded_files) > 10:
    st.warning("Please upload no more than 10 files.")
    st.stop()

if uploaded_files:
    uploaded_filenames = [f.name for f in uploaded_files]
    session_filenames = [d["filename"] for d in st.session_state.get("doc_corpus", [])]

    if uploaded_filenames != session_filenames:
        doc_corpus = build_doc_corpus(uploaded_files)
        st.session_state["doc_corpus"] = doc_corpus
        store_doc_corpus_to_chroma(st.session_state.active_kelp, doc_corpus)
        st.success(f"Loaded {len(doc_corpus)} document(s) into memory for reasoning.")

   
# ---------- Manage Documents ----------
try:
    collection = get_or_create_collection(st.session_state.active_kelp)
    docs = collection.get(include=["documents", "metadatas"])
    doc_ids = docs.get("ids", [])
    doc_texts = docs.get("documents", [])
    metas = docs.get("metadatas", [])

    if doc_ids:
        doc_options = [f"{i+1}. {meta.get('filename', 'Untitled')}" for i, meta in enumerate(metas)]
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

if st.button("Start a New Chat"):
    st.session_state.chat_session_id = "New Chat"
    st.session_state.chat_history = []
    st.session_state["user_prompt"] = ""
    st.rerun()

# Load chats
chats = list_chat_sessions(st.session_state.active_kelp)
chat_display = ["New Chat"] + chats
current_chat_id = st.session_state.get("chat_session_id")

# Safely fall back to "New Chat" if current session is not in the list
if current_chat_id not in chat_display:
    current_chat_id = "New Chat"

selected_chat = st.selectbox(
    "Select a Chat:",
    options=chat_display,
    index=chat_display.index(current_chat_id)
)

if selected_chat != st.session_state.chat_session_id:
    st.session_state.chat_session_id = selected_chat
    st.session_state.chat_history = load_chat_session(
        st.session_state.active_kelp, selected_chat
    )
    st.session_state.clear_prompt_flag = True
    st.rerun()

if selected_chat != "New Chat":
    if st.button("Delete Selected Chat"):
        delete_chat_session(st.session_state.active_kelp, selected_chat)
        st.session_state.chat_session_id = None
        st.session_state.chat_history = []
        st.rerun()

# ---------- Chat Logic ----------
st.subheader("Chat with your Kelp!")

# Initialize flags
if "clear_prompt_flag" not in st.session_state:
    st.session_state.clear_prompt_flag = False

# Actually clear input AFTER rerun
if st.session_state.clear_prompt_flag:
    st.session_state["user_prompt"] = ""
    st.session_state.clear_prompt_flag = False

reasoning_mode = st.radio(
    "Choose Reasoning Mode:", ["KBase", "Kawl (Advanced)"], horizontal=True
)

user_input = st.text_area(
    "Ask a question",
    height=150,
    key="user_prompt"
)

if "doc_corpus" not in st.session_state or not st.session_state["doc_corpus"]:
    doc_corpus = load_doc_corpus_from_chroma(st.session_state.active_kelp)
    st.session_state["doc_corpus"] = doc_corpus

if st.button("Ask") and user_input:
    # Create a new session if needed
    if st.session_state.chat_session_id is None:
        chat_name = user_input.strip().replace("_", " ").strip()[:30]
        st.session_state.chat_session_id = chat_name
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("user", user_input))

    if "doc_corpus" not in st.session_state or not st.session_state["doc_corpus"]:
        st.error("No documents uploaded or parsed. Please upload some first.")
        st.stop()

    # Run KBase and extract answer + context
    result = kelp_kbase_reasoning(user_input, st.session_state["doc_corpus"])
    base_answer = result["answer"]
    memory_context = result["context"]

    # If user selected Kawl, run Claude-based enhancement
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

    st.session_state.clear_prompt_flag = True
    st.rerun()


# ---------- Display Chat ----------
for role, text in st.session_state.chat_history:
    if role == "user":
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#000000; color:white; padding:10px 14px; border-radius:8px; max-width: 70%; margin-left:auto; margin-bottom: 10px; text-align: left;">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:  # Kelp's response
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#f9f9f9; padding:12px 16px; border-radius:10px; border: 1px solid #ddd; margin-bottom: 15px;">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )


# backend.py

import os
import io
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
from openai import OpenAI
from dotenv import load_dotenv
import csv
import docx2txt
import PyPDF2
import json
import tiktoken
import uuid
import anthropic


# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

import tiktoken

def estimate_token_count(text: str, model="gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# ========== File Reading ==========
def extract_text_from_file(file):
    file_type = file.name.split('.')[-1].lower()

    if file_type == "txt":
        return file.read().decode('utf-8', errors='ignore')

    elif file_type == "pdf":
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return ""

    elif file_type == "docx":
        try:
            return docx2txt.process(file)
        except Exception as e:
            print(f"DOCX parsing error: {e}")
            return ""

    elif file_type == "csv":
        try:
            decoded = file.read().decode('utf-8', errors='ignore')
            reader = csv.reader(io.StringIO(decoded))
            rows = [" ".join(row) for row in reader]
            return "\n".join(rows)
        except Exception as e:
            print(f"CSV parsing error: {e}")
            return ""

    else:
        return ""

def build_doc_corpus(uploaded_files: list) -> list:
    """
    Converts uploaded files into a corpus of full-document strings.
    Also adds a preview string for reranking.
    """
    corpus = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text.strip():
            preview = generate_doc_preview(text, file.name)
            corpus.append({
                "filename": file.name,
                "text": text.strip(),
                "preview": preview  # NEW FIELD
            })
    return corpus

def generate_doc_preview(text: str, filename: str) -> str:
    file_type = filename.split('.')[-1].lower()

    # Clean and trim excerpt
    clean_text = text.strip().replace('\n', ' ')
    excerpt = ' '.join(clean_text.split()[:80])  # ~80 words = safe, readable

    # General-purpose structured preview
    preview = (
        f"Filename: {filename}\n"
        f"Type: {file_type.upper()} Document\n"
        f"Excerpt:\n{excerpt}"
    )

    return preview

def store_doc_corpus_to_chroma(kelp_name: str, doc_corpus: list):
    """
    Stores full documents (not chunks) into Chroma for persistence.
    """
    collection = get_kelp_collection(kelp_name)
    for doc in doc_corpus:
        full_text = doc["text"]
        filename = doc["filename"]
        doc_id = str(uuid.uuid4())

        collection.add(
            ids=[doc_id],
            documents=[full_text],
            metadatas=[{
                "filename": filename,
                "kelp_name": kelp_name,
                "type": "corpus_doc"
            }]
        )

def get_kelp_collection(kelp_name: str):
    """
    Returns the ChromaDB collection for the given kelp name.
    """
    return chroma_client.get_or_create_collection(name=kelp_name)

def load_doc_corpus_from_chroma(kelp_name: str) -> list:
    """
    Loads all full documents for a given Kelp name.
    Returns a list of dicts: [{filename, text}]
    """
    collection = get_kelp_collection(kelp_name)
    results = collection.get(
        where={"$and": [{"kelp_name": kelp_name}, {"type": "corpus_doc"}]},
        include=["documents", "metadatas"]
    )

    doc_corpus = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        doc_corpus.append({
            "filename": metadata.get("filename", "Unknown"),
            "text": doc_text
        })
    return doc_corpus

# ========== Basic ChromaDB Operations ==========

def get_or_create_collection(collection_name):
    return chroma_client.get_or_create_collection(name=collection_name)

def delete_documents_from_kelp(collection_name, doc_ids: list):
    collection = get_or_create_collection(collection_name)
    try:
        if doc_ids:
            collection.delete(ids=doc_ids)
            return True
        return False
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False

def delete_entire_kelp(collection_name: str):
    try:
        chroma_client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"Error deleting kelp: {str(e)}")
        return False

# ========== Reasoning ==========

def split_text_to_chunks(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> list:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        token_len = len(encoding.encode(word))
        if current_length + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = token_len
        else:
            current_chunk.append(word)
            current_length += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def rerank_documents_claude(doc_corpus: list, user_prompt: str) -> list:
    """
    Use Claude 3 Sonnet to select top 1–3 relevant documents.
    """
    labeled_docs = []
    for i, doc in enumerate(doc_corpus):
        preview = doc.get("preview", doc["text"][:1500])
        labeled_docs.append(f"[{i+1}]\n{preview}")

    joined = "\n\n".join(labeled_docs)
    prompt = (
        f"{anthropic.HUMAN_PROMPT} You are a document selector. Based on the user's question, "
        f"choose the most relevant 1–3 documents. Respond ONLY with comma-separated numbers.\n\n"
        f"Question: {user_prompt}\n\nDocuments:\n{joined}"
        f"{anthropic.AI_PROMPT}"
    )

    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=800,
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    if isinstance(response.content, list):
        selected_text = "".join(str(block.text) for block in response.content if hasattr(block, "text"))
    else:
        selected_text = str(response.content)

    selected = [int(x.strip()) - 1 for x in selected_text.split(",") if x.strip().isdigit()]
    return [doc_corpus[i] for i in selected if 0 <= i < len(doc_corpus)]

def kelp_kbase_reasoning(user_prompt: str, doc_corpus: list) -> dict:
    """
    Uses Claude to rerank docs, then uses GPT-3.5 to answer.
    Returns both answer and context.
    """
    if not doc_corpus:
        return {"answer": "No documents available. Try Again/Refresh", "context": ""}

    try:
        selected_docs = rerank_documents_claude(doc_corpus, user_prompt)
        if not selected_docs:
            return {"answer": " No relevant documents found. Try again/Refresh", "context": ""}

        full_context = "\n\n".join(doc["text"] for doc in selected_docs)

        # Estimate token count
        token_limit = 14000  # leave buffer for prompt + answer
        context_tokens = estimate_token_count(full_context)

        # Truncate if too long
        if context_tokens > token_limit:
            chunks = split_text_to_chunks(full_context, max_tokens=token_limit)
            full_context = chunks[0]  # take the best-fitting one

        messages = [
            {"role": "system", "content": "You are a helpful assistant answering strictly based on the following documents."},
            {"role": "user", "content": f"DOCUMENTS:\n{full_context}\n\nQUESTION:\n{user_prompt}"}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "context": full_context}

    except Exception as e:
        print("Error in kelp_kbase_reasoning:", e)
        return {"answer": "Internal error during reasoning. Try again/Refresh", "context": ""}

def kelp_kawl_reasoning(user_input, base_answer=None, memory_context=None, mode="default") -> str:
    """
    Uses Claude 3 Haiku to rewrite KBase answers with improved clarity and structure.
    """
    try:
        prompt = (
            f"{anthropic.HUMAN_PROMPT} You are Kawl, an intelligent reasoning assistant. "
            f"You are given a user question, a base answer, and memory context. "
            f"Your job is to clarify, expand, and enhance the base answer without hallucinating. "
            f"Do not explain what you're doing or say you're enhancing it."
            f"If the base answer is good, you may elaborate with more insight or formatting. "
            f"If the question is data-heavy (e.g. numeric, csv), double-check and express the math clearly.\n\n"
            f"USER QUESTION:\n{user_input}\n\n"
            f"MEMORY CONTEXT:\n{memory_context or '[None]'}\n\n"
            f"BASE ANSWER:\n{base_answer or '[None]'}\n\n"
            f"{anthropic.AI_PROMPT}"
        )

        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1200,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        if isinstance(response.content, list):
            final_output = "".join(str(block.text) for block in response.content if hasattr(block, "text"))
        else:
            final_output = str(response.content)

        return final_output.strip()

    except Exception as e:
        print(f"Kawl error: {str(e)}")
        return "Kawl encountered an internal error. Try again/Refresh"


CHAT_DIR = "kelp_chats"

def get_chat_path(kelp_name):
    return os.path.join(CHAT_DIR, f"{kelp_name}.json")

def load_chat_history(kelp_name):
    path = get_chat_path(kelp_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(kelp_name, history):
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(get_chat_path(kelp_name), "w") as f:
        json.dump(history, f)

def reset_chat_history(kelp_name):
    path = get_chat_path(kelp_name)
    if os.path.exists(path):
        os.remove(path)

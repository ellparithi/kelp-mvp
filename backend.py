
# backend.py

import os
import io
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

sparse_indexes = {}  # kelp_name -> {vectorizer, matrix, documents}

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize Chroma client (persistent mode)
chroma_client = PersistentClient(path="./chroma_db")

# ========== Smart Chunking Initialization ==========
smart_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

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

def build_sparse_index(collection_name):
    """
    Build and cache TF-IDF sparse index for a given Kelp collection.
    """
    collection = get_or_create_collection(collection_name)
    data = collection.get(include=["documents"])
    documents = data.get("documents", [])

    if not documents:
        print(f"⚠️ No documents to index for {collection_name}")
        return

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    sparse_indexes[collection_name] = {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "documents": documents
    }

# ========== Basic ChromaDB Operations ==========

def get_or_create_collection(collection_name):
    return chroma_client.get_or_create_collection(name=collection_name)

def add_document_to_kelp(collection_name, document_text, metadata=None):
    collection = get_or_create_collection(collection_name)
    chunks = smart_text_splitter.split_text(document_text)
    source_title = metadata.get("source") if metadata else "uploaded"
    metadata_list = [{"source_title": source_title}] * len(chunks)
    source_title = metadata.get("source") if metadata else "uploaded"
    metadata_list = [{"source_title": source_title}] * len(chunks)
    chunk_ids = [f"doc-{hash(chunk)}" for chunk in chunks]

    try:
        collection.add(
            documents=chunks,
            metadatas=metadata_list,
            ids=chunk_ids,
        )
    except Exception as e:
        print(f"❌ Error adding document to Kelp: {str(e)}")

def delete_documents_from_kelp(collection_name, doc_ids: list):
    collection = get_or_create_collection(collection_name)
    try:
        if doc_ids:
            collection.delete(ids=doc_ids)
            return True
        return False
    except Exception as e:
        print(f"❌ Error deleting documents: {str(e)}")
        return False

def delete_entire_kelp(collection_name: str):
    try:
        chroma_client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"❌ Error deleting kelp: {str(e)}")
        return False

# ========== Retrieval ==========

def retrieve_relevant_memories(collection_name, query, top_k=10, similarity_threshold=0.65):
    try:
        collection = get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances"],
        )
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        if not documents:
            return []

        filtered_docs = []
        for doc, distance in zip(documents, distances):
            similarity_score = 1 - distance
            if similarity_score >= similarity_threshold:
                filtered_docs.append(doc)

        return filtered_docs if filtered_docs else documents  # fallback if none pass

    except Exception as e:
        print(f"❌ Error retrieving memories: {str(e)}")
        return []

def hybrid_retrieve_memories(collection_name, query, top_k=10, dense_weight=0.5):
    """
    Combine dense Chroma and sparse TF-IDF scores to retrieve top memory chunks.
    """
    # ----- Step 1: Dense (Chroma) -----
    collection = get_or_create_collection(collection_name)
    dense_results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances"]
    )

    dense_docs = dense_results.get("documents", [[]])[0]
    dense_distances = dense_results.get("distances", [[]])[0]
    dense_scores = [1 - d for d in dense_distances]  # convert distance to similarity

    # ----- Step 2: Sparse (TF-IDF) -----
    if collection_name not in sparse_indexes:
        build_sparse_index(collection_name)

    index = sparse_indexes.get(collection_name)
    if not index:
        return dense_docs  # fallback

    vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(vec, index["matrix"]).flatten()
    sparse_scores = sims.tolist()
    sparse_docs = index["documents"]

    # ----- Step 3: Merge scores -----
    doc_scores = {}

    for doc, score in zip(dense_docs, dense_scores):
        doc_scores[doc] = doc_scores.get(doc, 0) + score * dense_weight

    for doc, score in zip(sparse_docs, sparse_scores):
        doc_scores[doc] = doc_scores.get(doc, 0) + score * (1 - dense_weight)

   
    # ------ Step 4: Rerank & fallback buffer ------
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Separate high- and low-confidence chunks
    confidence_threshold = 0.75
    top_strong = [doc for doc, score in sorted_docs if score >= confidence_threshold]
    top_weak = [doc for doc, score in sorted_docs if score < confidence_threshold]

    # If strong set is too small, pad with top weak matches
    if len(top_strong) < top_k and top_weak:
        top_weak_trimmed = top_weak[: (top_k - len(top_strong))]
        top_docs = top_strong + top_weak_trimmed
    else:
        top_docs = top_strong[:top_k]

    return top_docs


# ========== Reasoning ==========

def kelp_kbase_reasoning(collection_name: str, user_prompt: str, max_tokens: int = 300) -> dict:
    """
    Perform KBase reasoning: retrieves memory, builds context, and returns both answer and memory.
    """
    try:
        # Step 1: Retrieve relevant memory chunks
        # Dynamically adjust top_k based on prompt length
        word_count = len(user_prompt.strip().split())
        top_k = 8 if word_count < 20 else 4

        # Initial hybrid retrieval
        memories = hybrid_retrieve_memories(collection_name, user_prompt, top_k=top_k)

        if not memories:
            return {
                "answer": "❌ Sorry, I couldn't find any relevant information in your Kelp documents.",
                "context": "",
                "memories": []
            }

        # Optional GPT reranking if too many chunks or if fuzzy match likely
        if len(memories) > 4:
            ranking_prompt = (
                f"You are a memory filter. From the following context chunks, choose the 3 most useful ones "
                f"for answering this user question.\n\n"
                f"User Question: {user_prompt}\n\n"
                f"Chunks:\n" + "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(memories)) +
                "\n\nReturn only the chunk texts you consider most relevant."
            )

            try:
                rerank_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": ranking_prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                ranked_text = rerank_response.choices[0].message.content
                # Basic parsing: keep only those chunks that reappear in the answer
                filtered = [chunk for chunk in memories if chunk[:20] in ranked_text]
                if filtered:
                    memories = filtered
            except Exception as e:
                print(f"⚠️ GPT reranking failed: {e}")

        # Step 2: Format retrieved memory into structured context
        context = "\n\n".join(f"- {mem}" for mem in memories if mem.strip())
        if len(context) > 3000:
            context = context[:3000] + "\n\n[Content Truncated]"

        # Clean + deduplicate + trim
        clean_chunks = list(dict.fromkeys([mem.strip() for mem in memories if mem.strip()]))
        context = "\n\n".join(clean_chunks)

        # Limit context to ~3,500 tokens worth of text (≈ 10,000 chars)
        if len(context) > 10000:
            context = context[:10000] + "\n\n[Truncated]"

        # Step 3: Build final prompt
        final_prompt = (
            f"You are Kelp, a helpful AI assistant trained strictly on the following context. "
            f"Answer the user's question based ONLY on the given context. "
            f"If you cannot find the answer in the context, politely say so.\n\n"
            f"Context:\n{context}\n\n"
            f"User's Question: {user_prompt}\n\n"
            f"Answer:"
        )

        # Step 4: Call OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Kelp, a private AI assistant."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "context": context,
            "memories": memories
        }

    except Exception as e:
        print(f"❌ Error in kelp_kbase_reasoning: {str(e)}")
        return {
            "answer": "❌ Internal Error while reasoning over Kelp memories.",
            "context": "",
            "memories": []
        }

def kelp_kawl_reasoning(user_input, base_answer=None, memory_context=None, mode="default") -> str:
    """
    Kawl performs enhanced reasoning using the user's query, KBase's answer, and optional memory context.
    It rewrites answers with clarity, performs small logical deductions, and applies optional formatting styles.
    """
    try:
        system_prompt = (
            "You are Kawl, an advanced AI assistant that builds on the user's memory context and base answer. "
            "Improve clarity, tone, and logical precision in your response. Stay within the context boundaries. "
            "Avoid making up facts. Optionally apply the following style modes: 'default', 'business', 'summary', 'story'."
        )

        # Build prompt content
        full_prompt = f"""User Question: {user_input}

Memory Context:
{memory_context or '[None Provided]'}

Base Answer (from KBase):
{base_answer or '[None Provided]'}

Style Mode: {mode}

Your Task: Provide an improved, well-structured, thoughtful response using the context above. Do NOT hallucinate.
"""

        response = client.chat.completions.create(
            model="gpt-4",  # You can swap with "gpt-4o" if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=600,
            temperature=0.5,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Kawl error: {str(e)}")
        return "❌ Kawl encountered an internal error."


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

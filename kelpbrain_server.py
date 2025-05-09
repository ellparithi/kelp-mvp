
# kelpbrain_server.py

from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from backend import hybrid_retrieve_memories
from openai import OpenAI
import torch


# 🚀 FastAPI app
app = FastAPI()

# 🧠 Global KelpBrain model cache
kelpbrain_models = {}

# 🚀 OpenAI client setup
client = OpenAI()

# 📦 Load local model (for future use)
def load_kelpbrain_model(kelp_name):
    model_path = f"kelpmodels/{kelp_name}"

    if not os.path.exists(model_path):
        raise Exception(f"No model found at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        offload_folder="./offload",
    )
    model.eval()
    return tokenizer, model

# 📩 Query schema
class QueryRequest(BaseModel):
    kelp_name: str
    prompt: str
    reasoning_mode: str = "kbase"
    max_tokens: int = 300

# 📩 Main query endpoint
@app.post("/query/")
def infer_kelpbrain(req: QueryRequest):
    try:
        # 🧠 Memory Retrieval
        # Step 1: Dynamic top_k based on prompt length
        word_count = len(req.prompt.strip().split())
        top_k = 40 if word_count < 20 else 20
        memories = hybrid_retrieve_memories(req.kelp_name, req.prompt, top_k=top_k)

        # 🔁 Multi-Query Expansion Fallback (if very few matches found)
        if len(memories) < 3:
            expansion_prompt = f"Rewrite the following question using 2–3 alternative phrasings or synonyms:\n\n'{req.prompt}'"
            expansion_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.7,
            )
            # Parse the response (assumes comma-separated or newline list)
            alt_queries = expansion_response.choices[0].message.content.strip().split("\n")
            alt_queries = [q.strip(" -•") for q in alt_queries if len(q.strip()) > 0]

            # Expand the query list
            expanded_prompts = [req.prompt] + alt_queries[:3]

            # Run hybrid retrieval for each variation and merge results
            all_memories = []
            for variant in expanded_prompts:
                all_memories.extend(hybrid_retrieve_memories(req.kelp_name, variant, top_k=top_k))

            # De-duplicate
            memories = list(dict.fromkeys([m.strip() for m in all_memories if m.strip()]))


        # Step 2: GPT reranking if too many chunks
        if len(memories) > 4:
            memories = memories[:6]  # Cap before rerank
            rerank_prompt = (
                f"You are a memory filter that needs to retrieve the right memory for the prompt user sends. From the following chunks, select the 3 most useful and relevant ones to answer the user prompt:\n\n"
                f"Return only the texts that are relevant. Ignore unrelated or vague ones.\n\n"
                f"User Question: {req.prompt}\n\n"
                f"Chunks:\n" + "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(memories)) +
                "\n\nReturn only the chunk texts you find relevant."
            )
            try:
                rerank_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rerank_prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                rerank_text = rerank_response.choices[0].message.content
                filtered = [chunk for i, chunk in enumerate(memories) if f"[{i+1}]" in rerank_text]
                if filtered:
                    memories = filtered
            except Exception as e:
                print(f"⚠️ GPT reranking failed: {e}")

        clean_chunks = list(dict.fromkeys([m.strip() for m in memories if m.strip()]))
        context = "\n\n".join(clean_chunks)
        if len(context) > 10000:
            context = context[:10000] + "\n\n[Truncated]"

        # 🧠 Form final augmented prompt
        final_prompt = f"{context}\n\nUser Question: {req.prompt}"

        if req.reasoning_mode.lower() == "kbase":
            # 🔥 KBase Mode — simulate using GPT-3.5 level
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are Kelp, a private assistant that answers based only on the provided context. If the answer is not in the context, say 'I'm sorry, I don't have enough information based on the memory.'"},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=req.max_tokens,
                temperature=0.5,
                top_p=0.9
            )
            kelp_response = response.choices[0].message.content.strip()

        elif req.reasoning_mode.lower() == "kawl":
            # 🔥 Kawl Mode — simulate using GPT-4 deeper reasoning
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Kelp, an advanced assistant using uploaded memories. You must combine insights if possible. If context is missing, explain politely."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=req.max_tokens,
                temperature=0.4,
                top_p=0.8
            )
            kelp_response = response.choices[0].message.content.strip()

        else:
            kelp_response = "❌ Invalid reasoning mode selected."

        return {"response": kelp_response}

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return {"response": f"❌ Internal KelpBrain Error: {str(e)}"}

# 🏁 Run with: uvicorn kelpbrain_server:app --reload --port 8000

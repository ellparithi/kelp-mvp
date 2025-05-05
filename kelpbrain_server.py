
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
        top_k = 8 if word_count < 20 else 4
        memories = hybrid_retrieve_memories(req.kelp_name, req.prompt, top_k=top_k)

        # Step 2: GPT reranking if too many chunks
        if len(memories) > 4:
            rerank_prompt = (
                f"You are a memory filter. From the following chunks, select the 3 most useful ones to answer:\n\n"
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
                filtered = [chunk for chunk in memories if chunk[:20] in rerank_text]
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
                    {"role": "system", "content": "You are a private assistant that answers based only on the provided context. If the answer is not in the context, say 'I'm sorry, I don't have enough information based on the memory.'"},
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
                    {"role": "system", "content": "You are an advanced assistant using uploaded memories. You must combine insights if possible. If context is missing, explain politely."},
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

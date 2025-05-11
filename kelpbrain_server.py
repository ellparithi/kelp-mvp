
# kelpbrain_server.py

from dotenv import load_dotenv
load_dotenv()
import os

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from backend import kelp_kbase_reasoning, kelp_kawl_reasoning
import anthropic

# 🚀 FastAPI app
app = FastAPI()

# 🚀 OpenAI client setup
client = OpenAI()
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# 📩 Query schema
from typing import Optional, List

class QueryRequest(BaseModel):
    kelp_name: Optional[str] = None
    prompt: str
    doc_corpus: Optional[List[dict]] = None  # expects [{filename, text}]
    reasoning_mode: str = "kbase"
    max_tokens: int = 300

# 📩 Main query endpoint

@app.post("/query/")
def infer_kelpbrain(req: QueryRequest):
    try:
        if req.doc_corpus:
            print("⚡ Using document-level reasoning via Kelp v2.0")

            base = kelp_kbase_reasoning(req.prompt, req.doc_corpus)

            if req.reasoning_mode.lower() == "kbase":
                return {"response": base["answer"]}

            elif req.reasoning_mode.lower() == "kawl":
                answer = kelp_kawl_reasoning(
                    user_input=req.prompt,
                    base_answer=base["answer"],
                    memory_context=base["context"]
                )
                return {"response": answer}

            else:
                return {"response": "❌ Invalid reasoning mode selected."}

        else:
            return {"response": "❌ No document corpus provided."}

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return {"response": f"❌ Internal KelpBrain Error: {str(e)}"}

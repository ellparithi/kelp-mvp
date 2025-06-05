# Kelp MVP

**Kelp** is a secure, private AI system for regulated industries.

---

##  Features

- Upload up to **10 documents** (`.pdf`, `.docx`, `.txt`, `.csv`)
- Talk to **KBase** (fast, local LLM) or **Kawl** (enhanced via Claude)
- Saved conversations per document set (called **Kelps**)
- Delete, create, and manage document collections
- Works locally & can be deployed (Render, Fly.io, etc.)

---

##  This MVP

This MVP is a **demo version** to show functionality.

-  Do **not** upload confidential data — this is not fully secure
- Hosted on **Render** – memory is **not persisted**
- Reach out if you want a persistent private deployment

---

## Tech Stack

- Python + Streamlit (frontend)
- ChromaDB (for vector search)
- OpenAI (for embeddings + KBase)
- Anthropic Claude (for reasoning in Kawl)
- Custom local session manager

---

##  Setup Locally

```bash
git clone https://github.com/yourusername/kelp-mvp.git
cd kelp-mvp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export CLAUDE_API_KEY=your_key

streamlit run app.py
```

---

##  Contact

Made by **Elamparithi Kavi Elango**

- Email: [elamparithi.ke@gmail.com](mailto:elamparithi.ke@gmail.com)
- Website: [www.kelpllm.com](https://www.kelpllm.com)

---

##  License

MIT License

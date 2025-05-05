
import os
import json

CHAT_ROOT = "./kelp_chats"

def get_chat_dir(kelp_name):
    return os.path.join(CHAT_ROOT, kelp_name)

def list_chat_sessions(kelp_name):
    kelp_dir = get_chat_dir(kelp_name)
    if not os.path.exists(kelp_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(kelp_dir) if f.endswith(".json"))

def save_chat_session(kelp_name, session_id, messages):
    kelp_dir = get_chat_dir(kelp_name)
    os.makedirs(kelp_dir, exist_ok=True)
    path = os.path.join(kelp_dir, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(messages, f)
    
    safe_name = session_id.replace(" ", "_")
    path = os.path.join(get_chat_dir(kelp_name), f"{safe_name}.json")

def load_chat_session(kelp_name, session_id):
    path = os.path.join(get_chat_dir(kelp_name), f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def delete_chat_session(kelp_name, session_id):
    path = os.path.join(get_chat_dir(kelp_name), f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)

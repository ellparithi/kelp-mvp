
import os
import time
import subprocess

WATCH_INTERVAL = 60  # seconds
KELP_ROOT = "kelpdata"

def get_raw_doc_path(kelp_name):
    return os.path.join(KELP_ROOT, kelp_name, "raw_docs.txt")

def get_last_modified(filepath):
    try:
        return os.path.getmtime(filepath)
    except FileNotFoundError:
        return None

def fine_tune_kelp(kelp_name):
    print(f"🏋️‍♂️ Fine-tuning triggered for '{kelp_name}'...")
    subprocess.run([
        "python",
        "fine_tune_lora_kelp.py",
        "--kelp_name", kelp_name,
        "--epochs", "3"   # or you can pass custom epochs
    ])
    print(f"✅ Fine-tuning completed for '{kelp_name}'!")

def watch_kelp():
    print("🔭 Starting Kelp Fine-Tune Watcher...")
    kelp_states = {}

    while True:
        kelp_names = [name for name in os.listdir(KELP_ROOT) if os.path.isdir(os.path.join(KELP_ROOT, name))]

        for kelp_name in kelp_names:
            raw_path = get_raw_doc_path(kelp_name)
            current_mtime = get_last_modified(raw_path)

            if current_mtime is None:
                continue  # no raw docs yet

            last_mtime = kelp_states.get(kelp_name)

            if last_mtime is None:
                # first time seeing this kelp
                kelp_states[kelp_name] = current_mtime
            elif current_mtime > last_mtime:
                # change detected
                kelp_states[kelp_name] = current_mtime
                fine_tune_kelp(kelp_name)

        time.sleep(WATCH_INTERVAL)

if __name__ == "__main__":
    watch_kelp()


import requests

# URL of your running KelpBrain server
url = "http://127.0.0.1:8000/query/"

# Prepare the test payload
payload = {
    "kelp_name": "test-kelp",  # Any name for now
    "prompt": "Where was Kelp founded?",
    "reasoning_mode": "kbase",  # Try "kbase" first
    "max_tokens": 300
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print("KBase Answer:", response.json())

# 🧠 Now let's test Kawl mode too!

payload["reasoning_mode"] = "kawl"  # Change reasoning mode

response2 = requests.post(url, json=payload)

print("Kawl Answer:", response2.json())

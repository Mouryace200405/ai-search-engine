
import os
import toml
from huggingface_hub import InferenceClient

try:
    secrets = toml.load(".streamlit/secrets.toml")
    HF_TOKEN = secrets["HF_TOKEN"]
except:
    print("No secrets found")
    exit(1)

models_to_test = [
    "mistralai/Mistral-7B-Instruct-v0.2", # 32k context
    "google/gemma-7b-it", # 8k but good
    "HuggingFaceH4/zephyr-7b-beta", # Current (8k)
    "allenai/led-large-16384-arxiv" # Summarization (16k)
]

for model in models_to_test:
    print(f"Testing {model}...")
    client = InferenceClient(api_key=HF_TOKEN)
    try:
        # Try chat (for instruct models)
        if "led" not in model:
            messages = [{"role": "user", "content": "Hello"}]
            client.chat_completion(messages, model=model, max_tokens=10)
            print(f"✅ {model} supports chat_completion")
        else:
            # Try summarization for LED
            client.summarization("This is a long text to summarize.", model=model)
            print(f"✅ {model} supports summarization")
    except Exception as e:
        print(f"❌ {model} failed: {e}")

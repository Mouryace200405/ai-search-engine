
import os
import toml
from huggingface_hub import InferenceClient

def verify_long_context():
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        HF_TOKEN = secrets["HF_TOKEN"]
    except Exception as e:
        print(f"Could not load secrets: {e}")
        exit(1)

    REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Testing {REPO_ID} with large context...")

    # Create a 20k character dummy context
    dummy_text = "This is a sentence about AI search engines. " * 500 # ~22k chars
    dummy_context = f"Context:\n{dummy_text}\n\nQuestion: Summarize the context."
    
    print(f"Context length: {len(dummy_context)} characters")

    client = InferenceClient(api_key=HF_TOKEN)
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": dummy_context}
        ]
        
        response = client.chat_completion(messages, model=REPO_ID, max_tokens=100)
        print("✅ Response received:")
        print(response.choices[0].message.content)
        print("✅ Verification Passed: Model accepted large context.")
    except Exception as e:
        print(f"❌ Failed: {e}")
        exit(1)

if __name__ == "__main__":
    verify_long_context()


import os
import toml
from langchain_huggingface import HuggingFaceEndpoint

try:
    secrets = toml.load(".streamlit/secrets.toml")
    HF_TOKEN = secrets["HF_TOKEN"]
except Exception as e:
    print(f"Could not load secrets: {e}")
    exit(1)

# Try Zephyr
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
print(f"Testing {REPO_ID} with invoke...")

try:
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID, 
        huggingfacehub_api_token=HF_TOKEN, 
        temperature=0.7
    )
    response = llm.invoke("Why is the sky blue?")
    print(f"Response: {response[:50]}...")
    print("Verification Passed!")
except Exception as e:
    print(f"Failed: {e}")
    exit(1)

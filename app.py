import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from huggingface_hub import InferenceClient

# --- CONFIG & STYLING ---
st.set_page_config(page_title="AI Search Engine", page_icon="🔎")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTextInput > div > div > input { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS & API CONFIG ---
HF_TOKEN = st.secrets["HF_TOKEN"]

# Mistral model for answering
LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# --- LOGIC FUNCTIONS ---

def scrape_url(url):
    """Fetches and cleans text from a URL."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove junk elements
        for script in soup(["script", "style", "nav", "footer", "header", "form", "svg"]):
            script.decompose()
            
        # Try to find main content or body
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = main_content.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
            
        # heavy cleaning
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        clean_text = ' '.join(lines)
        return clean_text[:2000] # Limit per site
    except Exception as e:
        return ""

def run_ai_search(user_query):
    # Step 1: Search for 2 best URLs
    with st.status("🔍 Searching Web...", expanded=True) as status:
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.text(user_query, max_results=2)]

        # Step 2: Scrape and collect limited text
        best_context = ""
        for res in search_results:
            st.write(f"Scraping: {res['title']}")
            text = scrape_url(res['href'])
            # Take only the first 1000 characters from each site
            best_context += f"Source: {res['title']}\nContent: {text[:1000]}\n\n"

        status.update(label="Analysis Complete!", state="complete", expanded=False)

    return best_context, search_results

# --- USER INTERFACE ---
st.title("🔎 Real-Time AI Search Engine")
st.caption("Powered by Zephyr-7B (Generation)")

user_input = st.text_input("Enter your question:", placeholder="e.g. What is the latest news about SpaceX Starship?")

if user_input:
    context, sources = run_ai_search(user_input)

    if context:
        # Step 4: Final Generation with Mistral
        client = InferenceClient(api_key=HF_TOKEN)

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer the user question primarily using the provided context. If the context is irrelevant, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]

        with st.chat_message("assistant"):
            response = client.chat_completion(messages, model=LLM_REPO_ID, max_tokens=512)
            st.write(response.choices[0].message.content)

        st.sidebar.header("Sources Found")
        for s in sources:
            st.sidebar.markdown(f"- [{s['title']}]({s['href']})")
    else:
        st.error("Could not find enough relevant data. Try a different query.")

import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG & STYLING ---
# --- CONFIG & STYLING ---
st.set_page_config(page_title="Web Search Bot", page_icon="🔎", layout="wide")

st.markdown("""
    <style>
    /* --- DUAL THEME SUPPORT (Tokyo Night & Ash) --- */
    
    /* Default Variables (Tokyo Night - Dark) */
    /* --- DUAL THEME SUPPORT (Tokyo Night & Default Light) --- */
    
    /* Force Iosevka everywhere (Global) */
    * {
        font-family: 'Iosevka', monospace !important;
    }

    /* TOKYO NIGHT THEME (Only active in Dark Mode) */
    [data-theme="dark"] {
        --bg-color: #1a1b26;
        --card-bg: #24283b;
        --text-color: #c0caf5;
        --accent-color: #7aa2f7;
        --input-bg: #24283b;
        --border-color: #414868;
        --hover-bg: #414868;
        --shadow: rgba(0,0,0,0.3);
        --placeholder-color: #565f89;
    }

    /* Apply styles ONLY when theme is dark */
    [data-theme="dark"] .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }

    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3, 
    [data-theme="dark"] h4, [data-theme="dark"] h5, [data-theme="dark"] h6, 
    [data-theme="dark"] p, [data-theme="dark"] label, [data-theme="dark"] span, 
    [data-theme="dark"] div {
        color: var(--text-color) !important;
    }

    [data-theme="dark"] .stTextInput > div > div > input { 
        border-radius: 20px !important;
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        caret-color: var(--text-color) !important;
    }

    [data-theme="dark"] .stTextInput input::placeholder {
        color: var(--placeholder-color) !important;
        opacity: 1 !important;
    }

    /* Source Cards (Dark Mode specific) */
    [data-theme="dark"] .source-card {
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px var(--shadow);
    }
    [data-theme="dark"] .source-card:hover {
        background-color: var(--hover-bg);
        border-color: var(--accent-color);
    }
    
    /* Source Cards (Light Mode Fallback / Default) */
    [data-theme="light"] .source-card {
        background-color: #ffffff;
        color: #31333f !important;
        border: 1px solid #d6d6d8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-theme="light"] .source-card:hover {
        background-color: #f0f2f6;
        border-color: #ff4b4b; /* Streamlit Default Red Accent */
    }

    /* Shared Layout Styles (Both Themes) */
    .source-card {
        display: inline-block;
        padding: 12px;
        border-radius: 12px;
        margin-right: 10px;
        margin-bottom: 10px;
        text-decoration: none;
        font-size: 0.85em;
        transition: all 0.2s;
        width: 100%;
        height: 110px;
        overflow: hidden;
    }

    .source-title {
        font-weight: bold;
        margin-bottom: 6px;
        display: block;
        font-size: 1.1em;
    }
    [data-theme="dark"] .source-title { color: var(--accent-color) !important; }
    [data-theme="light"] .source-title { color: #ff4b4b !important; }

    .source-url {
        font-size: 0.75em;
        opacity: 0.7;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    [data-theme="dark"] .source-url { color: var(--text-color) !important; }
    [data-theme="light"] .source-url { color: #31333f !important; }

    /* Answer Card */
    .answer-card {
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
    }
    [data-theme="dark"] .answer-card {
        background-color: var(--card-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px var(--shadow);
    }
    [data-theme="light"] .answer-card {
        background-color: #ffffff;
        color: #31333f;
        border: 1px solid #d6d6d8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .answer-card p {
        text-align: justify;
        line-height: 1.6;
        font-size: 1.05em;
    }
    [data-theme="dark"] .answer-card p { color: var(--text-color) !important; }
    [data-theme="light"] .answer-card p { color: #31333f !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS & API CONFIG ---
HF_TOKEN = st.secrets["HF_TOKEN"]

# Mistral model for answering (32k context)
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

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
        return clean_text[:5000] # Limit per site (increased for Mistral)
    except Exception as e:
        return ""

def run_ai_search(user_query):
    # Step 1: Search for 5 best URLs
    with st.status("🔍 Searching Web...", expanded=True) as status:
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.text(user_query, max_results=5)]

        # Step 2: Parallel scraping
        best_context = ""
        st.write(f"Scraping {len(search_results)} sites concurrently...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create a map of future -> result_metadata
            future_to_url = {
                executor.submit(scrape_url, res['href']): res 
                for res in search_results
            }
            
            completed_count = 0
            for future in as_completed(future_to_url):
                res = future_to_url[future]
                try:
                    text = future.result()
                    if text:
                        completed_count += 1
                        # Update status cleanly
                        status.update(label=f"Scraping... ({completed_count}/{len(search_results)} done)", state="running")
                        
                        # Take only the first 4000 characters from each site to fit in context
                        best_context += f"Source: {res['title']}\nContent: {text[:4000]}\n\n"
                except Exception as e:
                    pass # Fail silently to avoid UI clutter

        status.update(label="Analysis Complete!", state="complete", expanded=False)

    return best_context, search_results

# --- USER INTERFACE ---
st.title("🔎 Real-Time AI Search Engine")
st.caption("Powered by Mistral-7B-v0.2 (32k Context)")

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

        # Perplexity-style Layout: Sources -> Answer
        st.subheader("Sources")
        
        # Display sources as cards
        cols = st.columns(5)
        for i, s in enumerate(sources):
            with cols[i % 5]:
                st.markdown(
                    f'''
                    <a href="{s['href']}" target="_blank" class="source-card">
                        <span class="source-title">{i+1}. {s['title'][:20]}...</span>
                        <span class="source-url">{s['href']}</span>
                    </a>
                    ''', 
                    unsafe_allow_html=True
                )

        st.subheader("Answer")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat_completion(messages, model=LLM_REPO_ID, max_tokens=1024)
                
                # Wrap answer in styled card
                st.markdown(f"""
                <div class="answer-card">
                    {response.choices[0].message.content}
                </div>
                """, unsafe_allow_html=True)

    else:
        st.error("Could not find enough relevant data. Try a different query.")

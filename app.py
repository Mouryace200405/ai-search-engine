import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Web Search AI",
    page_icon="🔎",
    layout="wide"
)

HF_TOKEN = st.secrets["HF_TOKEN"]

MODEL_PRIMARY = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_FALLBACK = "HuggingFaceH4/zephyr-7b-beta"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------- UI ----------------
st.title("Web Search Bot")
st.caption("• Open Models • Limited context Length")

query = st.text_input(
    "Ask anything",
    placeholder="What is the latest update on SpaceX Starship?"
)
with st.sidebar:
    st.header("⚙️ Settings")
    MAX_RESULTS = st.slider(
        "Search results",
        min_value=3,
        max_value=8,      
        value=5
    )
    MAX_CONTEXT_CHARS = st.slider(
        "Context per site (chars)",
        min_value=500,
        max_value=2000,  
        value=1500
    )
    SHOW_RAW = st.checkbox("Show scraped context")

# ---------------- SCRAPING ----------------
def scrape_url(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "svg", "form"]):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 80
        ]

        text = " ".join(paragraphs)
        return text[:MAX_CONTEXT_CHARS]

    except Exception:
        return ""

# ---------------- SEARCH ----------------
def search_and_scrape(q):
    with DDGS() as ddgs:
        results = list(ddgs.text(q, max_results=MAX_RESULTS))

    collected = []

    with ThreadPoolExecutor(max_workers=MAX_RESULTS) as ex:
        futures = {
            ex.submit(scrape_url, r["href"]): r
            for r in results
        }

        for f in as_completed(futures):
            text = f.result()
            meta = futures[f]
            if text:
                collected.append({
                    "title": meta["title"],
                    "url": meta["href"],
                    "text": text
                })

    return collected

# ---------------- LLM ----------------
def ask_llm(context, question):
    client = InferenceClient(api_key=HF_TOKEN)

    prompt = f"""
You are a factual assistant.
Answer ONLY using the context below.
If context is insufficient, say "Not enough reliable data."

Context:
{context}

Question:
{question}
"""

    for model in [MODEL_PRIMARY, MODEL_FALLBACK]:
        try:
            res = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700
            )
            return res.choices[0].message.content
        except Exception:
            continue

    return "⚠️ Model unavailable. Try again later."

# ---------------- MAIN FLOW ----------------
if query:
    with st.status("🔍 Searching & scraping...", expanded=True):
        sources = search_and_scrape(query)

    if not sources:
        st.error("No usable sources found.")
        st.stop()

    context = ""
    for i, s in enumerate(sources, 1):
        context += f"[{i}] {s['title']}\n{s['text']}\n\n"

    st.subheader("📌 Sources")
    cols = st.columns(len(sources))
    for i, s in enumerate(sources):
        cols[i].markdown(f"**[{i+1}. {s['title'][:30]}]({s['url']})**")

    if SHOW_RAW:
        with st.expander("📄 Scraped Context"):
            st.text(context)

    st.subheader("🤖 Answer")
    with st.spinner("Thinking..."):
        answer = ask_llm(context, query)
        st.markdown(answer)

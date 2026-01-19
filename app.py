import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Web Search Bot", page_icon="🔎", layout="wide")




HF_TOKEN = st.secrets["HF_TOKEN"]


LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"



def scrape_url(url):
    
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
       
        for script in soup(["script", "style", "nav", "footer", "header", "form", "svg"]):
            script.decompose()
            
       
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = main_content.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
            
      
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        clean_text = ' '.join(lines)
        return clean_text[:5000] 
    except Exception as e:
        return ""

def run_ai_search(user_query):
    
    with st.status("🔍 Searching Web...", expanded=True) as status:
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.text(user_query, max_results=5)]

      
        best_context = ""
        st.write(f"Scraping {len(search_results)} sites concurrently...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
        
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
                        
                        status.update(label=f"Scraping... ({completed_count}/{len(search_results)} done)", state="running")
                        
                        
                        best_context += f"Source: {res['title']}\nContent: {text[:4000]}\n\n"
                except Exception as e:
                    pass 

        status.update(label="Analysis Complete!", state="complete", expanded=False)

    return best_context, search_results


st.title("🔎 Web Search Bot")
st.caption("Powered by Mistral-7B-v0.2 (32k Context)")

user_input = st.text_input("Enter your question:", placeholder="e.g. What is the latest news about SpaceX Starship?")

if user_input:
    context, sources = run_ai_search(user_input)

    if context:
      
        client = InferenceClient(api_key=HF_TOKEN)

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer the user question primarily using the provided context. If the context is irrelevant, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]

        
        st.subheader("Sources")
        
        cols = st.columns(5)
        for i, s in enumerate(sources):
            with cols[i % 5]:
                st.markdown(f"**[{i+1}. {s['title'][:20]}...]({s['href']})**")

        st.subheader("Answer")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat_completion(messages, model=LLM_REPO_ID, max_tokens=1024)
                st.markdown(response.choices[0].message.content)

    else:
        st.error("Could not find enough relevant data. Try a different query.")

#  AI Search Engine

A locally hosted AI search engine that performs real-time web searches, scrapes content concurrently, and generates answers using **Mistral-7B-Instruct-v0.2** (via Hugging Face API).


##  Features

- **Real-Time Search**: Fetches live results using DuckDuckGo.
- **Concurrent Scraping**: Scrapes 5+ sources in parallel for fast context retrieval.
- **Smart Answering**: Uses Mistral-7B (32k context) to answer questions based on searched content.

##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mouryace200405/ai-search-engine.git
   cd ai-search-engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**:
   Create a `.streamlit/secrets.toml` file and add your Hugging Face Token:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```

##  Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

##  Tech Stack

- **Frontend**: Streamlit (Python)
- **Search**: `ddgs` (DuckDuckGo Search)
- **Scraping**: `requests`, `beautifulsoup4`
- **LLM**: `huggingface_hub` (Mistral-7B)



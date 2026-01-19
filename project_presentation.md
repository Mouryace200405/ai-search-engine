# AI Search Engine Project

---

## 1. Problem Statement

*   **Information Overload**: Traditional search engines provide lists of links, requiring users to manually sift through multiple pages to find answers.
*   **Static Knowledge**: Standard Large Language Models (LLMs) have training data cutoffs and cannot answer questions about recent events (e.g., "SpaceX Starship Flight 6").
*   **Privacy Concerns**: Many modern search tools track user history aggressively.
*   **Goal**: Create a tool that combines the *real-time* capability of a search engine with the *synthesis* capability of an LLM, without the complexity of heavy tracking.

---

## 2. Proposed Solution

*   **Real-Time AI Search Engine**: A Python-based web application.
*   **Key Features**:
    *   **Live Web Search**: Uses DuckDuckGo to find the most relevant current URLs.
    *   **Intelligent Scraping**: Extracts only the meaningful content (article text) from websites, filtering out ads and navigation.
    *   **Generative Synthesis**: Uses a powerful open-source LLM (`Zephyr-7B`) to read the scraped content and answer the user's question directly.
    *   **Privacy-First**: No user tracking or history storage.

---

## 3. System Development Approach

*   **Agile Methodology**: Iterative development focusing on getting a working MVP (Minimum Viable Product) first.
*   **Modular Architecture**:
    *   **Frontend**: Streamlit (fast UI prototyping, Python-native).
    *   **Backend Logic**: Search -> Scrape -> Generate pipeline.
    *   **AI Integration**: Hugging Face Inference API (serverless LLM inference).
*   **Optimization Phase**: Initially used BERT for re-ranking (slow), then optimized to direct scraping of top results for a "real-time" feel.

---

## 4. Algorithm & Deployment

### Algorithm
1.  **Input**: User enters query (e.g., "latest pixel phone").
2.  **Search**: `ddgs` library fetches top 2 URL results.
3.  **Extraction**: `requests` + `BeautifulSoup` downloads page HTML.
    *   *Cleaning Strategy*: Remove `<script>`, `<nav>`, `<footer>`. Filter short lines. Limit to 2000 chars.
4.  **Prompt Engineering**: Construct a prompt with `System: You are a helpful assistant` and `User: Context: {scraped_text} Question: {query}`.
5.  **Inference**: Send prompt to `HuggingFaceH4/zephyr-7b-beta` via `InferenceClient`.
6.  **Output**: Stream response to UI and list sources in sidebar.

### Deployment
*   **Local Host**: Runs via `streamlit run app.py`.
*   **Requirements**: Python 3.13, dependencies (`streamlit`, `ddgs`, `huggingface_hub`, `bs4`).

---

## 5. Result

*   **Functional Success**: The application successfully answers questions about current events (verified with "SpaceX Flight 6" query).
*   **Performance**: Search-to-answer latency reduced significantly by removing the BERT re-ranking step.
*   **Accuracy**: High relevance due to direct context injection (RAG - Retrieval Augmented Generation).
*   **Quality**: "No web data" issues resolved by improved HTML cleaning logic.

---

## 6. Conclusion

*   We successfully built a **Real-Time AI Search Engine** that bridges the gap between static LLMs and dynamic web search.
*   The system demonstrates the power of **Retrieval Augmented Generation (RAG)** using purely open-source tools and free inference APIs.
*   It offers a clean, ad-free, and privacy-respecting alternative to major search engines for quick queries.

---

## 7. Future Scope

*   **Cloud Deployment**: Host on Streamlit Community Cloud or Hugging Face Spaces for public access.
*   **Multi-Modal Search**: Support image and video search results.
*   **History & Sessions**: Optional session storage for follow-up questions.
*   **Citation Linking**: Highlight exactly which sentence in the answer came from which source.
*   **Model Selection**: Allow users to toggle between models (e.g., Llama 3, Mistral, Zephyr).

---

## 8. References

*   **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
*   **Hugging Face Hub**: [huggingface.co/docs/huggingface_hub](https://huggingface.co/docs/huggingface_hub)
*   **DuckDuckGo Search (ddgs)**: [pypi.org/project/duckduckgo-search/](https://pypi.org/project/duckduckgo-search/)
*   **Beautiful Soup**: [crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/)
*   **Zephyr 7B Model**: [huggingface.co/HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

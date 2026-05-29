# 🧠 AI Knowledge Assistant

> Upload a document. Ask anything. Powered by RAG + LLMs.

An end-to-end **Agentic RAG application** built from scratch to demonstrate enterprise-grade Generative AI concepts — including conversational retrieval, multi-document reasoning, explainable AI, and agentic tool routing.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **LLM Support** | OpenAI GPT-4.1-mini + Local Llama 3.2 (via Ollama) |
| **RAG Pipeline** | PDF ingestion → Chunking → Embeddings → FAISS vector indexing |
| **Agentic Router** | Automatically routes to the right tool based on user intent |
| **Conversational RAG** | History-aware query rewriting for multi-turn Q&A |
| **Retrieval Dashboard** | Chunk preview, source attribution, confidence scoring |
| **Cost Monitoring** | Real-time token tracking and estimated cost per query |
| **Multi-document** | Compare, question, and summarise across multiple PDFs |

---

## 🛠️ Three Agentic Tools

### 1. `question_tool()` — Conversational RAG
Ask questions against your uploaded documents with full retrieval explainability.
- History-aware query rewriting
- FAISS semantic retrieval
- Source attribution with confidence scores

### 2. `compare_tool()` — Multi-Document Reasoning
Compare two or more documents side by side.
- Resume vs Job Description → skill gap analysis
- Contract vs Policy → clause comparison
- Research papers → cross-document Q&A

### 3. `summarize_tool()` — Document Summarisation
Generate concise, structured summaries from any uploaded document.
- Executive summaries
- Key point extraction
- Technical or business-focused output

---

## 🏗️ Architecture

```
User
 ↓
Agent Router
 ├── question_tool()
 │    ↓
 │    History-Aware Query Rewriter
 │    ↓
 │    FAISS Retrieval → Retrieval Dashboard → LLM → Answer
 │
 ├── compare_tool()
 │    ↓
 │    Document Grouping → Comparison Dashboard → LLM → Analysis
 │
 └── summarize_tool()
      ↓
      Context Builder → LLM → Summary

All tools share:
  Conversation Memory → generate_response() → OpenAI / Ollama
```

---

## ⚙️ RAG Pipeline

```
PDF Upload
 ↓
Text Extraction
 ↓
Chunking (configurable size + overlap)
 ↓
Embedding Generation
 ↓
FAISS Vector Indexing
 ↓
Semantic Retrieval → Metadata Enrichment → LLM → Answer
```

---

## 🖥️ Tech Stack

- **Frontend** — Streamlit
- **LLMs** — OpenAI API, Ollama (local inference)
- **Vector Store** — FAISS
- **Embeddings** — OpenAI / HuggingFace
- **Document Processing** — PyPDF
- **Language** — Python 3.10+

---

## 📦 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key (for cloud inference)
- [Ollama](https://ollama.com) installed (for local inference)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-knowledge-assistant.git
cd ai-knowledge-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Run

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📸 Screenshots

| Home & Upload | Query Rewriting + Retrieval |
|---|---|
| ![Home](screenshots/screenshot_1.png) | ![Retrieval](screenshots/screenshot_2.png) |

| Cost & Token Tracking | Document Analysis Output |
|---|---|
| ![Cost](screenshots/screenshot_3.png) | ![Output](screenshots/screenshot_4.png) |

---

## 🗺️ Roadmap

- [x] OpenAI + Ollama integration
- [x] FAISS vector store
- [x] Agentic tool routing
- [x] Conversational RAG with query rewriting
- [x] Retrieval dashboard + confidence scoring
- [x] Cost and token monitoring
- [ ] **LangGraph integration** — stateful graph workflows
- [ ] Retrieval grading + self-correcting RAG
- [ ] Multi-agent collaboration
- [ ] Inline source citation

---

## 🤝 Contributing

This is a learning project, but contributions and discussions are very welcome! Feel free to open an issue or reach out if you're working on similar problems.

---

## 📄 License

MIT License — free to use, learn from, and build upon.

---

## 👤 Author

Built with curiosity by **[Your Name]**
- LinkedIn: [your-linkedin-url]
- GitHub: [@your-username]

> ⭐ If this helped you learn something, a star on the repo means a lot!

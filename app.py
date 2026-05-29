import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import time
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0d0a0b, #1a1015, #231820);
        min-height: 100vh;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f59e0b, #f97316, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #a8947e;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 2rem;
    }

    .badge {
        background: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(249, 115, 22, 0.35);
        color: #fdba74;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 200, 150, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        color: #fde8d0 !important;
    }

    .glass-card p,
    .glass-card span,
    .glass-card li,
    .glass-card ol,
    .glass-card ul {
        color: #fde8d0 !important;
    }

    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #7a5c4a;
        margin-bottom: 0.5rem;
    }

    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 200, 150, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    [data-testid="metric-container"] label {
        color: #a8947e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #fde8d0 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(249, 115, 22, 0.35) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: border-color 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(249, 115, 22, 0.75) !important;
    }
    
    .stTextInput input {
        background: #1a1015 !important;
        border: 1px solid rgba(249, 115, 22, 0.35) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        caret-color: #f97316 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.3s ease;
    }

    .stTextInput input:focus {
        border-color: rgba(249, 115, 22, 0.65) !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.12) !important;
        color: #ffffff !important;
        background: #1a1015 !important;
    }

    .stTextInput input::placeholder {
        color: #7a5c4a !important;
        opacity: 1 !important;
    }

    .stTextInput > div > div > input {
        color: #ffffff !important;
        background: #1a1015 !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #c2410c, #d97706) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.2s ease !important;
    }

    .stButton button:hover {
        opacity: 0.88 !important;
    }

    .stSuccess {
        background: rgba(45, 212, 191, 0.08) !important;
        border: 1px solid rgba(45, 212, 191, 0.25) !important;
        border-radius: 10px !important;
        color: #5eead4 !important;
    }

    .stInfo {
        background: rgba(251, 191, 36, 0.08) !important;
        border: 1px solid rgba(251, 191, 36, 0.25) !important;
        border-radius: 10px !important;
        color: #fcd34d !important;
    }

    .stWarning {
        background: rgba(249, 115, 22, 0.08) !important;
        border: 1px solid rgba(249, 115, 22, 0.25) !important;
        border-radius: 10px !important;
        color: #fdba74 !important;
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255, 200, 150, 0.08) !important;
        border-radius: 10px !important;
        color: #d4b8a0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255, 200, 150, 0.06) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        color: #a8947e !important;
    }

    hr {
        border-color: rgba(255, 200, 150, 0.08) !important;
    }

    .stSpinner > div {
        border-top-color: #2dd4bf !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(13, 10, 11, 0.97) !important;
        border-right: 1px solid rgba(255, 200, 150, 0.08) !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #d4b8a0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #fde8d0 !important;
        font-family: 'Syne', sans-serif !important;
    }

    p, li, span, label {
        color: #d4b8a0;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #fde8d0 !important;
    }

    .stCaption {
        color: #7a5c4a !important;
        font-size: 0.78rem !important;
    }

    .agent-tag {
        display: inline-block;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fcd34d;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #5a3f30;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255, 200, 150, 0.07);
    }

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =====================================================
# SESSION STATE INIT (must be before sidebar stats)
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# FULL CONVERSATION MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("## ⚙️ Model Selection")

model_choice = st.sidebar.radio(
    "Choose your AI model:",
    [
        "OpenAI (GPT-4.1-mini)",
        "Local (Llama 3.2 3B)",
        "Local (Llama 3.2 1B) - Faster"
    ],
    key="model_selection"
)

if model_choice == "OpenAI (GPT-4.1-mini)":
    st.sidebar.success("✅ OpenAI API — fast with token tracking")

elif model_choice == "Local (Llama 3.2 3B)":
    st.sidebar.warning("⚠️ Local CPU — 15–30 sec per response")

else:
    st.sidebar.warning("⚠️ Local CPU — 8–10 sec per response")

st.sidebar.markdown("---")

# ── RAG Settings ─────────────────────────────────────
st.sidebar.markdown("## 🔧 RAG Settings")
chunk_size    = st.sidebar.slider("Chunk size (tokens)", 200, 1000, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap",         0,  200, 100, step=10)
top_k         = st.sidebar.slider("Top-K results",         1,    8,   3)

st.sidebar.markdown("---")

# ── Reset Session ────────────────────────────────────
st.sidebar.markdown("Reset Session")
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.vector_store = None
    st.session_state.loaded_files = []
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")

# ── Session Stats ────────────────────────────────────
st.sidebar.markdown("## 📈 Session Stats")
q_count = len(st.session_state.chat_history)
st.sidebar.metric("Questions asked", q_count)

# ── Query Tips ───────────────────────────────────────
st.sidebar.markdown("## 💡 Query Tips")
st.sidebar.info(
    "**Summarize** — *'Summarize this document'*\n\n"
    "**Deep dive** — *'What skills are required?'*\n\n"
    "**Analyze** — *'Compare the sections'*"
)

st.sidebar.markdown("---")

# =====================================================
# LOAD MODEL
# =====================================================

if model_choice == "OpenAI (GPT-4.1-mini)":
    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    llm_client   = OpenAI(api_key=OPENAI_API_KEY)
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

elif model_choice == "Local (Llama 3.2 3B)":
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    llm_client    = OllamaLLM(model="llama3.2:3b", num_predict=150)
    embeddings_model = OllamaEmbeddings(model="llama3.2:3b")

else:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    llm_client    = OllamaLLM(model="llama3.2:1b", num_predict=150)
    embeddings_model = OllamaEmbeddings(model="llama3.2:1b")

# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="main-title">📚 AI Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Upload a document. Ask anything. Powered by RAG + LLMs.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <span class="badge">⚡ LLMs</span>
    <span class="badge">🔍 RAG</span>
    <span class="badge">🧠 Embeddings</span>
    <span class="badge">🗄️ FAISS Vector DB</span>
    <span class="badge">🤖 Agentic AI</span>
    <span class="badge">🛠️ Tool Calling</span>
    <span class="badge">📊 Token Tracking</span>
    <span class="badge">💰 Cost Monitoring</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# FILE UPLOAD
# =====================================================

st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop your PDFs here",
    type="pdf",
    accept_multiple_files=True
)

# =====================================================
# PROCESS PDF
# =====================================================
if uploaded_files:

    st.success("✅ PDF uploaded successfully!")

    # READ PDF  
    with st.spinner("📖 Reading PDF..."):
        all_documents = []
        text = ""

        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            file_text = ""

            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    file_text += extracted

            file_text = file_text.replace("\n", " ")
            text += file_text  # accumulate all text for stats + tools

    # DOCUMENT STATS
    st.markdown('<div class="section-header">Document Statistics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Characters", f"{len(text):,}")
    col2.metric("Words",      f"{len(text.split()):,}")

    # per-file breakdown
    if len(uploaded_files) > 1:
        st.markdown('<div class="section-header">Files Loaded</div>', unsafe_allow_html=True)
        for uf in uploaded_files:
            st.caption(f"📄 {uf.name}")

    with st.spinner("✂️ Creating chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_documents = []

        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            file_text = ""

            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    file_text += extracted

            file_text = file_text.replace("\n", " ")
            raw_chunks = text_splitter.split_text(file_text)

            for i, chunk in enumerate(raw_chunks):
                all_documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": uploaded_file.name,
                        "chunk_id": i
                    }
                ))

    st.success(f"✅ {len(all_documents)} chunks created across {len(uploaded_files)} file(s)  (size={chunk_size}, overlap={chunk_overlap})")

    # VECTOR STORE
    # rebuild if uploaded files changed
    uploaded_names = sorted([f.name for f in uploaded_files])

    if st.session_state.get("loaded_files") != uploaded_names:
        with st.spinner("🧠 Building vector database..."):
            st.session_state.vector_store = FAISS.from_documents(all_documents, embeddings_model)
            st.session_state.loaded_files = uploaded_names
            st.session_state.messages = []      # reset memory for new docs
            st.session_state.chat_history = []  # reset chat history
        st.success("✅ FAISS Vector DB created successfully!")
    else:
        st.success(f"✅ FAISS Vector DB loaded — {len(uploaded_files)} file(s) indexed")

    vector_store = st.session_state.vector_store

    # =====================================================
    # RESPONSE FUNCTION
    # =====================================================

    def generate_response(messages, prompt=None):

        # =====================================================
        # BUILD FULL CONVERSATION HISTORY
        # =====================================================

        conversation_history = []

        for msg in st.session_state.messages:

            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # ADD CURRENT REQUEST
        conversation_history.extend(messages)

        start_time = time.time()

        if model_choice == "OpenAI (GPT-4.1-mini)":

            response = llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=conversation_history
            )
            end_time = time.time()
            answer   = response.choices[0].message.content

            st.markdown('<div class="section-header">AI Response</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass-card">{answer}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Model · Tokens · Cost · Performance</div>', unsafe_allow_html=True)
            st.caption(f"Model: {response.model}")

            usage = response.usage
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens",  f"{usage.prompt_tokens:,}")
            col2.metric("Output Tokens", f"{usage.completion_tokens:,}")
            col3.metric("Total Tokens",  f"{usage.total_tokens:,}")

            input_cost  = (usage.prompt_tokens     / 1_000_000) * 0.40
            output_cost = (usage.completion_tokens / 1_000_000) * 1.60
            total_cost  = input_cost + output_cost

            col4, col5 = st.columns(2)
            col4.metric("Estimated Cost",  f"${total_cost:.6f}")
            col5.metric("Response Time",   f"{end_time - start_time:.2f}s")

        else:

            # =====================================================
            # BUILD MEMORY-AWARE PROMPT
            # =====================================================

            memory_prompt = ""

            for msg in st.session_state.messages:

                role = msg["role"].upper()

                memory_prompt += f"{role}: {msg['content']}\n\n"

            # ADD CURRENT PROMPT

            memory_prompt += f"USER: {prompt}\n\nASSISTANT:"

            # =====================================================
            # OLLAMA INVOCATION
            # =====================================================

            answer = llm_client.invoke(memory_prompt)

            end_time = time.time()

            st.markdown('<div class="section-header">AI Response</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass-card">{answer}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Model · Tokens · Cost · Performance</div>', unsafe_allow_html=True)

            model_name = "llama3.2:3b" if model_choice == "Local (Llama 3.2 3B)" else "llama3.2:1b"
            st.caption(f"Model: {model_name} (Ollama — Local CPU)")

            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens",  "N/A")
            col2.metric("Output Tokens", "N/A")
            col3.metric("Total Tokens",  "N/A")

            col4, col5 = st.columns(2)
            col4.metric("Estimated Cost", "$0.00 (Local)")
            col5.metric("Response Time",  f"{end_time - start_time:.2f}s")

        return answer

    # =====================================================
    # TOOLS
    # =====================================================

    def summarize_tool(context):
        context_limit = 4000 if model_choice == "OpenAI (GPT-4.1-mini)" else 1000
        
        messages = [
            {"role": "system", "content": "You are a helpful document summarization assistant. Generate concise and meaningful summaries."},
            {"role": "user",   "content": f"Summarize this document:\n\n{context[:context_limit]}"}
        ]
        prompt = f"""You are a helpful summarization assistant. Summarize this document concisely:{context[:context_limit]}
        Summary:"""
        return generate_response(messages, prompt)

    # =====================================================
    # QUERY REWRITER
    # =====================================================

    def rewrite_query(user_question):

        # ==========================================
        # BUILD RECENT CHAT HISTORY
        # ==========================================

        history_text = ""

        recent_messages = st.session_state.messages[-4:]   #change to 6 or 8 for more tokens which means less amgiguity

        for msg in recent_messages:

            history_text += (
                f"{msg['role'].upper()}: "
                f"{msg['content']}\n"
            )

        # ==========================================
        # HISTORY-AWARE REWRITE PROMPT
        # ==========================================

        rewrite_prompt = f"""
    You are a query rewriting assistant.

    Conversation History:

    {history_text}

    Current Question:

    {user_question}

    Task:

    Convert the current question into a standalone
    semantic search query.

    Rules:
    - Return ONLY the rewritten query.
    - No explanation.
    - No bullet points.
    - No multiple options.
    - Maximum 20 words.
    """

        # ==========================================
        # OPENAI
        # ==========================================

        if model_choice == "OpenAI (GPT-4.1-mini)":

            response = llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": rewrite_prompt
                    }
                ]
            )

            rewritten_query = (
                response
                .choices[0]
                .message
                .content
            )

        # ==========================================
        # OLLAMA
        # ==========================================

        else:

            rewritten_query = llm_client.invoke(
                rewrite_prompt
            )

        return rewritten_query.strip()

    def question_tool(question):
        
        # =====================================================
        # QUERY REWRITING
        # =====================================================

        rewritten_query = rewrite_query(question)

        st.subheader("🔄 Rewritten Query")

        st.code(rewritten_query)

        docs_with_scores = vector_store.similarity_search_with_score(rewritten_query,k=top_k)
        relevant_text = "\n\n".join(doc.page_content for doc, score in docs_with_scores)       

        # =====================================================
        # RETRIEVAL DASHBOARD
        # =====================================================

        st.subheader("📊 Retrieval Dashboard")

        relevant_text = ""

        # collect source + score per doc
        sources = {}

        for idx, (doc, score) in enumerate(docs_with_scores):

            source_name = doc.metadata.get(
                "source",
                "unknown"
            )

            # -----------------------------------------
            # CONVERT SCORE → CONFIDENCE %
            # -----------------------------------------

            confidence = round(
                (1 / (1 + score)) * 100,
                
                1
            )

            # Keep highest confidence per source

            if (
                source_name not in sources
                or confidence > sources[source_name]
            ):

                sources[source_name] = confidence

            # -----------------------------------------
            # CHUNK PREVIEW
            # -----------------------------------------

            chunk_preview = (
                doc.page_content[:500]
            )

            # -----------------------------------------
            # RETRIEVAL CARD
            # -----------------------------------------

            with st.expander(

                f"""
                Chunk {idx + 1}
                |
                {source_name}
                |
                Confidence: {confidence}%
                """

            ):

                st.markdown(
                    f"""
                    ### 📄 Source
                    {source_name}

                    ### 🎯 Confidence
                    {confidence}%

                    ### 🧠 Chunk Preview
                    """
                )

                st.code(chunk_preview)

            # -----------------------------------------
            # BUILD RAG CONTEXT
            # -----------------------------------------

            relevant_text += f'''
            SOURCE: {source_name}

            CONFIDENCE: {confidence}%

            CONTENT:
            {doc.page_content}

            ----------------------------------------
            '''
        
        messages = [
        {"role": "system", "content": "You are a RAG assistant. Answer ONLY from the provided context. If unavailable, say: 'Information not found in document.'"},
        {"role": "user",   "content": f"Context:\n{relevant_text}\n\nQuestion:\n{question}"}]
    
        prompt = f"""You are a RAG assistant.Answer ONLY from the provided context.If unavailable, say: 'Information not found in document.'Context: {relevant_text}
                    Question: {question}
                    Answer:"""        

        answer = generate_response(messages, prompt)
        
        # show sources below response
        if sources:
            sources_html = " &nbsp;|&nbsp; ".join(
                f'<span style="background:rgba(249,115,22,0.12); border:1px solid rgba(249,115,22,0.35); '
                f'color:#fdba74; padding:3px 10px; border-radius:999px; font-size:0.75rem;">'
                f'📄 {s} &nbsp;<span style="color:#2dd4bf; font-weight:600;">{score}%</span></span>'
                for s, score in sources.items()
            )
            st.markdown(
                f'<div style="margin-top:0.5rem;">Sources: {sources_html}</div>',
                unsafe_allow_html=True
            )

        return answer

    def compare_tool(user_input):

        # ==========================================
        # QUERY REWRITING
        # ==========================================

        # rewritten_query = rewrite_query(user_input)

        # st.subheader("🔄 Rewritten Query")

        # st.code(rewritten_query)

        # ==========================================
        # RETRIEVE RELEVANT CHUNKS
        # ==========================================

        all_docs = list(
            vector_store.docstore._dict.values()
        )

        source_context = {}

        for doc in all_docs:

            source = doc.metadata.get(
                "source",
                "unknown"
            )

            if source not in source_context:
                source_context[source] = []

            source_context[source].append(
                doc.page_content
            )
        
        st.subheader("📂 Documents Found")

        st.write(
            list(source_context.keys())
        )

        # ==========================================
        # COMPARISON DASHBOARD
        # ==========================================

        st.subheader("📊 Comparison Dashboard")

        for source, chunks in source_context.items():

            with st.expander(
                f"{source} | {len(chunks)} chunks"
            ):

                preview = chunks[0][:500]

                st.code(preview)

        # ==========================================
        # BUILD STRUCTURED CONTEXT
        # ==========================================

        comparison_context = ""

        document_names = []

        for source, chunks in source_context.items():

            document_names.append(source)

            comparison_context += f"""

        DOCUMENT NAME:
        {source}

        DOCUMENT CONTENT:
        {' '.join(chunks[:5])}

        =================================================

        """

        # ==========================================
        # COMPARISON PROMPT
        # ==========================================

        messages = [

            {
    "role": "system",
    "content": """
You are an expert document comparison assistant.

You will receive multiple documents.

Each document follows this format:

DOCUMENT NAME:
<filename>

DOCUMENT CONTENT:
<content>

Your tasks:

1. Identify the documents being compared.
2. Summarize each document.
3. Highlight similarities.
4. Highlight differences.
5. Identify gaps.
6. Provide recommendations.

IMPORTANT:
- Always mention document names.
- Use only the provided content.
- Never say documents are missing if document content is present.
"""
},

            {
    "role": "user",
    "content": f"""
Documents Available:

{', '.join(document_names)}

Document Content:

{comparison_context}

Comparison Request:

{user_input}
"""
}
        ]

        return generate_response(
            messages,
            prompt=user_input
        )

    # =====================================================
    # AGENT ROUTER
    # =====================================================

    def agent_router(user_input):

        lower = user_input.lower()

        # -----------------------------------------
        # SUMMARIZATION
        # -----------------------------------------

        if "summarize" in lower:
            return "summarize"

        # -----------------------------------------
        # COMPARISON
        # -----------------------------------------

        compare_keywords = [
            "compare",
            "difference",
            "differences",
            "similarity",
            "similarities",
            "versus",
            "vs",
            "match",
            "fit",
            "gap",
            "gaps",
            "analyze"
        ]

        if any(
            keyword in lower
            for keyword in compare_keywords
        ):
            return "compare"

        # -----------------------------------------
        # DEFAULT QUESTION ANSWERING
        # -----------------------------------------

        return "question"

    # =====================================================
    # USER INPUT
    # =====================================================

    st.markdown('<div class="section-header">Ask Your Question</div>', unsafe_allow_html=True)

    user_input = st.text_input(
        "Enter your prompt",
        placeholder="e.g. Summarize this document / What are the required skills?"
    )

    # =====================================================
    # AGENT EXECUTION
    # =====================================================

    if user_input:

        # STORE USER MESSAGE
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        decision = agent_router(user_input)
        st.markdown(f'<div class="agent-tag"> Agent selected tool: <strong>{decision}</strong></div>', unsafe_allow_html=True)

        with st.spinner("Agents are currently working..."):

            if decision == "summarize":
                result = summarize_tool(text)
            elif decision == "compare":
                result = compare_tool(user_input)
            else:
                result = question_tool(user_input)

            st.session_state.chat_history.append({
                "question": user_input,
                "answer":   result,
                "tool":     decision
            })

            # STORE AI RESPONSE FOR MEMORY

            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

            if decision == "question":
                st.caption(f"🔍 Answer generated using top-{top_k} similar chunks.")

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    if st.session_state.chat_history:

        st.markdown('<div class="section-header">💬 Chat History</div>', unsafe_allow_html=True)

        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['question']}"):
                st.caption(f"🛠️ Tool used: {chat['tool']}")
                st.write(chat["answer"])

# =====================================================
# NO FILE STATE
# =====================================================

else:

    st.markdown("""
    <div style="text-align: center; padding: 0.75rem 0 0.5rem; color: #5a3f30; font-size: 0.82rem; letter-spacing: 0.3px;">
        Supports any PDF — resumes, research papers, reports, JDs
    </div>
    """, unsafe_allow_html=True)
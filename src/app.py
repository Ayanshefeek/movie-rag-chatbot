import streamlit as st
from search import RAGSearch
import sys
import os
sys.path.append(os.path.dirname(__file__))
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineSearch AI",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --border:    #2a2a3a;
    --gold:      #c9a84c;
    --gold-dim:  #8a6f2e;
    --text:      #e8e6df;
    --muted:     #6b6878;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

.hero {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.75rem;
    font-weight: 300;
}
.hero-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.5rem auto 0;
}

[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }
[data-testid="stTextInput"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.bubble-user {
    background: var(--gold-dim);
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.92rem;
    color: var(--text);
    animation: fadeUp 0.3s ease both;
}
.bubble-bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.92rem;
    color: var(--text);
    line-height: 1.75;
    animation: fadeUp 0.3s ease both;
}
.bubble-label {
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1.25rem 0;
    justify-content: center;
}
.chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.35rem 0.85rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.02em;
}

.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.stats-bar {
    display: flex;
    gap: 2rem;
    justify-content: center;
    margin: 1.5rem 0 0;
}
.stat { text-align: center; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
}

[data-testid="stSpinner"] p {
    color: var(--gold) !important;
    font-size: 0.85rem !important;
}

[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 2px !important;
    padding: 0.3rem 0.9rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load RAG ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    return RAGSearch()

rag = load_rag()

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · Powered · Discovery</div>
    <h1 class="hero-title">Cine<span>Search</span></h1>
    <p class="hero-sub">Ask anything about movies — powered by RAG & LLaMA 3.3</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Stats ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat"><div class="stat-num">16K+</div><div class="stat-label">Movies</div></div>
    <div class="stat"><div class="stat-num">RAG</div><div class="stat-label">Retrieval</div></div>
    <div class="stat"><div class="stat-num">70B</div><div class="stat-label">Parameters</div></div>
</div>
""", unsafe_allow_html=True)

# ── Suggestion chips ───────────────────────────────────────────────────────────
st.markdown("""
<div class="chips">
    <div class="chip">🎬 Classic 90s films</div>
    <div class="chip">⭐ Top rated dramas</div>
    <div class="chip">🚀 Sci-fi adventures</div>
    <div class="chip">🎭 Christopher Nolan</div>
    <div class="chip">💀 Best horror films</div>
</div>
""", unsafe_allow_html=True)

# ── Chat history display ───────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="bubble-user">
                <div class="bubble-label">You</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bubble-bot">
                <div class="bubble-label">CineSearch</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

# ── Input & buttons ────────────────────────────────────────────────────────────
query = st.text_input(
    "Your Question",
    placeholder="e.g. Best thriller movies with plot twists...",
)

col1, col2 = st.columns([1, 5])
with col1:
    search = st.button("Search →")
with col2:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Handle query ───────────────────────────────────────────────────────────────
if search and query.strip():
    with st.spinner("Searching the archive..."):
        answer = rag.chat(query.strip(), st.session_state.chat_history, top_k=5)

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": query.strip()})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
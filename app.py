"""
Enhanced Medibot: Full Integration (Streamlit + Groq + FAISS)
Author: Saif+Abdul
Description:
  - Uses FAISS vectorstore with sentence-transformers MiniLM embeddings.
  - Runs Groq LLaMA 4 Maverick model for retrieval-augmented Q&A.
  - Displays a clean two-column chat interface (left: chat, right: sources).
  - Uses ConversationalRetrievalChain to maintain chat history as context.
streamlit run app.py
"""

import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_groq import ChatGroq
import torch
import textwrap
import time
from translation_class import TranslatorPipeline

# --------------------------- CONFIG ---------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"

with open("config.json", "r") as f:
    config = json.load(f)


# --------------------------- HELPERS ---------------------------

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say you don't know — do not make up information.
    Only use what is in the provided context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])


def create_conversational_chain():
    """Creates and returns a ConversationalRetrievalChain with chat history support."""
    vectorstore = get_vectorstore()
    
    # Initialize the language model
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        groq_api_key=config["GROQ_API_KEY"],
    )
    
    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": set_custom_prompt()},
        verbose=False
    )
    
    return qa_chain


def call_conversational_chain(query: str, chat_history: list):
    """Calls the conversational chain with chat history and returns the result."""
    qa_chain = create_conversational_chain()
    
    # Convert Streamlit chat history to LangChain format
    langchain_chat_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "bot":
            langchain_chat_history.append(AIMessage(content=msg["content"]))
    
    # Call the chain with chat history
    response = qa_chain.invoke({
        "question": query, 
        "chat_history": langchain_chat_history
    })
    
    answer = response["answer"]
    source_docs = response["source_documents"]

    # Format sources for sidebar
    sources = []
    for i, doc in enumerate(source_docs):
        src = {
            "title": f"Document {i+1}",
            "snippet": textwrap.shorten(doc.page_content, width=300, placeholder="..."),
            "page": doc.metadata.get("page_label") or doc.metadata.get("page"),
            "doc_id": doc.metadata.get("source"),
        }
        sources.append(src)

    return {"answer": answer, "sources": sources}


def gpu_available():
    if torch is None:
        return False
    return torch.cuda.is_available()


# --------------------------- UI STYLING ---------------------------

# st.set_page_config(page_title="Medibot — AI Medical Assistant", layout="wide")
st.set_page_config(page_title="MediBot - AI Health Assistant", page_icon="🤖", layout="wide") #. 🤖 💊 🧬
st.markdown("""
<style>
.chat-container{max-width:1100px;margin:auto}
.user-bubble{background:#0b5cff;color:white;padding:12px;border-radius:12px;display:inline-block;}
.bot-bubble{background:#f1f3f5;color:#111;padding:12px;border-radius:12px;display:inline-block;border:1px solid #e6e9ef}
.user-row{display:flex;justify-content:flex-end;padding:6px}
.bot-row{display:flex;justify-content:flex-start;padding:6px}
.small{font-size:13px;color:#6b7280}
.logo{height:52px;width:52px;border-radius:10px;background:linear-gradient(135deg,#0b5cff,#6a00f4);display:flex;align-items:center;justify-content:center;color:white;font-weight:700}
</style>
""", unsafe_allow_html=True)

# Initialize translator only once per session
if 'translator' not in st.session_state:
    st.session_state.translator = TranslatorPipeline()
    st.session_state.user_lang = None


# --------------------------- HEADER ---------------------------

c1, c2 = st.columns([0.1, 0.9])
with c1:
    from PIL import Image
    image = Image.open("LOGO.png")
    st.image(image, use_container_width=True)
    # st.markdown('<div class="logo">MB</div>', unsafe_allow_html=True)
with c2:
    st.markdown("## MediAssist — Medical Chat Assistant")
    st.caption("Retrieval-Augmented Medical Q&A using GALE Encyclopedia + Groq LLaMA-4 (with Chat History)")

st.write("---")

# --------------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    gpu_status = gpu_available()
    if gpu_status:
        st.success("✅ GPU available")
    else:
        st.warning("⚠️ GPU not detected — running on CPU")

    st.markdown("---")
    st.markdown("**Powered by:** Groq API + FAISS + HuggingFace Embeddings")
    
    # Optional: Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "bot", "content": "Hi, I'm Medibot 👋 — Ask a medical question."}
        ]
        st.rerun()


# --------------------------- MAIN CHAT ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hi, I'm Medibot 👋 — Ask a medical question."}
    ]

chat_col, src_col = st.columns([0.7, 0.3])

with chat_col:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-row'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-row'><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_area("Ask your medical question:", placeholder="e.g. What are canker sores and how to treat them?", height=90)
        user_query_ori = user_query
        if user_query:
            # Detect language only once
            if st.session_state.user_lang is None:
                st.session_state.user_lang = st.session_state.translator.detect_language(user_query, config["DETECT_LANG_KEY"])

            # Translate user question to English (for RAG)
            if st.session_state.user_lang!="en":
                user_query = st.session_state.translator.translate_to_english(user_query)

        submitted = st.form_submit_button("Ask Medibot")


    if submitted and user_query:
        # 1️⃣ Immediately store and display user message (original text)
        st.session_state.messages.append({"role": "user", "content": user_query_ori})
        st.rerun()  # force immediate refresh so user message shows instantly

    # After rerun, Streamlit reloads and shows the message, now we handle the bot response below

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_query_ori = st.session_state.messages[-1]["content"]

        # Detect and translate
        if st.session_state.user_lang is None:
            st.session_state.user_lang = st.session_state.translator.detect_language(user_query_ori, config["DETECT_LANG_KEY"])
        user_query = st.session_state.translator.translate_to_english(user_query_ori) if st.session_state.user_lang != "en" else user_query_ori

        # Display "Thinking..." message temporarily
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<div class='bot-row'><div class='bot-bubble'>🤖 Thinking...</div></div>", unsafe_allow_html=True)

        try:
            chat_history = st.session_state.messages[:-1]
            resp = call_conversational_chain(user_query, chat_history)

            if st.session_state.user_lang != "en":
                answer = st.session_state.translator.translate_from_english(resp["answer"], st.session_state.user_lang)
                sources = resp["sources"]  # Keep untranslated by default
            else:
                answer = resp["answer"]
                sources = resp["sources"]

            placeholder.empty()
            st.session_state.messages.append({"role": "bot", "content": answer, "sources": sources})
            st.rerun()

        except Exception as e:
            placeholder.empty()
            st.error(f"Error: {e}")


with src_col:
    st.header("📚 Sources")

    recent_bot = None
    for m in reversed(st.session_state.messages):
        if m.get("role") == "bot" and m.get("sources"):
            recent_bot = m
            break

    if not recent_bot:
        st.info("Sources will appear here after Medibot answers.")
    else:
        # Add toggle to enable/disable translation
        translate_sources = st.toggle("🔄 Translate Sources", value=False)

        for s in recent_bot["sources"]:
            title = s.get("title", "Document")
            snippet = s.get("snippet", "")
            page = s.get("page")
            doc_id = s.get("doc_id")

            # Translate snippet only if toggle is ON and user's language is not English
            if translate_sources and st.session_state.user_lang != "en":
                try:
                    # Ensure snippet length within limit
                    snippet_text = snippet[:4800]
                    snippet_translated = st.session_state.translator.translate_from_english(
                        snippet_text,
                        st.session_state.user_lang
                    )
                except Exception as e:
                    snippet_translated = f"⚠️ Translation error: {e}"
                snippet_display = snippet_translated
            else:
                snippet_display = snippet

            # Display source
            with st.expander(f"{title} (page {page})" if page else title, expanded=False):
                st.markdown(snippet_display)
                if doc_id:
                    st.caption(f"Doc ID: `{doc_id}`")


st.write("---")
st.caption("Medibot © 2025 | Built with LangChain, Groq, HuggingFace, and Streamlit at PSU.")

import streamlit as st
import os
from Loader import Loader
from Store_And_Retrieve import ChromaDBHandler
from LLM_Image_Summarizer import llm
from Chain_Setup import Chain_Setup
from Embedders import embedder
from PromptTemplate import prompt
from langchain_core.output_parsers import StrOutputParser
from Text_Splitter import TextSplitter

# ---------------------
# Config
# ---------------------
UPLOAD_DIR = r"Z:\Genai_Projects\MultiModal_RAG_Assistant\Knowledge_Base"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # create if not exists

st.set_page_config(page_title="📚 Multimodal Chatbot", layout="wide")
st.title("🤖 Multimodal Chatbot")

parser = StrOutputParser()

# ---------------------
# Load documents from Knowledge Base
# ---------------------
docs = Loader().docs  # Loader class should return docs list
texts=TextSplitter(docs).split_text()

if not docs:
    st.warning("❌ No documents loaded. Please check KnowledgeBase/")
else:
    # Store in Chroma
    ChromaDBHandler.store_in_chroma(texts)
    retriever = ChromaDBHandler.get_retriever()
    Rag_chain = Chain_Setup(retriever, prompt, llm, parser)

# ---------------------
# Chat history
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    st.chat_message(role).write(content)

# ---------------------
# User input
# ---------------------
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # Generate response using RAG chain
    if retriever:
        response = Rag_chain.run(user_input)
        st.session_state.messages.append(("assistant", response))
        st.chat_message("assistant").write(response)

# ---------------------
# Sidebar: Upload files
# ---------------------
with st.sidebar:
    st.subheader("➕ Add Documents / Images")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["png", "jpg", "jpeg", "pdf", "txt", "md"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Save file to Knowledge Base
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved file to: {file_path}")

        # Reload Loader for new files
        new_docs = Loader().docs
        if new_docs:
            ChromaDBHandler.store_in_chroma(new_docs)
            retriever = ChromaDBHandler.get_retriever()
            Rag_chain = Chain_Setup(retriever, prompt, llm, parser)
            st.info("📂 Knowledge base updated with new documents!")
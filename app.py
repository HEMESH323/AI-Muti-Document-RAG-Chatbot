import streamlit as st
import os
import tempfile
from google import genai

from src.document_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.chatbot import ChatbotManager
from src.memory import MemoryManager
from src.utils import logger

# ✅ Configure Gemini API (ONLY ONCE - FIXED)


# Page configuration
st.set_page_config(
    page_title="AI Multi-Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-bubble {
        background-color: #1e293b;
    }
    .ai-bubble {
        background-color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ✅ Initialize session state
def initialize_session_state():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


# ✅ Process documents
def process_documents(uploaded_files):
    with st.spinner("Processing documents... This might take a while."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        # 1. Load Documents
        loader = PDFLoader()
        documents = loader.load_pdfs(file_paths)

        # 2. Split Text
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        # 3. Embeddings
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()

        # 4. Vector Store
        vector_store_manager = VectorStoreManager(embeddings)
        vector_store = vector_store_manager.create_vector_store(chunks)

        # 5. Retriever + Memory
        retriever = DocumentRetriever(vector_store).as_retriever()
        memory_manager = MemoryManager()

        # 6. Chatbot
        st.session_state.chatbot = ChatbotManager(
            retriever,
            memory_manager.get_memory()
        )

        st.session_state.processed_files = [f.name for f in uploaded_files]

        st.success("✅ Documents processed successfully!")


# ✅ Main app
def main():
    initialize_session_state()

    st.title("🤖 AI Multi-Document Chatbot")
    st.subheader("Chat with your PDFs using RAG & Memory")

    # 🔍 Debug (optional - remove later)
    st.sidebar.write("API Loaded:", "GOOGLE_API_KEY" in st.secrets)

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process Documents") and uploaded_files:
            process_documents(uploaded_files)

        if st.session_state.processed_files:
            st.write("### Uploaded Files:")
            for file_name in st.session_state.processed_files:
                st.write(f"- {file_name}")

        if st.button("Clear Chat History"):
            if st.session_state.chatbot:
                st.session_state.chatbot.memory.clear()
            st.session_state.messages = []
            st.rerun()

    # Chat UI
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📚 Show Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.chatbot:
            st.error("⚠️ Please upload and process documents first!")
            return

        # User message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = st.session_state.chatbot.ask(prompt)

                    st.markdown(answer)

                    # Extract sources
                    sources = list(set([
                        doc.metadata.get("source", "Unknown")
                        for doc in source_docs
                    ]))

                    if sources:
                        with st.expander("📚 Show Sources"):
                            for source in sources:
                                st.write(f"- {os.path.basename(source)}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [os.path.basename(s) for s in sources]
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(str(e))


if __name__ == "__main__":
    main()

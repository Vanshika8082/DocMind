import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------
PERSIST_DIR = "chroma-db"
REGISTRY_FILE = Path(PERSIST_DIR) / "books.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "openai/gpt-oss-120b"

st.set_page_config(page_title="Book Chat (RAG)", page_icon="📚", layout="wide")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document.\"""",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


# --------------------------------------------------
# Cached resources
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm(temperature: float):
    return init_chat_model(LLM_NAME, model_provider="groq", temperature=temperature)


# --------------------------------------------------
# Book registry (keeps track of which books are indexed)
# --------------------------------------------------
def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def save_registry(registry: dict) -> None:
    Path(PERSIST_DIR).mkdir(exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def get_vectorstore(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


# --------------------------------------------------
# Indexing
# --------------------------------------------------
def index_pdf(uploaded_file, chunk_size: int, chunk_overlap: int) -> str:
    data = uploaded_file.getvalue()
    collection = "book_" + hashlib.md5(data).hexdigest()[:16]

    registry = load_registry()
    if collection in registry:
        return collection  # already indexed

    with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as status:
        # PyPDFLoader needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            st.write("Reading PDF...")
            docs = PyPDFLoader(tmp_path).load()

            st.write(f"Splitting {len(docs)} pages into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(docs)

            st.write(f"Embedding {len(chunks)} chunks (this can take a moment)...")
            Chroma.from_documents(
                documents=chunks,
                embedding=get_embeddings(),
                collection_name=collection,
                persist_directory=PERSIST_DIR,
            )
        finally:
            os.unlink(tmp_path)

        registry[collection] = {
            "name": uploaded_file.name,
            "pages": len(docs),
            "chunks": len(chunks),
        }
        save_registry(registry)
        status.update(label="Book indexed!", state="complete", expanded=False)

    return collection


def delete_book(collection: str) -> None:
    try:
        get_vectorstore(collection).delete_collection()
    except Exception:
        pass
    registry = load_registry()
    registry.pop(collection, None)
    save_registry(registry)
    st.session_state.chats.pop(collection, None)


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {collection: [messages]}
if "active_book" not in st.session_state:
    st.session_state.active_book = None

registry = load_registry()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📚 Your Books")

    st.subheader("Upload a book")
    uploaded = st.file_uploader("PDF file", type=["pdf"])

    with st.expander("Indexing settings"):
        chunk_size = st.slider("Chunk size", 200, 2000, 500, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 50, step=10)

    if st.button("Index book", type="primary", disabled=uploaded is None, use_container_width=True):
        try:
            st.session_state.active_book = index_pdf(uploaded, chunk_size, chunk_overlap)
            st.rerun()
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    st.divider()

    registry = load_registry()
    if registry:
        st.subheader("Chat with")
        ids = list(registry.keys())
        if st.session_state.active_book not in ids:
            st.session_state.active_book = ids[0]

        selected = st.selectbox(
            "Select a book",
            ids,
            index=ids.index(st.session_state.active_book),
            format_func=lambda c: registry[c]["name"],
            label_visibility="collapsed",
        )
        st.session_state.active_book = selected
        info = registry[selected]
        st.caption(f"{info['pages']} pages · {info['chunks']} chunks")

        col1, col2 = st.columns(2)
        if col1.button("Clear chat", use_container_width=True):
            st.session_state.chats[selected] = []
            st.rerun()
        if col2.button("Delete book", use_container_width=True):
            delete_book(selected)
            st.session_state.active_book = None
            st.rerun()
    else:
        st.info("No books yet. Upload a PDF to get started.")

    st.divider()
    st.subheader("Answer settings")
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    fetch_k = st.slider("Candidates (fetch_k)", k, 30, max(10, k))
    lambda_mult = st.slider(
        "MMR diversity (lambda)", 0.0, 1.0, 0.5, step=0.1,
        help="0 = max diversity, 1 = max relevance",
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.2, step=0.1,
        help="Lower values stay closer to the document. Your original script used 0.9, "
             "which is high for strict document Q&A.",
    )

# --------------------------------------------------
# Main chat area
# --------------------------------------------------
st.title("Chat with your book")

active = st.session_state.active_book
if not active or active not in registry:
    st.info("👈 Upload a PDF in the sidebar and click **Index book** to start chatting.")
    st.stop()

st.caption(f"Currently chatting with: **{registry[active]['name']}**")

messages = st.session_state.chats.setdefault(active, [])


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. Page {s['page']}**")
            st.caption(s["text"])


# Replay history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# New question
query = st.chat_input("Ask something about the book...")

if query:
    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            retriever = get_vectorstore(active).as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
            )

            with st.spinner("Searching the book..."):
                docs = retriever.invoke(query)

            context = "\n\n".join(d.page_content for d in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": query})
            llm = get_llm(temperature)

            def stream():
                for chunk in llm.stream(final_prompt):
                    if chunk.content:
                        yield chunk.content

            answer = st.write_stream(stream())

            sources = [
                {
                    "page": int(d.metadata.get("page", -1)) + 1,  # PyPDF pages are 0-indexed
                    "text": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
                }
                for d in docs
            ]
            render_sources(sources)

            messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(f"Something went wrong: {e}")
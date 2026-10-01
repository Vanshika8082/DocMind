from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Load PDF
# --------------------------------------------------

loader = PyPDFLoader("document loaders/gate.pdf")

docs = loader.load()

print(f"Loaded {len(docs)} pages")


# --------------------------------------------------
# 3. Split PDF into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# --------------------------------------------------
# 4. Hugging Face Embedding Model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 5. Create Chroma Vector Database
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Chroma database created successfully!")
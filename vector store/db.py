from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = HuggingFaceEmbeddings()

vector_store = chroma.Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db",
)
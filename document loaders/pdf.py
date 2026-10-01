from langchain_community.document_loaders import PyPDFLoader, PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

data=PyPDFLoader("document loaders/gate.pdf")
docs=data.load()

# print(docs)

chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()

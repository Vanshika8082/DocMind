
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# ----CHARACTER BASED TEXT SPLITTER----
from langchain_text_splitters import CharacterTextSplitter
splitter=CharacterTextSplitter(
    separator = "",
    chunk_size=100,
    chunk_overlap=1
    ) 

#load the txt file 
file_path = Path(__file__).parent / "notes.txt"

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

docs = loader.load()

print("Document loaded successfully!\n")

for doc in docs:
    print(doc.page_content)


# printing the chunks --- CHARACTER BASED TEXT SPLITTER---
chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()
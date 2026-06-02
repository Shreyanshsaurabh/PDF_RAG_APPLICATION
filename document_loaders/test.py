from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
spliter=CharacterTextSplitter(
    separator="\n",
    chunk_size=10, chunk_overlap=1
    )
loader = TextLoader("document_loaders/AI.txt").load()
docs = spliter.split_documents(loader)

for i in docs:
    print(i.page_content)
    print("\n")
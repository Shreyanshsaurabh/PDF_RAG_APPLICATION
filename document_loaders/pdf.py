from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
data= PyPDFLoader("document_loaders/Gpsych.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs = text_splitter.split_documents(data)
print(len(docs))
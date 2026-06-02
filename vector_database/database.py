from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs=[
    Document(page_content="This is a test document.", metadata={"source": "test.txt"}),
    Document(page_content="This is another test document.", metadata={"source": "test2.txt"}),
    Document(page_content="This is yet another test document.", metadata={"source": "test3.txt"})
]
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="chroma_db"
)

result = vectorstore.similarity_search("test document", k=2) #cant invoke retriver directly on vectorstore? why do i need to create a retriver object?
retriver = vectorstore.as_retriever()
result = retriver.invoke("test document")
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_DIR = "chroma_db"

st.set_page_config(page_title="PDF RAG System", layout="wide")
st.title("📚 PDF RAG System with Mistral AI")

# --- SIDEBAR: File Upload ---
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing PDF and generating embeddings..."):
                try:
                    # We import community components inside the conditional block
                    # to prevent startup segmentation faults
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain_community.vectorstores import Chroma
                    
                    temp_dir = "temp_loaders"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    split_docs = splitter.split_documents(docs)
                    clean_docs = [d for d in split_docs if d.page_content.strip()]
                    
                    embedding_model = MistralAIEmbeddings(model="mistral-embed")
                    
                    # Clear existing vector store if any to prevent collision
                    vectorstore = Chroma.from_documents(
                        documents=clean_docs,
                        embedding=embedding_model,
                        persist_directory=DB_DIR
                    )
                    
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"Successfully processed {uploaded_file.name}!")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- MAIN INTERFACE: Chat System ---
if "processed_file" in st.session_state:
    st.info(f"Active Document: `{st.session_state.processed_file}`")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        
        embedding_model = MistralAIEmbeddings(model="mistral-embed")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 10}
        )
        
        llm = ChatMistralAI(model="mistral-small-2506")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the question. If you don't know the answer, say you don't know."),
            ("human", "context: {context}\n\nquestion: {question}"),
        ])
        
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Searching..."):
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                formatted_prompt = prompt.format(context=context, question=query)
                response = llm.invoke(formatted_prompt)
                
                st.markdown("### Answer")
                st.write(response.content)
    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.warning("Please upload a PDF file in the sidebar to initialize the RAG system.")
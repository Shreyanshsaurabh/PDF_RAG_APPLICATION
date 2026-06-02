import os
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# Configuration
# ----------------------------
load_dotenv()
st.set_page_config(
    page_title="PDF RAG System",
    layout="wide"
)

st.title("📚 PDF RAG System with Mistral AI")

# ----------------------------
# API KEY
# ----------------------------
MISTRAL_API_KEY = st.secrets.get(
    "MISTRAL_API_KEY",
    os.getenv("MISTRAL_API_KEY")
)

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY not found. "
        "Add it to Streamlit Secrets or your local .env file."
    )
    st.stop()

# ----------------------------
# Embedding Model
# ----------------------------
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)

# ----------------------------
# Sidebar Upload
# ----------------------------
with st.sidebar:
    st.header("📄 Document Setup")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if (
            "processed_file" not in st.session_state
            or st.session_state.processed_file != uploaded_file.name
        ):

            with st.spinner("Processing PDF..."):

                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain_community.vectorstores import Chroma

                    temp_dir = "temp_loaders"
                    os.makedirs(temp_dir, exist_ok=True)

                    temp_path = os.path.join(
                        temp_dir,
                        uploaded_file.name
                    )

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()

                    # Split text
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    split_docs = splitter.split_documents(docs)

                    clean_docs = [
                        doc
                        for doc in split_docs
                        if doc.page_content.strip()
                    ]

                    # Create vector database
               vectorstore = st.session_state.vectorstore

st.session_state.vectorstore = vectorstore
st.session_state.processed_file = uploaded_file.name

                    st.success(
                        f"Successfully processed {uploaded_file.name}"
                    )

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    st.exception(e)

# ----------------------------
# Main Chat Interface
# ----------------------------
if (
    "processed_file" in st.session_state
    and "vectorstore" in st.session_state
):

    st.info(
        f"Active Document: {st.session_state.processed_file}"
    )

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate

        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10
            }
        )

        llm = ChatMistralAI(
            model="mistral-small-2506",
            api_key=MISTRAL_API_KEY,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant.
Use ONLY the provided context.
If the answer is not in the context,
say 'I don't know based on the document.'"""
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{question}"
                ),
            ]
        )

        query = st.text_input(
            "Ask a question about the document"
        )

        if query:

            with st.spinner("Searching..."):

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                chain_input = prompt.format(
                    context=context,
                    question=query
                )

                response = llm.invoke(chain_input)

                st.markdown("### Answer")
                st.write(response.content)

                with st.expander("Retrieved Context"):
                    st.write(context)

    except Exception as e:
        st.exception(e)

else:
    st.warning(
        "Upload a PDF from the sidebar to start."
    )

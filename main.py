from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2} #similairity on 10 then return top 2 with mmr
)

llm = ChatMistralAI(model="mistral-small-2506")

#prompt_template

prompt=ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Use the following context to answer the question. If you don't know the answer, say you don't know."),
     ("human", "context:{context} question:{question}"),]
    )

print("RAG system is ready. You can ask questions now.")
print("Type 0 to quit.")
while True:
    query = input("Enter your question: ")
    if query == "0":
        print("Exiting...")
        break
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    response = llm.invoke(prompt.format(context=context, question=query))
    print("Answer:", response.content)
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.title("RAG App Demonstration")


urls = [
    "https://www.kempinski.com/en/hotel-villa-rosa/overview",
    "https://www.kempinski.com/en/hotel-villa-rosa/overview/hotel-information/our-hotel",
    "https://www.kempinski.com/en/about-us"
]

try:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    st.success("✅ URLs loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading URLs: {e}")
    st.stop()


try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    st.success(f"✅ Split documents into {len(docs)} chunks")
except Exception as e:
    st.error(f"❌ Error splitting documents: {e}")
    st.stop()


try:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    st.success("✅ Vector database created with embeddings")
except Exception as e:
    st.error(f"❌ Error creating vectorstore: {e}")
    st.stop()


try:
    llm = GoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.1,
        max_output_tokens=512
    )
    st.success("✅ Gemini LLM initialized")
except Exception as e:
    st.error(f"❌ Error initializing Gemini LLM: {e}")
    st.stop()


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


query = st.text_input("Ask me anything about Kempinski Hotel Villa Rosa", placeholder="Type your question here...")

if query:
    try:
        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        st.subheader("Answer:")
        st.write(response["answer"])
    except Exception as e:
        st.error(f"❌ Error during RAG pipeline execution: {e}")

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF + Chat History")
st.write("Upload a PDF and ask questions about its content.")

api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:

        temp_path = "temp.pdf"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )
        #When history_aware_retriever runs, two things happen internally.First, it uses the LLM only to rewrite the question, not to answer it.Second, this rewritten question is passed to the retriever (your Chroma vector DB retriever). The retriever then searches embeddings and returns relevant document chunks.Those documents are not the answer. They are just context.history_aware_retriever → only rewrite query + retrieve documents
        #the retrieved documents are automatically inserted into the QA prompt under {context}.
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        #The actual answering happens later in your code inside:
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )
        #The role of question_answer_chain is simply to generate the final answer using the retrieved documents. Retrieval and answering are intentionally separated.It does not retrieve because retrieval is already done by history_aware_retriever.
        #What create_stuff_documents_chain actually does is prepare a step that takes documents returned by the retriever and puts them into the {context} placeholder inside your prompt.

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:

            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("Assistant:", response["answer"])

            st.write("Chat History")
            for msg in session_history.messages:
                st.write(msg)

else:
    st.warning("Please enter the Groq API Key.")
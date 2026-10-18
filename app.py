import os
import hashlib

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------

load_dotenv()


# ---------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Document Chat",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📚 AI Document Chat")

st.write(
    "Upload a book, PDF, or notes and ask questions about it."
)


# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


embedding_model = load_embedding_model()


# ---------------------------------------------------
# LLM
# ---------------------------------------------------

@st.cache_resource
def load_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b"
    )

    return llm


llm = load_llm()


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload your Book / PDF / Notes",
        type=["pdf"]
    )

    st.divider()

    st.header("⚙️ RAG Settings")

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100
    )

    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4
    )

    st.divider()

    if st.session_state.current_file:

        st.success(
            f"📄 {st.session_state.current_file}"
        )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------
# Process uploaded PDF
# ---------------------------------------------------

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    # Save uploaded PDF
    with open(file_path, "wb") as f:

        f.write(uploaded_file.getbuffer())


    # ------------------------------------------------
    # Check whether this is a new file
    # ------------------------------------------------

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    current_file_id = (
        uploaded_file.name + "_" + file_hash
    )


    if st.session_state.current_file != current_file_id:

        with st.spinner(
            "📖 Processing your document..."
        ):

            try:

                # ----------------------------------------
                # 1. Load PDF
                # ----------------------------------------

                loader = PyPDFLoader(
                    file_path
                )

                docs = loader.load()


                # ----------------------------------------
                # 2. Split into chunks
                # ----------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                chunks = splitter.split_documents(
                    docs
                )


                # ----------------------------------------
                # 3. Create unique collection
                # ----------------------------------------

                collection_name = (
                    "document_" + file_hash
                )


                # ----------------------------------------
                # 4. Create embeddings + Chroma
                # ----------------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_db",
                    collection_name=collection_name
                )


                # ----------------------------------------
                # 5. Create retriever
                # ----------------------------------------

                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": top_k,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )


                # ----------------------------------------
                # Save in session
                # ----------------------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.retriever = retriever

                st.session_state.current_file = current_file_id

                st.session_state.messages = []


                # ----------------------------------------
                # Success message
                # ----------------------------------------

                st.success(
                    f"""
                    ✅ Document processed successfully!

                    **File:** {uploaded_file.name}

                    **Pages:** {len(docs)}

                    **Chunks:** {len(chunks)}
                    """
                )


            except Exception as e:

                st.error(
                    f"Error processing document: {str(e)}"
                )


# ---------------------------------------------------
# Chat Section
# ---------------------------------------------------

st.header("💬 Chat with your document")


# ---------------------------------------------------
# Display previous messages
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------

query = st.chat_input(
    "Ask something about your document..."
)


if query:

    # -----------------------------------------------
    # Check if document is uploaded
    # -----------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

        st.stop()


    # -----------------------------------------------
    # Display user question
    # -----------------------------------------------

    with st.chat_message("user"):

        st.markdown(query)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # -----------------------------------------------
    # Retrieve documents
    # -----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching the document..."
        ):

            docs = st.session_state.retriever.invoke(
                query
            )


            # ----------------------------------------
            # Create context
            # ----------------------------------------

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # ----------------------------------------
            # Prompt
            # ----------------------------------------

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are a helpful AI assistant.

Use ONLY the provided context to answer
the question.

Do not use outside knowledge.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."

Give a clear and concise answer.
"""
                    ),
                    (
                        "human",
                        """
Context:

{context}

Question:

{question}
"""
                    )
                ]
            )


            # ----------------------------------------
            # Create prompt
            # ----------------------------------------

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # ----------------------------------------
            # Call LLM
            # ----------------------------------------

            response = llm.invoke(
                final_prompt
            )

            answer = response.content


        # --------------------------------------------
        # Display answer
        # --------------------------------------------

        st.markdown(answer)


    # -----------------------------------------------
    # Store assistant response
    # -----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
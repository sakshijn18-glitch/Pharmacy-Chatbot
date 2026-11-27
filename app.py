import os
import streamlit as st
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Pharma Dictionary Chatbot",
    page_icon="💊",
    layout="wide"
)

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Pharmacy Dictionary.pdf"
GROQ_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5


def get_groq_api_key():
    """Read the key from Streamlit Secrets or the environment; never hard-code it."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    return api_key or os.getenv("GROQ_API_KEY")


# ---------------------------
# LOAD VECTOR DATABASE
# ---------------------------
@st.cache_resource
def initialize_rag():

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud under "
            "Manage app > Settings > Secrets as GROQ_API_KEY = \"your-key\"."
        )

    doc = fitz.open(PDF_PATH)

    pages = []

    for i, page in enumerate(doc):
        pages.append(
            Document(
                page_content=page.get_text(),
                metadata={
                    "page": i + 1,
                    "source": "Pharmacy Dictionary"
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = ChatGroq(
        model_name=GROQ_MODEL,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful pharmacy assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
respond with:

"I don't know."

Context:
{context}

Question:
{question}
"""
    )

    chain = prompt | llm | StrOutputParser()

    return retriever, chain


retriever, chain = initialize_rag()

# ---------------------------
# UI
# ---------------------------
st.title("💊 Pharma Dictionary Chatbot")

st.write(
    "Ask any pharmaceutical term or concept from the Pharmacy Dictionary."
)

question = st.text_input(
    "Enter your question:"
)

if question:

    with st.spinner("Searching..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):

        for doc in docs:
            st.write(
                f"Page {doc.metadata.get('page')}"
            )

import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------
# ✅ 1.  Basic setup
# ---------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Immunization ChatBot", page_icon="🤖")
st.title("📘 Immunization Guidelines Chatbot (Groq RAG)")

VECTOR_DIR = Path("vectorstore_immunization/faiss_index")   # use forward slashes
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------
# ✅ 2.  Load vector store
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=HF_MODEL)
    return FAISS.load_local(str(VECTOR_DIR), emb, allow_dangerous_deserialization=True)

# ---------------------------------------------------------------------
# ✅ 3.  Prompt template
# ---------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering strictly from the provided CONTEXT (excerpts of official guidelines).
- If the answer is not in the context, say you don't know and suggest asking a more specific question.
- Quote short key phrases and include labeled sources as [source: filename p.X].

QUESTION: {question}

CONTEXT:
{context}

Provide a concise, step-by-step answer.
""")

def format_docs(docs):
    chunks = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", None)
        tag = f"[source: {src} p.{page+1}]" if page is not None else f"[source: {src}]"
        chunks.append(f"{d.page_content}\n{tag}")
    return "\n\n".join(chunks)

# ---------------------------------------------------------------------
# ✅ 4.  Build the retrieval + generation chain
# ---------------------------------------------------------------------
if not VECTOR_DIR.exists():
    st.warning("No vector index found. Please run `python ingest.py` first.")
else:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=GROQ_API_KEY)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # -----------------------------------------------------------------
    # ✅ 5.  Streamlit chat UI
    # -----------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask a question from the guidelines…")

    if user_input:
        # show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # model reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = chain.invoke(user_input)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

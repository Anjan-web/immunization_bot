import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # ✅ Correct import

from langchain_groq import ChatGroq

load_dotenv()
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIR = Path("vectorstore_immunization")
VECTOR_DIR.mkdir(exist_ok=True)


def load_pdf(pdf_dir: Path):
    docs = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        for d in loader.load():
            d.metadata["source"] = pdf.name
            docs.append(d)
    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_documents(docs)


def main():
    pdf_dir = Path(r"C:\Users\boyin\Agentic_AI\demouv\.venv\ML_Models\immunization")
    print("📚 Loading PDFs from:", pdf_dir.resolve())
    raw_docs = load_pdf(pdf_dir)
    print(f"Loaded {len(raw_docs)} pages from PDF(s)")

    print("🔪 Splitting into chunks…")
    docs = chunk_docs(raw_docs)
    print(f"Created {len(docs)} chunks")

    # ✅ Correct class and argument
    emb = HuggingFaceEmbeddings(model_name=HF_MODEL)

    print("🧠 Building FAISS index…")
    vs = FAISS.from_documents(docs, emb)

    index_path = VECTOR_DIR / "faiss_index"
    vs.save_local(str(index_path))
    print("✅ Saved index to:", index_path)


if __name__ == "__main__":
    main()

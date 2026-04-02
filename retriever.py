import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory where vector DB will be stored
DB_DIR = "./chroma_db"


# 🔹 Step 1: Load file content
def load_document(file_path):
    """
    Reads a text file and converts it into a LangChain Document object
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return Document(page_content=f.read())


# 🔹 Main function: builds retriever
def build_retriever(file_path):
    """
    Takes file path → creates embeddings → stores in Chroma → returns retriever
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise Exception("File not found!")

    # Step 1: Load document
    document = load_document(file_path)

    # Step 2: Split document into smaller chunks
    # Why? → LLMs work better with smaller context pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # max characters per chunk
        chunk_overlap=100    # overlap for better context continuity
    )

    docs = splitter.split_documents([document])
    print(f"📦 Created {len(docs)} chunks")

    # Step 3: Create embeddings
    # Converts text → numerical vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # lightweight + fast model
    )

    # Step 4: Reset vector DB (fresh run every time)
    shutil.rmtree(DB_DIR, ignore_errors=True)

    # Step 5: Store embeddings in Chroma DB
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("✅ Vector DB ready")

    # Step 6: Return retriever
    # k=3 → returns top 3 most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": 3})
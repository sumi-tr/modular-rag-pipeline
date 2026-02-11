from langchain_community.vectorstores import faiss
from load_docs import LoadDir
from chunk_docs import chunk_documents
from embeddings import get_embedding_model

def build_vector_store(data_path: str, index_path: str = "faiss_index"):
    """
    Builds and saves a FAISS vector store from documents.
    """
    # 1. Load documents
    documents = LoadDir(data_path)
    print(f"Loaded {len(documents)} documents")

    # 2. Chunk dcouments
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Get embedding model
    embedding_model = get_embedding_model()

    # 4. Build FAISS index
    vectorstore = faiss.FAISS.from_documents(
        documents = chunks,
        embedding=embedding_model        
    )

    vectorstore.save_local(index_path)
    print(f"FAISS index is stored at {index_path}")

    return vectorstore

if __name__ == "__main__":
    build_vector_store("data")
# Step: 5
from langchain_community.vectorstores.faiss import FAISS
from embeddings import get_embedding_model
import os

def load_vector_store(index_path = "faiss_index"):
    embedding_model = get_embedding_model()
    if not os.path.exists("faiss_index"):
        raise RuntimeError("FAISS index not found. Run vector_store.py first.")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_docs(query: str, k: int=3):
    vectorstore = load_vector_store()
    docs = vectorstore.similarity_search(query, k=k)
    return docs

if __name__ == "__main__":
    query = "What is ontology?"
    result = retrieve_docs(query)
    for i, doc in enumerate(result, 1):
        print(doc.page_content[:300])
        print("Meta data:", doc.metadata)

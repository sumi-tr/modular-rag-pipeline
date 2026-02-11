# Step: 2
from load_docs import LoadDir
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Create text splitter
def chunk_documents(docs: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunked = text_splitter.split_documents(docs)
    return chunked

if __name__ == "__main__":
    # 1. Load documents
    documents = LoadDir("data")  # data - pdf documents' directory
    print(f"Loaded {len(documents)} documents")
    chunked_docs = chunk_documents(documents)

    print(f"Created {len(chunked_docs)} Chunks")

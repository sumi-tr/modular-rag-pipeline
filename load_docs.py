# Step:1
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def LoadDir(dir_path):

    dir_loader = DirectoryLoader(
        path = dir_path,
        glob = "**/*.pdf",
        loader_cls = PyPDFLoader,
    )
    docs = dir_loader.load()
    return docs

if __name__ == "__main__":
    dir_path = "data"
    docs = LoadDir(dir_path)
    print(docs[0][:100])


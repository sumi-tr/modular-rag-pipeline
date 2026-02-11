from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns an embedding model for converting text to vectors
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": True}
    )
    return embedding_model

if __name__ == "__main__":
    emb_model = get_embedding_model()   

    # Testing
    v1 = emb_model.embed_query("climate change")
    v2 = emb_model.embed_query("global warming")
    print(sum(a*b for a,b in zip(v1, v2)))
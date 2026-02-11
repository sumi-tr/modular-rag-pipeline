# Step: 6
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from embeddings import get_embedding_model


def load_vector_store(index_path = 'faiss_index'):
    embedding_model = get_embedding_model()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings = embedding_model,
        allow_dangerous_deserialization= True
    )
    return vectorstore

def ask_question(query, k=3):
    # Load vector store
    vectorstore = load_vector_store()

    # Retrieve relevant docs (chunks)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k)

    for doc, score in docs_and_scores:
        print(f"Score: {score}")
        print(doc.page_content[:300])

    context = "\n\n".join([ doc.page_content for doc, _ in docs_and_scores])

    # Build prompt

    system_prompt = (
        "You are a an assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not present, say 'I don't know'."
    )

    # Call LLM
  
    llm = ChatOllama(
    model="phi3:mini",
    temperature=0
    )

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = f"Context:\n {context} \n\n Question: \n {query}")
    ]

    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = ask_question(question)
    print(answer)



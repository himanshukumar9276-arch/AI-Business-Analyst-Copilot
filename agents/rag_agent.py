from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from agents.model import llm

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    folder_path=str(INDEX_PATH),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, reply exactly:
"I couldn't find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
""")

def rag_agent(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = llm.invoke(messages)

    return response.content
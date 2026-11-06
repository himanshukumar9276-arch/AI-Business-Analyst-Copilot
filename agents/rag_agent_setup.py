from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from agents.model import llm


loader = PyPDFDirectoryLoader("data/docs")
documents = loader.load()

print(f"Loaded {len(documents)} pages")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Created embedding model")

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

vectorstore.save_local("data/faiss_index")


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


question = "What is the refund policy?"


docs = retriever.invoke(question)

print(f"Retrieved {len(docs)} documents")

for i, doc in enumerate(docs):
    print(f"\n----- Document {i+1} -----")
    print(doc.page_content[:500])

context = "\n\n".join(
    doc.page_content for doc in docs
)


answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, reply:

"I couldn't find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
""")


messages = answer_prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

answer = llm.invoke(messages).content

print(answer)










# cd "C:\Users\pranj\Desktop\deep learning\project"
# python rag_agent.py
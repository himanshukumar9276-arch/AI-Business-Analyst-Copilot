import config

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    api_key=config.GROQ_API_KEY, 
    temperature=0.0 ,
)


import os
from dotenv import load_dotenv

# Load sensitive keys from a hidden .env file so they don't get leaked
load_dotenv()

# --- API KEYS & CREDENTIALS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enables reliable LangSmith tracing

# --- MODEL CONFIGURATIONS ---
# Remove the '#' from the lines below if you want to use them in your code
# DEFAULT_MODEL = "claude-3-5-sonnet"
# TEMPERATURE = 0.7
# MAX_TOKENS = 4096

# --- APPLICATION SETTINGS ---
DEBUG = True
LOG_LEVEL = "INFO"
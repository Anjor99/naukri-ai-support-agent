from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        self.mock_llm = os.getenv("MOCK_LLM", "true").lower() == "true"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "")
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
        
# Create a single instance of Settings to be used throughout the application
settings = Settings()
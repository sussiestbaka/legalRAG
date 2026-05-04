# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from the same folder

class Config:
    GROQ_API_KEY = os.getenv("grokAPIKeyGenAIClass")
    
    # Model settings
    LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.0"))
    
    # Retrieval settings
    SIMILARITY_K = int(os.getenv("RAG_SIMILARITY_K", "8"))
    
    # Agent settings
    MAX_AGENT_ITERATIONS = int(os.getenv("RAG_MAX_ITERATIONS", "5"))
    
    # GOOGLE_API_KEY
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Debug
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    # Persist.py config
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR")
    TRACKER_FILE = os.getenv("TRACKER_FILE")
    # Embedder.py config
    db_embedder_model_name = os.getenv("db_embedder_model_name")
    db_chunking_model_name = os.getenv("db_chunking_model_name")

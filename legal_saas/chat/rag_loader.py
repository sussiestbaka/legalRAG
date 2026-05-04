import os
import sys
from pathlib import Path

# Absolute path to your legalRAG project root (where ingestion/, faiss_index/, article_index.json live)
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # goes up to legalRAG/

# Add BASE_DIR to Python path so that 'ingestion' and 'agentic_rag' can be imported
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Change working directory to BASE_DIR so that relative paths (like in config) work
os.chdir(BASE_DIR)

# Load environment variables from the correct .env file
from dotenv import load_dotenv
dotenv_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Now we can import config and other local modules
import config
from ingestion.persist import load_index
from agentic_rag import doRAG

# Optional: Override config paths if they are relative
if hasattr(config, 'FAISS_INDEX_PATH') and not os.path.isabs(config.FAISS_INDEX_PATH):
    config.FAISS_INDEX_PATH = str(BASE_DIR / config.FAISS_INDEX_PATH)
    print(f"Adjusted FAISS_INDEX_PATH to {config.FAISS_INDEX_PATH}")

# Load FAISS index
vectordb = load_index()
if vectordb is None:
    # Try to load directly using absolute path as fallback
    from langchain_community.vectorstores import FAISS
    from ingestion.embedder import get_db_embedder
    embeddings = get_db_embedder()
    faiss_path = BASE_DIR / "faiss_index"
    if faiss_path.exists():
        vectordb = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS directly from {faiss_path}")
    else:
        raise RuntimeError(f"FAISS index not found at {faiss_path}")

# Load article index
import json
article_index_path = BASE_DIR / 'article_index.json'
if not article_index_path.exists():
    raise FileNotFoundError(f"Article index not found at {article_index_path}")
with open(article_index_path, 'r', encoding='utf-8') as f:
    article_index = json.load(f)

print(f"✅ vectordb loaded: {type(vectordb)}")
print(f"✅ article_index has {len(article_index)} entries")
from langchain_huggingface import HuggingFaceEmbeddings

# Cache models to avoid reloading repeatedly
_chunking_embedder = None
_db_embedder = None

def get_chunking_embedder():
    """Returns embedder for semantic chunking (lightweight)."""
    global _chunking_embedder
    if _chunking_embedder is None:
        _chunking_embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    return _chunking_embedder

def get_db_embedder():
    """Returns embedder for vector database (more powerful)."""
    global _db_embedder
    if _db_embedder is None:
        _db_embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    return _db_embedder
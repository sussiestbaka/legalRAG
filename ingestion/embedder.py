from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from config import Config



@lru_cache(maxsize=2)
def get_chunking_embedder():
    return HuggingFaceEmbeddings(model_name=Config.db_chunking_model_name)

@lru_cache(maxsize=1)
def get_db_embedder():
    return HuggingFaceEmbeddings(
        model_name=Config.db_embedder_model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
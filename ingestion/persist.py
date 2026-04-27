import os
import json
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from ingestion.chunker import tokeniseChunking
from ingestion.embedder import get_db_embedder

FAISS_INDEX_DIR = "faiss_index"
TRACKER_FILE = "ingested_files.json"

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_tracker(tracker):
    with open(TRACKER_FILE, "w") as f:
        json.dump(list(tracker), f)

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_index():
    embedding = get_db_embedder()
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, embedding, allow_dangerous_deserialization=True)
    return None

def save_index(vectordb):
    vectordb.save_local(FAISS_INDEX_DIR)

def add_document(file_path, vectordb=None):
    """
    Adds a PDF to the vector store.
    Returns (updated_vectordb, was_duplicate)
    """
    tracker = load_tracker()
    file_hash = get_file_hash(file_path)
    if file_hash in tracker:
        return vectordb, True

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    chunks = tokeniseChunking(pages)
    embedding = get_db_embedder()

    if vectordb is None:
        vectordb = FAISS.from_documents(chunks, embedding)
    else:
        vectordb.add_documents(chunks, embedding)

    save_index(vectordb)
    tracker.add(file_hash)
    save_tracker(tracker)
    return vectordb, False
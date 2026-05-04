import os
import json
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from ingestion.embedder import get_db_embedder
from ingestion.chunker import tokeniseChunkingWithMetadata
from .chunker_cy import hierarchicalChunking
from config import Config

'''
CHUNKING_STRATEGIES = {
    "token":        tokeniseChunkingWithMetadata,
    "recursive":    hierarchialRecursiveChunking,
    "article":      articleAwareChunking,
    "semantic":     semanticChunking,
    "hierarchical": hierarchicalChunking,   # <-- new
}
'''

FAISS_INDEX_DIR = Config.FAISS_INDEX_DIR
TRACKER_FILE = Config.TRACKER_FILE
DEBUG = Config.DEBUG


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
    tracker = load_tracker()
    file_hash = get_file_hash(file_path)
    if file_hash in tracker:
        return vectordb, True
    if DEBUG: print("[ADD_DOC] Loading PDF...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    if DEBUG: print(f"[ADD_DOC] {len(pages)} pages loaded, chunking...")
    chunks = hierarchicalChunking(pages)
    article_index = load_article_index() 
    new_index = build_article_index(chunks) 
    for art, entries in new_index.items():
        article_index.setdefault(art, []).extend(entries)
    save_article_index(article_index)
    
    if DEBUG:print(f"[ADD_DOC] {len(chunks)} chunks, embedding into FAISS...")
    if DEBUG:
        for chunk in chunks[:20]:
            print(f"[CHUNK] articulo={chunk.metadata.get('articulo')} | text={chunk.page_content[:80]}")
    embedding = get_db_embedder()
    if vectordb is None:
        vectordb = FAISS.from_documents(chunks, embedding)
    else:
        vectordb.add_documents(chunks)
    if DEBUG:print("[ADD_DOC] Saving...")
    save_index(vectordb)
    tracker.add(file_hash)
    save_tracker(tracker)
    if DEBUG:print("[ADD_DOC] Done.")
    return vectordb, False

ARTICLE_INDEX_FILE = "article_index.json"

def build_article_index(chunks):
    """Crea un dict: { str(articulo): [ {"text": texto, "source": source, ...}, ... ] }"""
    index = {}
    for chunk in chunks:
        art = chunk.metadata.get("articulo")
        if art is not None:
            art_str = str(art)
            entry = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                # puedes guardar más metadata si quieres
            }
            index.setdefault(art_str, []).append(entry)
    return index

def save_article_index(index):
    with open(ARTICLE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def load_article_index():
    if os.path.exists(ARTICLE_INDEX_FILE):
        with open(ARTICLE_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
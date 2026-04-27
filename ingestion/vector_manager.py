# ingestion/vector_manager.py

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingestion.chunker import tokeniseChunking
from ingestion.embedder import get_db_embedder

# ===================== CONFIGURATION =====================
# You can override these by setting environment variables or changing here
VECTOR_STORE_DIR = Path("data/vector_store")
METADATA_DB_PATH = Path("data/metadata.db")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ===================== METADATA DATABASE =====================
def init_metadata_db(db_path: Path = METADATA_DB_PATH) -> sqlite3.Connection:
    """Create SQLite tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            law_name TEXT,
            version TEXT,
            ingestion_date TEXT NOT NULL,
            faiss_ids TEXT,          -- JSON list of integers
            chunk_count INTEGER
        )
    """)
    # Optional: create index on file_hash for faster duplicate checking
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")
    conn.commit()
    return conn

def get_file_hash(file_path: str | Path) -> str:
    """Compute SHA‑256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def is_duplicate(file_hash: str, db_path: Path = METADATA_DB_PATH) -> bool:
    """Check if a file hash already exists in metadata DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM documents WHERE file_hash = ?", (file_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def record_ingested_file(
    file_hash: str,
    file_path: str | Path,
    law_name: Optional[str],
    version: Optional[str],
    faiss_ids: List[int],
    db_path: Path = METADATA_DB_PATH
) -> None:
    """Insert a record for a successfully ingested file."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (file_hash, file_path, law_name, version, ingestion_date, faiss_ids, chunk_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        file_hash,
        str(file_path),
        law_name,
        version,
        datetime.utcnow().isoformat(),
        json.dumps(faiss_ids),
        len(faiss_ids)
    ))
    conn.commit()
    conn.close()

# ===================== FAISS PERSISTENCE =====================
def load_index(embedding_model=None) -> FAISS:
    """
    Load FAISS index from disk. If not found, create an empty one.
    embedding_model is used only when creating a new index; if None, uses get_db_embedder().
    """
    if embedding_model is None:
        embedding_model = get_db_embedder()

    index_path = VECTOR_STORE_DIR / "index.faiss"
    if index_path.exists():
        # FAISS.load_local expects both .faiss and .pkl files
        return FAISS.load_local(str(VECTOR_STORE_DIR), embedding_model, allow_dangerous_deserialization=True)
    else:
        return FAISS.from_documents([], embedding_model)

def save_index(index: FAISS) -> None:
    """Persist FAISS index to disk."""
    index.save_local(str(VECTOR_STORE_DIR))

# ===================== MAIN INGESTION FUNCTION =====================
def add_documents(
    pages: List[Document],
    file_path: str | Path,
    law_name: Optional[str] = None,
    version: Optional[str] = None,
    index: Optional[FAISS] = None,
    db_conn: Optional[sqlite3.Connection] = None
) -> Tuple[FAISS, List[int]]:
    """
    Chunk, embed, and add documents to the vector store.
    Returns (updated_index, list_of_faiss_ids).
    """
    # 1. Duplicate check
    file_hash = get_file_hash(file_path)
    if is_duplicate(file_hash):
        raise ValueError(f"File {file_path} has already been ingested (hash: {file_hash}). Skipping.")

    # 2. Chunk the pages
    chunks = tokeniseChunking(pages)   # uses SentenceTransformersTokenTextSplitter

    if not chunks:
        raise ValueError("No chunks extracted from the document.")

    
    if index is None:
        index = load_index()

    embedding_model = get_db_embedder()
   
    before_count = index.index.ntotal if hasattr(index, 'index') else 0
    index.add_documents(documents=chunks, embedding=embedding_model)
    after_count = index.index.ntotal
    new_ids = list(range(before_count, after_count))

    # 5. Record metadata
    record_ingested_file(file_hash, file_path, law_name, version, new_ids)

    # 6. Save index to disk
    save_index(index)

    return index, new_ids

# ===================== CONVENIENCE FUNCTIONS =====================
def get_index() -> FAISS:
    """Load and return the current FAISS index (useful for retrieval)."""
    return load_index()

def list_ingested_files() -> List[dict]:
    """Return a list of all ingested file metadata (for debugging)."""
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, law_name, version, ingestion_date, chunk_count FROM documents ORDER BY ingestion_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from ingestion.embedder import get_chunking_embedder
from langchain_core.documents import Document
import re
import os
'''
FUNCTIONS DEFINITIONS
lenghtWiseChunking(pages)
    → Uses CharacterTextSplitter (fixed-length chunks)

hierarchialRecursiveChunking(pages)
    → Uses RecursiveCharacterTextSplitter (structure-aware splitting)

semanticChunking(pages)
    → Uses SemanticChunker (embedding-based splits)

tokeniseChunking(pages)
    → Uses SentenceTransformersTokenTextSplitter (token-based chunks)
'''


def lenghtWiseChunking(pages):
    textSplitterLength = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
    )
    chunks = textSplitterLength.split_documents(pages)
    return chunks

def hierarchialRecursiveChunking(pages):
    textSplitterPara = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators = ["\n\n", "\n", " ", ""]
    )
    chunks = textSplitterPara.split_documents(pages)
    return chunks

def semanticChunking(pages):
    embeddings = get_chunking_embedder()
    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(pages)
    return chunks

def tokeniseChunking(pages):
    tokenTextSplitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=20)
    chunks = tokenTextSplitter.split_documents(pages)
    return chunks

def tokeniseChunkingWithMetadata(pages):
    tokenTextSplitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = tokenTextSplitter.split_documents(pages)
    
    current_article = None
    for i, chunk in enumerate(chunks):  # ← add i here
        chunk.metadata["chunk_index"] = i  # ← add this
        match = re.search(r'art[ií]culo\s+(\d{1,4})\b', chunk.page_content, re.IGNORECASE)
        if match:
            current_article = match.group(1)
            chunk.metadata["articulo"] = current_article
            source_name = os.path.basename(chunk.metadata.get("source", ""))
            chunk.page_content = f"[{source_name}] Artículo {current_article}: {chunk.page_content}"
        else:
            if current_article is None:  # before any article is found
                source_name = os.path.basename(chunk.metadata.get("source", ""))
                chunk.page_content = f"[{source_name}] {chunk.page_content}"
            # if current_article exists, chunk is a continuation - keep tag from previous
            elif current_article:
                chunk.metadata["articulo"] = current_article
                source_name = os.path.basename(chunk.metadata.get("source", ""))
                chunk.page_content = f"[{source_name}] Artículo {current_article}: {chunk.page_content}"
    
    return chunks


# ── Ordinal words used in Mexican law headers ──────────────────────────────
ORDINALS = (
    r'(?:preliminar|[úu]nico|[úu]nica|'
    r'primero|segundo|tercero|cuarto|quinto|'
    r'sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo|'
    r'primera|segunda|tercera|cuarta|quinta|'
    r'sexta|s[eé]ptima|octava|novena|d[eé]cima)'
)

# Roman numerals up to ~50
ROMAN = (
    r'(?:X{0,3}'
    r'(?:IX|IV|V?I{0,3})'
    r')'
)

# ── Header patterns ────────────────────────────────────────────────────────
HEADER_PATTERNS = {
    'libro':    re.compile(
        rf'(?im)^LIBRO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b', re.UNICODE),
    'titulo':   re.compile(
        rf'(?im)^T[ÍI]TULO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b', re.UNICODE),
    'capitulo': re.compile(
        rf'(?im)^CAP[ÍI]TULO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b', re.UNICODE),
    'seccion':  re.compile(
        rf'(?im)^SECCI[ÓO]N\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b', re.UNICODE),
    'articulo': re.compile(
        r'(?im)^ART[ÍI]CULO\s{1,3}(\d{1,4})\s*[.\-°]*', re.UNICODE),
}

# Noise to strip before processing
NOISE = re.compile(
    r'\((?:REFORMADO|ADICIONADO|DEROGADO|REFORMADA)[^\)]*\)',
    re.IGNORECASE
)

def _clean(text):
    """Remove reform notes and stray page numbers."""
    text = NOISE.sub('', text)
    # Remove standalone page numbers (a digit alone on a line)
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)
    return text.strip()

def _find_all_headers(text):
    """Find all structural headers with their positions and types."""
    headers = []
    for level, pattern in HEADER_PATTERNS.items():
        for m in pattern.finditer(text):
            headers.append({
                'pos':   m.start(),
                'level': level,
                'value': m.group(1).strip()
            })
    # Sort by position in document
    headers.sort(key=lambda h: h['pos'])
    return headers


def hierarchicalChunking(pages):
    """
    Splits legal documents into chunks that are aware of their structural
    hierarchy: Libro > Título > Capítulo > Sección > Artículo.
    Each chunk carries full ancestry metadata and has the hierarchy
    baked into page_content for semantic search.
    """
    source = pages[0].metadata.get('source', '') if pages else ''
    source_name = os.path.basename(source)

    # Join all pages and clean noise
    full_text = _clean('\n'.join(p.page_content for p in pages))

    headers = _find_all_headers(full_text)

    if not headers:
        # No structure detected — fall back to token chunking
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=384, chunk_overlap=50)
        docs = splitter.create_documents([full_text])
        for i, doc in enumerate(docs):
            doc.metadata['source'] = source
            doc.metadata['chunk_type'] = 'unstructured'
            doc.metadata['chunk_index'] = i  # ← add this
        return docs

    # ── Build chunks from header boundaries ───────────────────────────────
    chunks = []
    # Track current ancestors
    ancestry = {
        'libro': None,
        'titulo': None,
        'capitulo': None,
        'seccion': None,
        'articulo': None,
    }
    LEVELS = ['libro', 'titulo', 'capitulo', 'seccion', 'articulo']

    for i, header in enumerate(headers):
        level  = header['level']
        value  = header['value']
        start  = header['pos']
        end    = headers[i + 1]['pos'] if i + 1 < len(headers) else len(full_text)

        # Update ancestry — reset all children when a parent changes
        level_idx = LEVELS.index(level)
        ancestry[level] = value
        for child in LEVELS[level_idx + 1:]:
            ancestry[child] = None

        content = full_text[start:end].strip()
        if not content:
            continue

        # Build a human-readable prefix so the embedding captures hierarchy
        breadcrumb_parts = []
        if ancestry['libro']:
            breadcrumb_parts.append(f"Libro {ancestry['libro']}")
        if ancestry['titulo']:
            breadcrumb_parts.append(f"Título {ancestry['titulo']}")
        if ancestry['capitulo']:
            breadcrumb_parts.append(f"Capítulo {ancestry['capitulo']}")
        if ancestry['seccion']:
            breadcrumb_parts.append(f"Sección {ancestry['seccion']}")
        if ancestry['articulo']:
            breadcrumb_parts.append(f"Artículo {ancestry['articulo']}")

        breadcrumb = ' > '.join(breadcrumb_parts)
        page_content = f"[{source_name}] {breadcrumb}:\n{content}"

        meta = {
            'source':    source,
            'chunk_type': level,
            'libro':     ancestry['libro'],
            'titulo':    ancestry['titulo'],
            'capitulo':  ancestry['capitulo'],
            'seccion':   ancestry['seccion'],
            'articulo':  ancestry['articulo'],
        }

        # If article chunk is too long for the embedder, split it further
        if level == 'articulo' and len(page_content.split()) > 300:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_size=384, chunk_overlap=50)
            sub_docs = splitter.create_documents([page_content])
            for sub in sub_docs:
                sub.metadata = meta.copy()
                sub.metadata['chunk_type'] = 'articulo_fragment'
            chunks.extend(sub_docs)
        else:
            chunks.append(Document(page_content=page_content, metadata=meta))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
# cython: boundscheck=False, wraparound=False, cdivision=True
import re
import os
import langchain_text_splitters
import langchain_core.documents

# ── Module‑level regex objects ────────────────────────────────────────────
cdef object ORDINALS, ROMAN, HEADER_PATTERNS, NOISE

ORDINALS = (
    r'(?:preliminar|[úu]nico|[úu]nica|'
    r'primero|segundo|tercero|cuarto|quinto|'
    r'sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo|'
    r'primera|segunda|tercera|cuarta|quinta|'
    r'sexta|s[eé]ptima|octava|novena|d[eé]cima)'
)

ROMAN = r'(?:X{0,3}(?:IX|IV|V?I{0,3}))'

HEADER_PATTERNS = {
    'libro':    re.compile(rf'(?im)^LIBRO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b'),
    'titulo':   re.compile(rf'(?im)^T[ÍI]TULO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b'),
    'capitulo': re.compile(rf'(?im)^CAP[ÍI]TULO\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b'),
    'seccion':  re.compile(rf'(?im)^SECCI[ÓO]N\s+({ROMAN}|\d{{1,2}}|{ORDINALS})\b'),
    'articulo': re.compile(r'(?im)^ART[ÍI]CULO\s{1,3}(\d{1,4})\s*[.\-°]*'),
}

NOISE = re.compile(r'\((?:REFORMADO|ADICIONADO|DEROGADO|REFORMADA)[^\)]*\)',
                   re.IGNORECASE)

# ── Cached token splitter ─────────────────────────────────────────────────
cdef object _cached_token_splitter = None

cdef inline object _get_token_splitter():
    global _cached_token_splitter
    if _cached_token_splitter is None:
        _cached_token_splitter = langchain_text_splitters.SentenceTransformersTokenTextSplitter(
            chunk_size=384, chunk_overlap=50)
    return _cached_token_splitter

# ── The extension type ────────────────────────────────────────────────────
cdef class HierarchicalChunker:
    cpdef str _clean(self, str text):
        cdef str t = NOISE.sub('', text)
        t = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', t)
        return t.strip()

    cdef list _find_all_headers(self, str text):
        cdef list decorated = []
        cdef object pattern, match
        cdef str level, value
        cdef dict header

        for level, pattern in HEADER_PATTERNS.items():
            for match in pattern.finditer(text):
                header = {
                    'pos': match.start(),
                    'level': level,
                    'value': match.group(1).strip()
                }
                decorated.append((header['pos'], header))

        decorated.sort()

        cdef list headers = []
        cdef object pos, hdr
        for pos, hdr in decorated:
            headers.append(hdr)
        return headers

    cpdef list chunk(self, list pages):
        cdef str source = pages[0].metadata.get('source', '') if pages else ''
        cdef str source_name = os.path.basename(source)

        cdef list page_texts = []
        for p in pages:
            page_texts.append(p.page_content)
        cdef str full_text = self._clean('\n'.join(page_texts))

        cdef list headers = self._find_all_headers(full_text)

        cdef list chunks = []
        cdef dict ancestry = {
            'libro': None, 'titulo': None, 'capitulo': None,
            'seccion': None, 'articulo': None
        }
        cdef tuple LEVELS = ('libro', 'titulo', 'capitulo', 'seccion', 'articulo')
        cdef int i, level_idx
        cdef dict header, meta
        cdef str level, value, content, breadcrumb, page_content
        cdef int start, end
        cdef list breadcrumb_parts
        cdef object splitter
        cdef list sub_docs

        if not headers:
            splitter = _get_token_splitter()
            docs = splitter.create_documents([full_text])
            for i, doc in enumerate(docs):
                doc.metadata['source'] = source
                doc.metadata['chunk_type'] = 'unstructured'
                doc.metadata['chunk_index'] = i
            return docs

        cdef dict LEVEL_IDX = {
            'libro': 0, 'titulo': 1, 'capitulo': 2,
            'seccion': 3, 'articulo': 4
        }

        for i, header in enumerate(headers):
            level   = header['level']
            value   = header['value']
            start   = header['pos']
            end     = headers[i + 1]['pos'] if i + 1 < len(headers) else len(full_text)

            level_idx = LEVEL_IDX[level]
            ancestry[level] = value
            for child in LEVELS[level_idx + 1:]:
                ancestry[child] = None

            content = full_text[start:end].strip()
            if not content:
                continue

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

            if level == 'articulo' and len(page_content.split()) > 300:
                splitter = _get_token_splitter()
                sub_docs = splitter.create_documents([page_content])
                for sub_doc in sub_docs:
                    sub_doc.metadata = meta.copy()
                    sub_doc.metadata['chunk_type'] = 'articulo_fragment'
                chunks.extend(sub_docs)
            else:
                chunks.append(langchain_core.documents.Document(
                    page_content=page_content, metadata=meta))

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i

        return chunks


# ── Public drop‑in function ───────────────────────────────────────────────
cdef HierarchicalChunker __chunker = HierarchicalChunker()

def hierarchicalChunking(list pages):
    return __chunker.chunk(pages)
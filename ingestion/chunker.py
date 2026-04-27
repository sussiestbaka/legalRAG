from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from ingestion.embedder import get_chunking_embedder
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

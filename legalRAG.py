from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import gradio as gr
from typing import List
import os

load_dotenv()
GROQ_API_KEY = os.getenv('grokAPIKeyGenAIClass')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

filePath = "/Users/thejas/Desktop/Fall2025/GenAI/Assignment 2 Part 2/CUDA_Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
folderPath = "/Users/thejas/Desktop/Fall2025/GenAI/Assignment 2 Part 2/CUDA_Affiliate_Agreements/"
testAddingNewFilePath = "/Users/thejas/Desktop/Fall2025/GenAI/Assignment 2 Part 2/CUDA_Affiliate_Agreements/LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf"

def buildDocumentDB(folderPath):
    docs=[]
    for file in os.listdir(folderPath):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folderPath,file))
            docs.extend(loader.load())
    return docs
    
def loadingSingleFile(filePath):
    loader = PyPDFLoader(filePath)
    pages=loader.load()
    return pages

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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(pages)
    return chunks

def tokeniseChunking(pages):
    tokenTextSplitter = SentenceTransformersTokenTextSplitter(chunk_size=128, chunk_overlap=20)
    chunks = tokenTextSplitter.split_documents(pages)
    return chunks

def analyzeChunks(chunks, n=3):
    print(f"Total chunks: {len(chunks)}\n")
    for i, c in enumerate(chunks[:n]):
        print(f"--- Chunk {i+1} ---")
        print(f"Text:\n{c.page_content[:300]}...")  # show first 300 chars
        print(f"Metadata: {c.metadata}\n")

def buildVectorDB(folderFilePath, folderFileCheck=True):
    if folderFileCheck:
        pages = buildDocumentDB(folderFilePath)
    else:
        pages = loadingSingleFile(folderFilePath)
    chunks = tokeniseChunking(pages)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vectordb, chunks

def addNewFiletoVectorDB(vectordb, newFilePath =testAddingNewFilePath):
    pages = loadingSingleFile(newFilePath)
    chunks = tokeniseChunking(pages)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb.add_documents(chunks,embedding)
    return vectordb,chunks

def kTopRetriever(vectordb):
    return vectordb.as_retriever(search_kwargs={"k": 8})

def MMRRetriever(vectordb):
    mmrRetriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 32,
        "lambda_mult": 0.5
        },
    )
    return mmrRetriever

def hybridBM25DenseRetriever(chunks, vectordb):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 8
    dense = vectordb.as_retriever(search_kwargs={"k": 8})
    hybridRetriver = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )
    return hybridRetriver

def doRAG(inputQuery,vectordb,chunks):
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_retries=2
    )
    
    promptTemplate = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    
    Context: {context}
    Question: {question}
    """

    #Other prompts I used
    # promptTemplate = """You are a legal contract assistant.  
    # First, explain your reasoning briefly using the provided context.  
    # Then, give a clear and concise final answer.  
    # If the context is insufficient, say "Not enough information."  
    
    # Context:
    # {context}
    
    # Question: {question}
    
    # Reasoning:
    # Final Answer:
    # """

    # promptTemplate = """You are an AI legal assistant.  
    # Answer the question using ONLY the context.  
    # Provide your answer in this structured format:  
    
    # - Key Clause or Section: <name/heading if available>  
    # - Answer: <short summary>  
    # - Source: <document ID or page>  
    
    # Context:
    # {context}
    
    # Question: {question}
    # """
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=promptTemplate)
    
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=hybridBM25DenseRetriever(chunks,vectordb),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    inputQueryDefault = "What does the link plus affiliate agreement about?"
    if inputQuery =="":
        inputQuery = inputQueryDefault
    # Run chain
    result = qa_chain({"query": inputQuery})
    answer = ""
    answer += result["result"] + "\n\n"
    
    # To understand the sources which the RAG retieved
    # answer += "Chunks used:\n"
    # for i, doc in enumerate(result["source_documents"], 1):
    #     answer += f"--- Chunk {i} ---\n"
    #     answer += f"Metadata: {doc.metadata}\n"
    #     answer += f"Content: {doc.page_content[:300]} ...\n\n"

    return answer

def _ask(file_path: str | None, user_msg: str, history, vectordb, chunks):
    history = history or []
    if not file_path:
        return history + [[user_msg or "", "Please upload one document first."]], vectordb, chunks

    if not user_msg or not user_msg.strip():
        return history + [[user_msg or "", "Please enter a question."]], vectordb, chunks

    try:
        if vectordb is None:  # first upload → build DB
            vectordb, chunks = buildVectorDB(file_path, folderFileCheck=False)
        else:
            vectordb, chunks = addNewFiletoVectorDB(vectordb, file_path)
        answer = doRAG(user_msg, vectordb, chunks)
        return history + [[user_msg, answer]], vectordb, chunks
    except Exception as e:
        return history + [[user_msg, f"Error while answering: {e}"]], vectordb, chunks


def _clear():
    return None, [], ""

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("Legal Document QA\nUpload one document, then ask questions about it.")

    with gr.Row():
        file = gr.File(
            label="Upload one document (PDF)",
            file_count="single",
            type="filepath",              
            file_types=[".pdf"],
        )
        clear_btn = gr.Button("Clear", variant="secondary")

    chat = gr.Chatbot(label="Chat", height=380)
    msg = gr.Textbox(
        label="Your question",
        placeholder="e.g., Summarize section 3…",
        lines=2
    )
    send_btn = gr.Button("Send", variant="primary")
    history_state = gr.State([])
    vectordb_state = gr.State(None)
    chunks_state = gr.State(None)

    send_btn.click(
        _ask, [file, msg, history_state, vectordb_state, chunks_state],
        [chat, vectordb_state, chunks_state]
        ).then(lambda h: h, chat, history_state).then(lambda: "", None, msg)

    msg.submit(
        _ask, [file, msg, history_state, vectordb_state, chunks_state],
        [chat, vectordb_state, chunks_state]
        ).then(lambda h: h, chat, history_state).then(lambda: "", None, msg)


    # Clear all
    clear_btn.click(_clear, None, [file, chat, msg]) \
             .then(lambda: [], None, history_state)
    
if __name__ == "__main__":
    demo.launch()
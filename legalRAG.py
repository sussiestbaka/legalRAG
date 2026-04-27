# legalRAG.py

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from ingestion.embedder import get_db_embedder
from ingestion.chunker import tokeniseChunking
from ingestion.persist import add_document, load_index
import gradio as gr
import os

load_dotenv()
GROQ_API_KEY = os.getenv('grokAPIKeyGenAIClass')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def loadingSingleFile(filePath):
    loader = PyPDFLoader(filePath)
    return loader.load()

def doRAG(inputQuery, vectordb):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5, max_retries=2)
    
    promptTemplate = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    
    Context: {context}
    Question: {question}
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=promptTemplate)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    if not inputQuery.strip():
        inputQuery = "What does the affiliate agreement say?"
    result = qa_chain({"query": inputQuery})
    return result["result"] + "\n\n"

def _ask(file_path, user_msg, history, vectordb):
    history = history or []
    if not file_path:
        return history + [[user_msg or "", "Please upload a PDF."]], vectordb
    if not user_msg or not user_msg.strip():
        return history + [[user_msg or "", "Please enter a question."]], vectordb
    try:
        if vectordb is None:
            vectordb = load_index()
        vectordb, was_dup = add_document(file_path, vectordb)
        if was_dup:
            answer = doRAG(user_msg, vectordb)
            history.append([user_msg, f"[Duplicate file - using existing index]\n\n{answer}"])
        else:
            answer = doRAG(user_msg, vectordb)
            history.append([user_msg, answer])
        return history, vectordb
    except Exception as e:
        history.append([user_msg, f"Error: {e}"])
        return history, vectordb

def _clear():
    return None, []

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("Legal Document QA")
    with gr.Row():
        file = gr.File(label="Upload PDF", file_count="single", type="filepath", file_types=[".pdf"])
        clear_btn = gr.Button("Clear Chat")
    chat = gr.Chatbot(label="Chat", height=380)
    msg = gr.Textbox(label="Your question", lines=2)
    send_btn = gr.Button("Send")
    
    history_state = gr.State([])
    vectordb_state = gr.State(None)
    
    send_btn.click(_ask, [file, msg, history_state, vectordb_state], [chat, vectordb_state])
    msg.submit(_ask, [file, msg, history_state, vectordb_state], [chat, vectordb_state])
    clear_btn.click(_clear, None, [file, chat, msg]).then(lambda: [], None, history_state)

if __name__ == "__main__":
    demo.launch()
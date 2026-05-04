# legalRAG.py

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from ingestion.embedder import get_db_embedder
from ingestion.chunker import tokeniseChunking
from ingestion.persist import add_document, load_index
from langchain_classic.chains import RetrievalQA  # Ajusta según tu import
import gradio as gr
import os
import threading
from langchain_ollama import ChatOllama
from ingestion.persist import load_article_index 
import re
import os

load_dotenv()
GROQ_API_KEY = os.getenv('grokAPIKeyGenAIClass')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Global state
vectordb = None
ingest_status = "Ready"


def ingest_pdf(file_path):
    global vectordb, ingest_status
    print(f"[THREAD] ingest_pdf started, file_path={file_path}, type={type(file_path)}")
    if not file_path:
        ingest_status = "No file selected."
        return
    try:
        ingest_status = "Ingesting..."
        db = load_index()
        db, duplicate = add_document(file_path, db)
        vectordb = db
        print(f"[THREAD] done, vectordb={vectordb}, duplicate={duplicate}")
        ingest_status = "Duplicate skipped." if duplicate else "Ingestion successful."
    except Exception as e:
        import traceback
        ingest_status = f"Ingestion failed: {e}"
        print(f"[THREAD] EXCEPTION: {traceback.format_exc()}")

def doRAG(inputQuery, vectordb):
    article_index = load_article_index()
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.0)
    
    promptTemplate = """Usa el siguiente contexto para responder la pregunta.
    Si no sabes la respuesta, di que no sabes, no inventes.
    Usa máximo tres oraciones. Sé conciso.
    
    Contexto: {context}
    Pregunta: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=promptTemplate)

    # ── Detect specific article request ───────────────────────────────────
    article_match = re.search(r'art[ií]culo\s+(\d+)', inputQuery, re.IGNORECASE)
    target_article = article_match.group(1) if article_match else None

    if target_article and target_article in article_index:
        entries = article_index[target_article]
        context = "\n\n".join([entry["text"] for entry in entries])
        prompt = QA_CHAIN_PROMPT.format(context=context, question=inputQuery)
        answer = llm.invoke(prompt).content
        sources = set()
        for entry in entries:
            src = entry.get("source", "")
            pdf_name = os.path.basename(src) if src else "desconocido"
            sources.add(f"{pdf_name} - Artículo {target_article}")
        answer += "\n\n**Fuente(s):** " + (", ".join(sources) if sources else "*Fuente no disponible*")
        return answer + "\n\n"

    # ── Semantic search ────────────────────────────────────────────────────
    docs_with_scores = vectordb.similarity_search_with_score(inputQuery, k=8)
    
    # Sort by relevance score (lower = more similar in FAISS)
    docs_with_scores.sort(key=lambda x: x[1])
    
    seen_articles = set()
    final_chunks = []

    for doc, score in docs_with_scores:
        if len(final_chunks) >= 15:
            break
            
        art = doc.metadata.get("articulo")
        if art:
            seen_articles.add(str(art))
        final_chunks.append(doc.page_content)

    # ── Enrich with neighbors ──────────────────────────────────────────────
    neighbor_texts = []
    for art_str in list(seen_articles):
        try:
            art_num = int(art_str)
        except ValueError:
            continue
        for neighbor in [art_num - 2, art_num - 1, art_num + 1, art_num + 2]:
            neighbor_str = str(neighbor)
            if neighbor_str not in seen_articles and neighbor_str in article_index:
                seen_articles.add(neighbor_str)  # prevent duplicates
                entries = article_index[neighbor_str]
                neighbor_texts.extend([e["text"] for e in entries])

    # Fill remaining slots with neighbors up to cap of 15
    for text in neighbor_texts:
        if len(final_chunks) >= 15:
            break
        final_chunks.append(text)

    # ── Build context and call LLM ─────────────────────────────────────────
    context = "\n\n".join(final_chunks)
    prompt = QA_CHAIN_PROMPT.format(context=context, question=inputQuery)
    answer = llm.invoke(prompt).content

    # ── Extract sources ────────────────────────────────────────────────────
    sources = set()
    for doc, _ in docs_with_scores:
        match = re.search(r'^\[([^\]]+\.pdf)\](?:.*?Artículo\s+(\d+))?', doc.page_content, re.IGNORECASE)
        if match:
            pdf_name = match.group(1)
            art_num = match.group(2) if match.group(2) else "?"
            sources.add(f"{pdf_name} - Artículo {art_num}")

    answer += "\n\n**Fuente(s):** " + (", ".join(sources) if sources else "*Fuente no disponible*")
    return answer + "\n\n"

def ask_question(user_msg, history):
    global vectordb
    history = history or []
    if not user_msg or not user_msg.strip():
        history.append({"role": "assistant", "content": "Please enter a question."})
        return history
    if vectordb is None:
        history.append({"role": "assistant", "content": "No PDF ingested yet. Please upload and ingest a PDF first."})
    try:
        answer = doRAG(user_msg, vectordb)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": f"Error: {e}"})
    return history

def clear_chat():
    return []

# Gradio UI
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("Legal Document QA – Upload and ingest a PDF, then ask questions")

    with gr.Row():
        file = gr.File(label="PDF", file_count="single", type="filepath", file_types=[".pdf"])
        ingest_btn = gr.Button("Ingest PDF", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False, value="Ready")

    chat = gr.Chatbot(label="Chat", height=380)
    msg = gr.Textbox(label="Your question", lines=2)
    send_btn = gr.Button("Send", variant="secondary")
    clear_btn = gr.Button("Clear Chat")

    # Ingest in background to avoid hang
    def start_ingest(file_path):
        threading.Thread(target=ingest_pdf, args=(file_path,), daemon=True).start()
        return "Ingesting... (please wait)"

    ingest_btn.click(start_ingest, inputs=file, outputs=status_box)


    timer = gr.Timer(value=1, active=True)
    def update_status():
        return gr.update(value=ingest_status)
    timer.tick(update_status, outputs=status_box)

    send_btn.click(ask_question, inputs=[msg, chat], outputs=chat)
    msg.submit(ask_question, inputs=[msg, chat], outputs=chat)
    clear_btn.click(clear_chat, outputs=chat)

if __name__ == "__main__":
    demo.launch()
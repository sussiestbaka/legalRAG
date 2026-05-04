from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from ingestion.embedder import get_db_embedder
from ingestion.chunker import tokeniseChunking
from ingestion.persist import add_document, load_index, load_article_index
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import gradio as gr
import os
import re
import threading
from config import Config

load_dotenv()
GROQ_API_KEY = Config.GROQ_API_KEY


# Global state
vectordb = None
ingest_status = "Ready"

def dprint(a):
    if Config.DEBUG:
        print(a)

def ingest_pdf(file_path):
    global vectordb, ingest_status
    #if not file_path:
        #ingest_status = "No file selected."
       # return
    try:
        ingest_status = "Ingesting..."
        db = load_index()
        db, duplicate = add_document(file_path, db)
        vectordb = db
        ingest_status = "Duplicate skipped." if duplicate else "Ingestion successful."
    except Exception as e:
        import traceback
        ingest_status = f"Ingestion failed: {e}"
        dprint(f"[THREAD] EXCEPTION: {traceback.format_exc()}")


import time

def doRAG(inputQuery, vectordb, history=None):
    inputQuery = inputQuery[:4000] #
    t_start = time.time()
    dprint(f"[doRAG] Start. Question: {inputQuery[:80]}...")

    # ── Model + index init ─────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=Config.GOOGLE_API_KEY
)
    article_index = load_article_index()

    # ── Private helpers (same logic as before) ─────────────────────
    def _rewrite_query(query: str) -> str:
        rewrite_prompt = (
            "Eres un experto legal mexicano. Reescribe la siguiente pregunta "
            "en lenguaje jurídico formal para mejorar la búsqueda en un índice "
            "de leyes y códigos mexicanos.\n\n"
            "REGLAS:\n"
            "- NO inventes nombres de leyes, NOMs, artículos o referencias específicas\n"
            "- NO agregues información que no esté en la pregunta original\n"
            "- Solo reformula con terminología jurídica más precisa\n"
            "- Responde SOLO con la pregunta reescrita, sin explicaciones\n\n"
            f"Pregunta original: {query}"
        )
        return llm.invoke(rewrite_prompt).content.strip()
    def _get_article(num: int, source_hint: str = ""):
        entries = article_index.get(str(num), [])
        if not entries:
            return "Artículo no encontrado.", set()

        sections = []
        for e in entries:
            fname = os.path.basename(e.get("source", "desconocido"))
            sections.append(f"[{fname}]\n{e['text'][:1500]}")

        text = "\n\n---\n\n".join(sections)
        sources = {
            f"{os.path.basename(e.get('source', 'desconocido'))} - Artículo {num}"
            for e in entries
        }
        return f"Artículo {num}:\n\n{text}", sources

    def _semantic_search(query: str, k: int = Config.SIMILARITY_K):
        docs_with_scores = vectordb.similarity_search_with_score(query, k=k)
        if not docs_with_scores:
            return "No se encontraron fragmentos relevantes.", set()
        results, sources = [], set()
        for doc, score in docs_with_scores:
            art = doc.metadata.get("articulo", "?")
            pdf = os.path.basename(doc.metadata.get("source", "desconocido"))
            snippet = doc.page_content[:200].replace("\n", " ")
            results.append(f"[{pdf}] Artículo {art} (score: {score:.4f}): {snippet}...")
            sources.add(f"{pdf} - Artículo {art}")
        return "\n".join(results), sources
    
    def _fetch_more(source: str, query: str, num: int = 0) -> tuple[str, set]:
        # ── Article index lookup (for legal codes with article numbers) ────────
        if num:
            entries = article_index.get(str(num), [])
            for e in entries:
                if source.lower() in os.path.basename(e.get("source", "")).lower():
                    fname = os.path.basename(e.get("source", "desconocido"))
                    return f"[{fname}] Artículo {num}:\n{e['text']}", {f"{fname} - Artículo {num}"}

        # ── Semantic search restricted to this source ──────────────────────────
        docs_with_scores = vectordb.similarity_search_with_score(query, k=50)
        matching = [
            (doc, score) for doc, score in docs_with_scores
            if source.lower() in os.path.basename(doc.metadata.get("source", "")).lower()
        ]

        if not matching:
            return "Entrada no encontrada.", set()

        matching.sort(key=lambda x: x[1])
        best_doc = matching[0][0]
        fname = os.path.basename(best_doc.metadata.get("source", "desconocido"))
        best_idx = best_doc.metadata.get("chunk_index")

        # ── Try positional adjacency first (works for everything with chunk_index) ──
        if best_idx is not None:
            adjacent = [
                doc for doc, _ in matching
                if doc.metadata.get("chunk_index") in [best_idx - 1, best_idx, best_idx + 1]
            ]
            adjacent.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            text = "\n\n---\n\n".join(d.page_content for d in adjacent)

        # ── Fallback: hierarchy siblings (legal codes without chunk_index) ─────
        else:
            best_capitulo = best_doc.metadata.get("capitulo")
            best_seccion = best_doc.metadata.get("seccion")

            if best_capitulo or best_seccion:
                siblings = [
                    doc for doc, _ in matching
                    if (best_seccion and doc.metadata.get("seccion") == best_seccion) or
                    (best_capitulo and doc.metadata.get("capitulo") == best_capitulo)
                ]
            else:
                # No structure at all — just take top 5
                siblings = [doc for doc, _ in matching[:5]]

            text = "\n\n---\n\n".join(d.page_content for d in siblings)

        return f"[{fname}]:\n{text}", {fname}

    # ── Shared source accumulator ──────────────────────────────────
    all_sources: set = set()

    # ── Tool dispatch (plain functions, no LangChain @tool decorator) ─
    # bind_tools / native Groq tool calling is unreliable with LLaMA —
    # the model emits <function=name{...}> instead of valid JSON, causing
    # a 400. We drive the loop with an explicit ReAct prompt instead.
    def _call_tool(name: str, args: dict) -> str:
        if name == "fetch_article":
            raw_num = args.get("article_number", 0)
            try:
                num = int(raw_num)
            except (ValueError, TypeError):
                num = 0
            txt, srcs = _get_article(num)
            all_sources.update(srcs)
            return txt
        if name == "search_documents":
            txt, srcs = _semantic_search(args.get("query", ""))
            all_sources.update(srcs)
            return txt
        if name == "fetch_more":
            raw_num = args.get("article_number", 0)
            try:
                num = int(raw_num)
            except (ValueError, TypeError):
                num = 0
            txt, srcs = _fetch_more(args.get("source", ""), num)
            all_sources.update(srcs)
            return txt

    # ── Conversation memory (last 2 exchanges → ~600 chars) ────────
    chat_context = ""
    '''
    History removal test
    if history:
        recent = history[-4:]
        chat_context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in recent
        )
    '''

    # ── ReAct system prompt ────────────────────────────────────────
    # The LLM outputs ONE JSON object per turn. We parse it, run the
    # tool (or tools in parallel), and feed the result back as a new
    # human message. When it's ready to answer it emits "final_answer".
    REACT_SYSTEM = """Eres un experto legal. Dispones de dos herramientas:

1. fetch_article(article_number): Úsala SOLO cuando el usuario pregunte explícitamente por un artículo con su número (ej. "artículo 45", "artículo 1234", "¿qué dice el artículo 15?").

2. search_documents(query): Úsala para TODAS las demás preguntas: temas generales, conceptos, límites, normas, procedimientos, etc. No importa si la pregunta es sobre leyes, reglamentos o normas técnicas – si no pide un número de artículo específico, usa search_documents.

3. fetch_more(source, article_number?): Úsala cuando search_documents devuelva
   un fragmento relevante pero incompleto.
   - source: nombre del archivo que apareció en los resultados
   - article_number: opcional, omitir si el documento no tiene artículos (ej. NOMs)

   Ejemplo sin artículo: {"tool": "fetch_more", "args": {"source": "NOM-003-SEMARNAT-1997"}}
   Ejemplo con artículo: {"tool": "fetch_more", "args": {"source": "ley-de-aguas-nacionales", "article_number": 124}}
REGLAS:
- En cada iteración llama UNA sola herramienta, nunca múltiples a la vez.
- NUNCA repitas la misma búsqueda o fetch_more dos veces.
- NUNCA inventes números de artículos.
Responde ÚNICAMENTE con JSON válido en uno de estos formatos:
- Para usar herramientas: {"actions": [{"tool": "fetch_article", "args": {"article_number": 123}}, ...]}
- Para respuesta final: {"final_answer": "Tu respuesta aquí"}
"""

    # ── Query rewriting ────────────────────────────────────────────
    rewritten = _rewrite_query(inputQuery)
    dprint(f"[doRAG] Rewritten query: {rewritten}")

    # ── Build conversation ─────────────────────────────────────────
    conversation = [
        SystemMessage(content=REACT_SYSTEM),
        HumanMessage(content=rewritten),
    ]

    # ── ReAct loop ─────────────────────────────────────────────────

    MAX_ITERATIONS = Config.MAX_AGENT_ITERATIONS
    answer = ""

    for iteration in range(MAX_ITERATIONS):
        t0 = time.time()
        dprint(f"[doRAG] Iteration {iteration + 1}")

        raw = llm.invoke(conversation).content.strip()
        dprint(f"[doRAG] LLM call: {time.time()-t0:.1f}s")
        dprint(f"[doRAG] Raw response: {raw[:200]}")

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Model broke format — try to salvage a plain-text answer
            dprint("[doRAG] JSON parse failed, treating as final answer.")
            answer = raw
            break

        # ── Final answer branch ────────────────────────────────────
        if "final_answer" in parsed:
            answer = parsed["final_answer"]
            break

        # ── Tool call branch ───────────────────────────────────────
        actions = parsed.get("actions", [])
        actions = [a for a in actions if "tool" in a]  # ← add this line
        if not actions:
            dprint("[doRAG] No actions and no final_answer — treating as answer.")
            answer = raw
            break

        # Execute all requested tools in parallel
        def _run(action):
            return action, _call_tool(action["tool"], action.get("args", {}))

        t0 = time.time()
        results = [_run(a) for a in actions]
        dprint(f"[doRAG] Tool execution ({len(actions)} tools): {time.time()-t0:.1f}s")
        

        # Append assistant turn + tool results as a single human message
        # so the next LLM call sees what was retrieved
        tool_summary = "\n\n".join(
            f"[{a['tool']}]:\n{res[:800]}" #Change
            for a, res in results
        )
        conversation.append(AIMessage(content=raw))
        conversation.append(HumanMessage(
            content=f"Resultados de las herramientas:\n{tool_summary}\n\n"
                    f"Continúa: usa más herramientas si necesitas más contexto, "
                    f"o entrega la respuesta final."
        ))

    else:
        # Loop exhausted — ask for best-effort answer without tools
        dprint("[doRAG] Max iterations reached — forcing final answer.")
        conversation.append(HumanMessage(
            content="Se agotaron las rondas. Entrega tu mejor respuesta final "
                    "basada en la información ya recopilada. "
                    'Responde con {"final_answer": "..."}.'
        ))
        raw = llm.invoke(conversation).content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")
        try:
            answer = json.loads(raw).get("final_answer", raw)
        except json.JSONDecodeError:
            answer = raw

    # ── Self-critique / grounding check ───────────────────────────
    # Tool results are stored as HumanMessages prefixed with
    # "Resultados de las herramientas:" — collect them all as context.
    tool_context = "\n".join(
        m.content for m in conversation
        if isinstance(m, HumanMessage)
        and m.content.startswith("Resultados de las herramientas:")
    )

    if tool_context.strip():
        critique_prompt = (
            "Revisa si CADA afirmación de esta respuesta está respaldada por el "
            "contexto proporcionado.\n"
            "Si todas las afirmaciones están respaldadas responde exactamente: GROUNDED\n"
            "Si alguna no lo está responde: UNGROUNDED: <lista los puntos no respaldados>\n\n"
            f"Contexto:\n{tool_context[:3000]}\n\n"
            f"Respuesta:\n{answer}"
        )
        t0 = time.time()
        critique = llm.invoke(critique_prompt).content
        dprint(f"[doRAG] Self-critique: {time.time()-t0:.1f}s  |  {critique[:60]}")

        if "UNGROUNDED" in critique:
            answer += (
                "\n\n⚠️ *Algunas afirmaciones podrían no estar directamente "
                "respaldadas por los artículos recuperados.*"
            )

    # ── Append sources ─────────────────────────────────────────────
    if all_sources:
        answer += "\n\n**Fuente(s):** " + ", ".join(sorted(all_sources))

    dprint(f"[doRAG] TOTAL time: {time.time()-t_start:.1f}s")
    return answer


# ── UI functions ───────────────────────────────────────────────────────────
def ask_question(user_msg, history):
    global vectordb
    history = history or []
    if not user_msg or not user_msg.strip():
        history.append({"role": "assistant", "content": "Please enter a question."})
        return history
    if vectordb is None:
        history.append({"role": "assistant", "content": "No PDF ingested yet. Please upload and ingest a PDF first."})
        return history
    try:
        # Pass history so doRAG can use the last few turns as context
        answer = doRAG(user_msg, vectordb, history=None)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": f"Error: {e}"})
    return history

def clear_chat():
    return []


# ── Gradio UI ──────────────────────────────────────────────────────────────
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
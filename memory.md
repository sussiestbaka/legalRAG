Below is a comprehensive **project memory document** you can save as `PROJECT_MEMORY.md` in your `legalRAG/` folder. It captures everything we’ve built, decisions made, and the current working state.  

You can paste this into a new ChatGPT conversation later, or hand it to another developer, to instantly restore context.

---

# Project Memory: Legal QA SaaS (RAG + Django)

**Last updated:** 2026-05-01  
**Status:** Backend API fully functional; chat UI working; allauth login integrated.

## 1. Project Overview
- **Goal:** Provide a legal question‑answering system for the Mexican Civil Code and related documents, using a ReAct agent with retrieval‑augmented generation (RAG).
- **Current state:** Local Django web app with session authentication, FAISS vector store, and a Gradio‑free HTML/JavaScript chat frontend.
- **Long‑term vision:** Scalable SaaS with multi‑tenancy, Google OAuth, subscription pricing (Stripe), and cloud vector database.

## 2. Directory Structure

```
C:\Users\bari\Documents\GitHub\legalRAG/
├── ingestion/                  # Chunking, embedding, FAISS persistence
│   ├── chunker.py
│   ├── embedder.py
│   └── persist.py
├── faiss_index/                # Pre‑built FAISS index (133 MB + 35 MB)
│   ├── index.faiss
│   └── index.pkl
├── article_index.json          # JSON map of article numbers → text + source PDFs
├── agentic_rag.py              # ReAct agent, doRAG() function, tool definitions
├── config.py                   # Reads environment variables (GROQ_API_KEY, paths)
├── .env                        # API keys, config overrides (not committed)
├── legal_saas/                 # Django project root (contains manage.py)
│   ├── manage.py
│   ├── db.sqlite3              # Development database (sessions, users, messages)
│   ├── legal_saas/             # Inner settings module
│   │   ├── settings.py
│   │   └── urls.py
│   ├── chat/                   # Django app for chat functionality
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── admin.py
│   │   └── rag_loader.py       # Loads FAISS & article index at startup
│   └── templates/              # Custom HTML templates
│       ├── account/            # allauth templates
│       │   └── login.html
│       └── home.html           # Main chat interface
└── requirements.txt            # Python dependencies
```

## 3. Environment & Dependencies

**Key packages:**  
- `django==6.0.4`  
- `djangorestframework`  
- `django-allauth`  
- `langchain` , `langchain-community`, `langchain-groq`  
- `faiss-cpu`  
- `python-dotenv`  
- `pypdf`, `tiktoken`  

**Environment variables (`.env`):**
```ini
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama3-70b-8192   # or other Groq model
TEMPERATURE=0.1
SIMILARITY_K=6
MAX_AGENT_ITERATIONS=5
FAISS_INDEX_PATH=faiss_index    # relative to project root
DEBUG=True
```

## 4. RAG Core (`agentic_rag.py`)

- **`doRAG(inputQuery, vectordb, history=None)`**  
  - Implements a ReAct loop with two tools:  
    - `search_documents(query)` – semantic search over FAISS.  
    - `fetch_article(article_number)` – direct lookup from `article_index.json`.  
  - Uses `ChatGroq` (Groq API) as the LLM.  
  - Executes multiple tool calls in parallel with `ThreadPoolExecutor`.  
  - Adds a self‑critique step to check if all claims are grounded.  
  - Returns final answer with sources formatted as `**Fuente(s):** ...`.

- **`article_index.json` structure:**  
  Keys are article numbers (as strings). Each entry is a list of objects:  
  ```json
  {
    "1234": [
      {"text": "Artículo 1234: ...", "source": "codigo-civil-federal.pdf"}
    ]
  }
  ```

- **FAISS index:** Built during ingestion (using `ingestion/` scripts). Stored in `faiss_index/`. Loading uses `load_index()` from `persist.py`, which relies on `config.FAISS_INDEX_PATH`.

## 5. Django Integration

### 5.1 Models (`chat/models.py`)
- `ChatSession` – user, title, created_at.  
- `ChatMessage` – session, role (user/assistant), content, sources (JSONField), timestamp.

### 5.2 Views (`chat/views.py`)
Single endpoint: `POST /api/chat/`  
- Accepts `{ "message": "...", "session_id": optional }`.  
- Requires authentication (session auth).  
- Retrieves last 8 messages (4 exchanges) from DB to build `history`.  
- Calls `doRAG` with those arguments.  
- Saves user message and assistant response.  
- Returns JSON: `{ "answer": "...", "session_id": id, "message_id": id }`.

**Authentication:** `@permission_classes([IsAuthenticated])` + `SessionAuthentication` (default). CSRF protection active.

### 5.3 URL Configuration (`legal_saas/urls.py`)
```python
urlpatterns = [
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('admin/', admin.site.urls),
    path('api/chat/', include('chat.urls')),
    path('accounts/', include('allauth.urls')),
]
```

### 5.4 RAG Loader (`chat/rag_loader.py`)
- Sets `sys.path` and `os.chdir` to project root (`C:/.../legalRAG/`).  
- Loads `.env` from correct location.  
- Imports `load_index()` and falls back to direct FAISS `load_local` if needed.  
- Loads `article_index.json`.  
- Makes `vectordb` and `article_index` available to views as module‑level globals.

### 5.5 Authentication (allauth)
- **Login page:** `templates/account/login.html` (custom styled).  
- **Signup / password reset:** allauth default templates (plain but functional).  
- **Settings in `settings.py`:**  
  - `INSTALLED_APPS` includes `allauth`, `allauth.account`, `allauth.socialaccount`, `django.contrib.sites`.  
  - `SITE_ID = 1`.  
  - `AUTHENTICATION_BACKENDS` includes allauth backend.  
  - `MIDDLEWARE` includes `AccountMiddleware`.  
  - `LOGIN_REDIRECT_URL = '/'`.

### 5.6 Frontend Chat UI (`templates/home.html`)
- Plain HTML/CSS + vanilla JavaScript.  
- Reads CSRF token from cookie.  
- Sends POST to `/api/chat/` with `message` and stored `session_id`.  
- Displays conversation, extracts and shows sources.  
- Remembers session ID across messages.

## 6. Current Working Flow

1. Start Django: `python manage.py runserver` (from `legal_saas/`).  
2. Visit `/accounts/login/`, log in with superuser or newly created user.  
3. Redirected to `/` – chat interface.  
4. Type a question (e.g., “What does Article 1234 say?”).  
5. Backend calls `doRAG`, which:  
   - Uses ReAct to search/fetch articles.  
   - Returns answer with sources.  
6. Frontend displays answer and saves session for context.

**Tested example response:**  
Works correctly, returns information from both Coahuila and Federal Civil Codes, with source attribution.

## 7. Known Issues & Workarounds

- **FAISS index loading:** Must be run from within Django after changing working directory – solved by `rag_loader.py`.  
- **CSRF token in fetch:** Must extract from cookie (handled in `home.html`).  
- **allauth context processor missing:** Removed from `TEMPLATES` to avoid `ModuleNotFoundError`. Not required for basic auth.

## 8. Next Steps (for SaaS production)

- [ ] Switch from SQLite to PostgreSQL.  
- [ ] Implement **Google OAuth** – add `allauth.socialaccount.providers.google`, set up credentials in admin.  
- [ ] Add **Stripe subscription** (`dj-stripe`) – create pricing tiers (free, pro) and enforce monthly request limits in API view.  
- [ ] Build a proper **React frontend** or improve the existing HTML UI with Tailwind CSS.  
- [ ] Replace FAISS with a **cloud vector database** (Pinecone, Qdrant, pgvector) for horizontal scaling.  
- [ ] Containerize with Docker and deploy on a PaaS (Railway, Fly.io, Heroku).  
- [ ] Set up **background tasks** (Celery) for any async ingestion (though uploader is removed – not needed for static corpus).  
- [ ] Add **analytics** (PostHog, Mixpanel) to track usage per user.

## 9. Useful Commands

| Action | Command (from `legal_saas/`) |
|--------|------------------------------|
| Run dev server | `python manage.py runserver` |
| Make migrations | `python manage.py makemigrations chat` |
| Apply migrations | `python manage.py migrate` |
| Create superuser | `python manage.py createsuperuser` |
| Collect static files | `python manage.py collectstatic` |
| Inspect URLs | `python manage.py show_urls` (requires `django-extensions`) |

## 10. Git & Backup

- **Repository:** `https://github.com/yourusername/legalRAG-saas` (private).  
- **Exclude from git:** `.env`, `db.sqlite3`, `__pycache__/`, optionally `faiss_index/` (large).  
- **Backup strategy:** Regularly commit code changes; copy entire folder to external drive or cloud storage.

---

**End of project memory.**  
With this document, a fresh ChatGPT session (or another developer) can resume work immediately.
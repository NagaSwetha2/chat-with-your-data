import io
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

from rag_engine import ask, build_index, documents_from_csv, documents_from_text
from charts import generate_charts
from sf_knowledge import get_sf_docs
import pandas as pd

# Pre-load Salesforce knowledge base at startup
_sf_index = build_index(get_sf_docs())

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Zarva — Chat With Your Data")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://zarva.net",
        "https://www.zarva.net",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_indexes: dict[str, object] = {}
_histories: dict[str, list[dict]] = {}
_status: dict[str, dict] = {}
_dataframes: dict[str, object] = {}
_executor = ThreadPoolExecutor(max_workers=2)

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_INDEXES = 100
MAX_QUESTION_LEN = 1000

_INJECTION_PATTERNS = [
    "drop table", "delete from", "insert into", "update set",
    "exec(", "execute(", "xp_", "--", "/*", "*/", "union select",
    "or 1=1", "or '1'='1", "script>", "<script", "javascript:",
    "base64", "eval(", "system(", "os.system", "__import__",
]

def _is_safe_input(text: str) -> bool:
    lower = text.lower()
    return not any(p in lower for p in _INJECTION_PATTERNS)


class ChatRequest(BaseModel):
    index_id: str
    question: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str


class IndexResponse(BaseModel):
    index_id: str
    status: str


class StatusResponse(BaseModel):
    status: str
    record_count: int | None = None


def _build_in_background(index_id: str, docs: list, df=None):
    try:
        db = build_index(docs)
        _indexes[index_id] = db
        _histories[index_id] = []
        if df is not None:
            _dataframes[index_id] = df
        _status[index_id] = {"status": "ready", "record_count": len(docs)}
    except Exception as e:
        _status[index_id] = {"status": "error", "error": str(e)}


@app.post("/index", response_model=IndexResponse)
@limiter.limit("5/minute")
async def index_file(request: Request, file: UploadFile = File(...)):
    if len(_indexes) >= MAX_INDEXES:
        raise HTTPException(503, "Server is at capacity. Try again later.")

    raw = await file.read()

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 5MB.")

    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(400, "Only .csv or .txt files are supported.")

    if not _is_safe_input(file.filename):
        raise HTTPException(400, "Invalid filename.")

    buffer = io.BytesIO(raw)

    if file.filename.endswith(".csv"):
        docs = documents_from_csv(buffer)
    elif file.filename.endswith(".txt"):
        docs = documents_from_text(buffer)
    else:
        raise HTTPException(400, "Only .csv or .txt files are supported")

    df = None
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))

    index_id = str(uuid.uuid4())
    _status[index_id] = {"status": "indexing", "record_count": len(docs)}
    _executor.submit(_build_in_background, index_id, docs, df)

    return IndexResponse(index_id=index_id, status="indexing")


@app.get("/status/{index_id}", response_model=StatusResponse)
async def get_status(index_id: str):
    s = _status.get(index_id)
    if s is None:
        raise HTTPException(404, "Unknown index_id")
    return StatusResponse(**s)


class SFRequest(BaseModel):
    question: str


@app.post("/sf-chat")
@limiter.limit("30/minute")
async def sf_chat(request: Request, req: SFRequest):
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(400, "Question too long.")
    if not _is_safe_input(req.question):
        raise HTTPException(400, "Invalid input detected.")
    answer = ask(_sf_index, req.question)
    return {"answer": answer}


@app.get("/charts/{index_id}")
async def get_charts(index_id: str):
    df = _dataframes.get(index_id)
    if df is None:
        raise HTTPException(404, "No chart data available for this index.")
    return generate_charts(df)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(400, "Question too long. Max 1000 characters.")

    if not _is_safe_input(req.question):
        raise HTTPException(400, "Invalid input detected. Please ask a normal question.")

    db = _indexes.get(req.index_id)
    if db is None:
        s = _status.get(req.index_id, {})
        if s.get("status") == "indexing":
            raise HTTPException(409, "Index is still being built — please wait")
        raise HTTPException(404, "Unknown index_id — build an index first via /index")

    history = _histories.get(req.index_id, [])
    answer = ask(db, req.question, history=history)

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    _histories[req.index_id] = history[-20:]

    return ChatResponse(answer=answer)


_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))

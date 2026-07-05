import io
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from rag_engine import ask, build_index, documents_from_csv, documents_from_text

app = FastAPI(title="Chat With Your Data API")

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


class ChatRequest(BaseModel):
    index_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str


class IndexResponse(BaseModel):
    index_id: str
    record_count: int


@app.post("/index", response_model=IndexResponse)
async def index_file(file: UploadFile = File(...)):
    raw = await file.read()
    buffer = io.BytesIO(raw)

    if file.filename.endswith(".csv"):
        docs = documents_from_csv(buffer)
    elif file.filename.endswith(".txt"):
        docs = documents_from_text(buffer)
    else:
        raise HTTPException(400, "Only .csv or .txt files are supported")

    db = build_index(docs)
    index_id = str(uuid.uuid4())
    _indexes[index_id] = db

    return IndexResponse(index_id=index_id, record_count=len(docs))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    db = _indexes.get(req.index_id)
    if db is None:
        raise HTTPException(404, "Unknown index_id — build an index first via /index")

    answer = ask(db, req.question)
    return ChatResponse(answer=answer)


_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))

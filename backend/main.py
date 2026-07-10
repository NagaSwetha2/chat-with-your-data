import io
import uuid
import os
import datetime
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

import json, re as _re
import numpy as np

def _to_python(obj):
    """Recursively convert numpy/pandas types to plain Python for JSON serialization."""
    import math
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

from rag_engine import ask, build_index, documents_from_csv, documents_from_text, documents_from_pdf, documents_from_image
from backend.charts import generate_charts, generate_dynamic_chart, generate_dynamic_chart_from_text
from backend.ml_engine import run_ml_predictions
from backend.hypothesis_engine import run_hypotheses, run_doc_hypotheses
from backend.eda_engine import run_eda
from backend.proactive_engine import run_proactive_analysis
from backend.business_context_engine import run_pipeline
from backend.extractor import extract_structured_data
from backend.sf_knowledge import get_sf_docs
import pandas as pd

# SF knowledge base — built lazily on first request to save memory
_sf_index = None

def get_sf_index():
    global _sf_index
    if _sf_index is None:
        _sf_index = build_index(get_sf_docs())
    return _sf_index

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
_docs: dict[str, list] = {}
_suggestions: dict[str, list[str]] = {}
_telemetry: dict[str, list[dict]] = {}   # device_id → last 100 readings
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
    followups: list[str] = []


class IndexResponse(BaseModel):
    index_id: str
    status: str


class StatusResponse(BaseModel):
    status: str
    record_count: int | None = None
    suggestions: list[str] | None = None


def _doc_suggestions(docs: list) -> dict:
    """LLM-driven: reads the actual document and generates relevant question categories."""
    from backend.model_router import get_model
    from langchain_core.output_parsers import StrOutputParser

    text_sample = " ".join(docs[:6])[:2500]

    prompt = f"""You are reading a document that a user just uploaded. Generate exactly 4 categories of questions the user would naturally want to ask about THIS specific document.

Document content:
---
{text_sample}
---

Return ONLY valid JSON. No explanation. Format:
{{
  "Category Name": ["Short question 1?", "Short question 2?", "Short question 3?"],
  "Category Name": ["Short question 1?", "Short question 2?", "Short question 3?"],
  "Category Name": ["Short question 1?", "Short question 2?", "Short question 3?"],
  "Category Name": ["Short question 1?", "Short question 2?", "Short question 3?"]
}}

Rules:
- Category names must match what this document actually contains (e.g. "Flight Info" for a flight booking, "Tax Details" for a tax document, "Ingredients" for a recipe)
- Questions must be answerable from this specific document
- Questions must be short (under 8 words)
- No generic placeholders — use real names, dates, amounts from the document where possible"""

    try:
        llm = get_model("narrate")
        chain = llm | StrOutputParser()
        result = chain.invoke(prompt)
        match = _re.search(r'\{[\s\S]*\}', result)
        if match:
            data = json.loads(match.group())
            # Validate structure
            if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                return {k: v[:3] for k, v in list(data.items())[:4]}
    except Exception:
        pass

    # Fallback — generic but safe
    return {
        "Summary":  ["What is this document about?", "Summarize the key points", "Who is involved?"],
        "Details":  ["What are the main facts?", "List all dates and numbers", "What are the key terms?"],
        "Analysis": ["What is most important here?", "Are there any action items?", "What should I do next?"],
        "Search":   ["Find a specific topic", "What conclusions are drawn?", "List all key names"],
    }


def _pandas_suggestions(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    cat_cols = [c for c in cols if df[c].dtype == object]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower()]

    result = {}

    # Overview — always present
    overview = [f"How many total records?", f"What columns does this data have?"]
    if cat_cols:
        top_val = str(df[cat_cols[0]].value_counts().index[0])[:30]
        overview.append(f"How many unique {cat_cols[0]} values?")
        overview.append(f"Most common {cat_cols[0]}?")
    result["Overview"] = overview[:4]

    # By category column
    if cat_cols:
        col = cat_cols[0]
        top5 = df[col].value_counts().head(5).index.tolist()
        by_cat = [f"Count records per {col}", f"Which {col} has the most records?"]
        if len(top5) >= 2:
            by_cat.append(f"Compare {top5[0]} vs {top5[1]}")
        by_cat.append(f"Show all unique {col} values")
        result[f"By {col}"] = by_cat[:4]

    # Stats — numeric columns
    if num_cols:
        col = num_cols[0]
        stats = [f"Average {col}", f"Highest {col}", f"Lowest {col}", f"Distribution of {col}"]
        result["Statistics"] = stats[:4]
    elif cat_cols:
        col = cat_cols[0]
        result["Statistics"] = [
            f"Count labels starting with A",
            f"Count labels starting with S",
            f"Which label appears most?",
            f"How many distinct labels?",
        ]

    # Trends / time
    if date_cols:
        col = date_cols[0]
        result["Trends"] = [f"Records by month in {col}", f"Peak activity period", f"Growth trend over time", f"Recent vs earlier records"]
    else:
        col = cat_cols[0] if cat_cols else cols[0]
        sample_vals = df[col].dropna().astype(str).unique()[:3].tolist()
        search_prompts = [f"Show top 10 records", f"Summarize all data", f"How many unique {col} values?"]
        if sample_vals:
            search_prompts.insert(0, f"Find records where {col} is '{sample_vals[0]}'")
        result["Search"] = search_prompts[:4]

    return result


def _build_in_background(index_id: str, docs: list, df=None):
    try:
        # ── Generate sidebar suggestions FIRST (fast: pandas = instant, LLM = ~5s) ──
        # This runs before build_index so prompts appear immediately after upload
        try:
            _suggestions[index_id] = (
                _pandas_suggestions(df) if df is not None
                else _doc_suggestions(docs)
            )
        except Exception:
            _suggestions[index_id] = {}

        db = build_index(docs)
        _indexes[index_id] = db
        _histories[index_id] = []
        _docs[index_id] = docs

        if df is not None:
            _dataframes[index_id] = df

        # ── MARK READY so the UI unblocks ──────────────────────────────
        _status[index_id] = {"status": "ready", "record_count": len(docs)}

        # ── PDF/image/text: extract structured DataFrame in background ──
        # Runs after "ready" so it never blocks the user
        if df is None:
            try:
                extracted = extract_structured_data(docs)
                if extracted is not None:
                    _dataframes[index_id] = extracted
            except Exception:
                pass  # fall back to RAG-only — already working

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

    _IMAGE_EXTS = (
        ".jpg", ".jpeg", ".jfif", ".jpe",           # JPEG family
        ".png",                                       # PNG
        ".webp",                                      # WebP
        ".gif",                                       # GIF
        ".bmp", ".dib",                               # BMP
        ".tiff", ".tif",                              # TIFF / multi-page
        ".heic", ".heif",                             # iPhone / HEIF
        ".avif",                                      # AVIF
        ".ico", ".cur",                               # Icons
        ".ppm", ".pgm", ".pbm", ".pnm",              # Netpbm
        ".tga", ".icb", ".vda", ".vst",              # TGA
        ".pcx",                                       # PCX
        ".svg",                                       # SVG (Pillow with cairosvg)
    )
    _ALLOWED_EXTS = (".csv", ".txt", ".pdf") + _IMAGE_EXTS
    if not file.filename.lower().endswith(_ALLOWED_EXTS):
        raise HTTPException(400, "Supported: CSV, PDF, TXT and images (JPG, PNG, WEBP, HEIC, AVIF, GIF, BMP, TIFF, and more)")

    if not _is_safe_input(file.filename):
        raise HTTPException(400, "Invalid filename.")

    buffer = io.BytesIO(raw)

    if file.filename.endswith(".csv"):
        docs, df = documents_from_csv(buffer)
    elif file.filename.endswith(".txt"):
        docs, df = documents_from_text(buffer)
    elif file.filename.lower().endswith(".pdf"):
        docs, df = documents_from_pdf(buffer)
    elif file.filename.lower().endswith(_IMAGE_EXTS):
        docs, df = documents_from_image(raw)
    else:
        raise HTTPException(400, "Supported: .csv, .txt, .pdf, .jpg, .jpeg, .png, .webp, .gif")

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
    result = ask(get_sf_index(), req.question)
    answer = result["answer"] if isinstance(result, dict) else result
    return {"answer": answer, "followups": result.get("followups", []) if isinstance(result, dict) else []}


@app.post("/dynamic-chart/{index_id}")
async def dynamic_chart(index_id: str, req: SFRequest):
    df = _dataframes.get(index_id)
    if df is not None:
        result = generate_dynamic_chart(df, req.question)
        return result
    # PDF/text mode — extract numbers from doc text and chart them
    docs = _docs.get(index_id, [])
    if docs:
        result = generate_dynamic_chart_from_text(docs, req.question)
        return result
    raise HTTPException(404, "No data for this index.")


def _fmt_value(val, unit: str) -> str:
    unit_up = unit.upper()
    try:
        fval = float(val)
        if "$" in unit or "USD" in unit_up or "DOLLAR" in unit_up:
            return f"${fval:,.2f}" if fval != int(fval) else f"${int(fval):,}"
        if "%" in unit:
            return f"{fval:.2f}%"
        return f"{fval:,.2f}" if fval != int(fval) else f"{int(fval):,}"
    except Exception:
        return str(val)


def _unit_icon(unit: str, label: str) -> str:
    u, l = unit.upper(), label.lower()
    if "$" in unit or "USD" in u or "DOLLAR" in u:
        return "💰"
    if "%" in unit:
        return "📊"
    if any(w in l for w in ["tax","withhold","deduct"]):
        return "🏛️"
    if any(w in l for w in ["net","gross","pay","salary","wage","compensation"]):
        return "💵"
    if any(w in l for w in ["benefit","insurance","health","fsa","401","retirement"]):
        return "🏥"
    return "📋"


@app.get("/kpis/{index_id}")
async def get_kpis(index_id: str):
    df = _dataframes.get(index_id)
    if df is None:
        raise HTTPException(404, "No data")

    # Extracted PDF/doc → return semantic cards, not CRM metrics
    if df.attrs.get("is_extracted", False) and "label" in df.columns and "value" in df.columns:
        unit_col = "unit" if "unit" in df.columns else None
        cat_col = "category" if "category" in df.columns else None
        cards = []
        seen = set()
        # Sort by absolute value so the biggest numbers come first
        work = df.copy()
        work["_abs"] = work["value"].abs()
        work = work.sort_values("_abs", ascending=False)
        # One pass per unit group so we don't mix currencies with percentages
        for unit_key in (work[unit_col].dropna().unique().tolist() if unit_col else [""]):
            group = work[work[unit_col] == unit_key] if unit_col else work
            for _, row in group.iterrows():
                label = str(row.get("label", "")).strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                unit = str(row.get(unit_col, "")) if unit_col else ""
                cat = str(row.get(cat_col, "")) if cat_col else ""
                cards.append({
                    "icon": _unit_icon(unit, label),
                    "label": label,
                    "value": _fmt_value(row["value"], unit),
                    "category": cat,
                })
                if len(cards) >= 6:
                    break
            if len(cards) >= 6:
                break
        return _to_python({"cards": cards, "doc_type": df.attrs.get("document_type", "document")})

    # CSV / tabular data → existing CRM-style KPIs
    kpis = {"total_records": len(df)}
    amt_col = next((c for c in df.columns if any(w in c.lower() for w in ["amount","revenue","value","price","cost"])), None)
    if amt_col:
        try:
            vals = pd.to_numeric(df[amt_col].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce")
            kpis["total_revenue"] = f"${vals.sum():,.0f}"
            kpis["avg_deal"] = f"${vals.mean():,.0f}"
        except: pass
    stage_col = next((c for c in df.columns if any(w in c.lower() for w in ["stage","status","state"])), None)
    if stage_col:
        counts = df[stage_col].value_counts()
        kpis["top_stage"] = f"{counts.index[0]} ({counts.iloc[0]})"
        won = df[df[stage_col].str.contains("won|closed|resolved|complete", case=False, na=False)]
        kpis["win_rate"] = f"{len(won)/len(df)*100:.0f}%"
    owner_col = next((c for c in df.columns if any(w in c.lower() for w in ["owner","rep","assignee","engineer","agent"])), None)
    if owner_col:
        top = df[owner_col].value_counts()
        kpis["top_owner"] = f"{top.index[0]} ({top.iloc[0]})"
    text_col = next((c for c in df.columns if c.lower() in ["notes","note","description","comments","text","body"]), None)
    if text_col:
        pos_words = ["excited","fantastic","great","excellent","thrilled","amazing","outstanding","happy","wonderful","positive"]
        pos = df[text_col].fillna("").astype(str).str.lower().apply(lambda t: any(w in t for w in pos_words))
        kpis["positive_pct"] = f"{pos.mean()*100:.0f}%"
    if stage_col and amt_col:
        try:
            stage_prob = {"prospecting":0.15,"discovery":0.30,"proposal":0.50,"negotiation":0.70,"closed won":1.0,"won":1.0,"resolved":1.0,"closed":0.9,"new":0.10,"in progress":0.40}
            df2 = df.copy()
            df2["_amt"] = pd.to_numeric(df2[amt_col].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce")
            df2["_prob"] = df2[stage_col].str.lower().map(lambda s: next((v for k,v in stage_prob.items() if k in s), 0.3))
            forecast = (df2["_amt"] * df2["_prob"]).sum()
            kpis["revenue_forecast"] = f"${forecast:,.0f}"
        except: pass
    return _to_python(kpis)


@app.get("/hypotheses/{index_id}")
async def get_hypotheses(index_id: str):
    df = _dataframes.get(index_id)
    if df is not None:
        return _to_python(run_hypotheses(df))
    docs = _docs.get(index_id)
    if docs:
        return _to_python(run_doc_hypotheses(docs))
    raise HTTPException(404, "No data for this index.")


@app.get("/ml/{index_id}")
async def get_ml_predictions(index_id: str):
    df = _dataframes.get(index_id)
    if df is None:
        return {"error": "ML predictions require CSV data"}
    return _to_python(run_ml_predictions(df))


@app.get("/eda/{index_id}")
async def get_eda(index_id: str):
    df = _dataframes.get(index_id)
    if df is None:
        return {"error": "EDA requires CSV data"}
    return _to_python(run_eda(df))


@app.get("/analysis/{index_id}")
async def get_analysis(index_id: str):
    df   = _dataframes.get(index_id)
    docs = _docs.get(index_id)
    doc_type = df.attrs.get("document_type", None) if df is not None else None
    try:
        return _to_python(run_pipeline(df=df, docs=docs, doc_type=doc_type))
    except Exception as e:
        return {"error": str(e), "findings": [], "total": 0}


@app.get("/suggestions/{index_id}")
async def get_suggestions(index_id: str):
    return {"suggestions": _suggestions.get(index_id, {})}


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
    df = _dataframes.get(req.index_id)
    try:
        result = ask(db, req.question, history=history, df=df)
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}", flush=True)
        err = str(e).lower()
        if "timeout" in err or "timed out" in err:
            return {"answer": "Zarva took too long to respond — please try again.", "followups": []}
        if "rate" in err or "429" in err:
            return {"answer": "AI service is busy right now. Wait a few seconds and try again.", "followups": []}
        return {"answer": f"Error: {str(e)[:200]}", "followups": []}

    answer = result["answer"] if isinstance(result, dict) else result
    followups = result.get("followups", []) if isinstance(result, dict) else []

    if not answer or not answer.strip():
        answer = "I'm ready to help. Try asking about your data — for example, 'What are the top records?' or 'Show me a summary.'"

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    _histories[req.index_id] = history[-20:]

    return {"answer": answer, "followups": followups}


@app.post("/chat-stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest):
    """Streaming chat — returns text/event-stream so the UI renders word by word."""
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(400, "Question too long.")
    if not _is_safe_input(req.question):
        raise HTTPException(400, "Invalid input detected.")

    db = _indexes.get(req.index_id)
    if db is None:
        raise HTTPException(404, "Unknown index_id")

    history = _histories.get(req.index_id, [])
    df = _dataframes.get(req.index_id)

    async def generate():
        from rag_engine import ask_stream
        full_answer = ""
        try:
            async for chunk in ask_stream(db, req.question, history=history, df=df):
                full_answer += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            print(f"[STREAM ERROR] {e}", flush=True)
            msg = "Zarva took too long to respond — please try again." if "timeout" in str(e).lower() else f"Error: {str(e)[:120]}"
            full_answer = msg
            yield f"data: {json.dumps({'chunk': msg})}\n\n"

        if not full_answer.strip():
            full_answer = "I'm ready to help. Try asking about your data."
            yield f"data: {json.dumps({'chunk': full_answer})}\n\n"

        # Followups after stream ends
        try:
            from rag_engine import generate_followups
            cols = list(df.columns) if df is not None else []
            followups = generate_followups(req.question, full_answer, cols)
        except Exception:
            followups = []

        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": full_answer})
        _histories[req.index_id] = history[-20:]

        yield f"data: {json.dumps({'done': True, 'followups': followups})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── IoT / Raspberry Pi telemetry ─────────────────────────────────────────────

class TelemetryReading(BaseModel):
    temperature: float | None = None
    vibration:   float | None = None
    power:       float | None = None
    humidity:    float | None = None
    custom:      dict  | None = None
    timestamp:   str   | None = None


def _pi_business_impact(device_id: str, alerts: list) -> dict | None:
    if not alerts:
        return None
    severity = "critical" if any(a["severity"] == "critical" for a in alerts) else "warning"
    # Scan loaded DataFrames for customer/account columns
    at_risk: list[str] = []
    for df in _dataframes.values():
        if df is None:
            continue
        cust_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["customer", "account", "client", "company", "name"])),
            None,
        )
        if cust_col:
            at_risk = [str(v) for v in df[cust_col].dropna().unique()[:3]]
            break

    impact: dict = {
        "device": device_id,
        "severity": severity,
        "production_impact": (
            f"Line 4 production disruption imminent" if severity == "critical"
            else "Line 4 performance degraded — monitor closely"
        ),
        "estimated_downtime": "4–8 hours if not addressed within 24 h" if severity == "critical" else None,
        "recommended_actions": [
            f"Schedule preventive maintenance for {device_id} within 24 h",
            "Notify warehouse supervisor immediately",
            "Check spare-parts inventory (compressor bearings, seals)",
        ],
    }
    if at_risk:
        impact["at_risk_customers"] = at_risk
        impact["customer_action"] = (
            f"Proactively notify {at_risk[0]} about potential delivery delay"
        )
    return impact


@app.post("/telemetry/{device_id}")
async def post_telemetry(device_id: str, reading: TelemetryReading):
    """Accept a single sensor reading from a Raspberry Pi or any IoT device."""
    if device_id not in _telemetry:
        _telemetry[device_id] = []
    r = {k: v for k, v in reading.dict().items() if v is not None}
    r.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
    _telemetry[device_id].append(r)
    _telemetry[device_id] = _telemetry[device_id][-100:]
    return {"ok": True, "device_id": device_id}


@app.get("/telemetry/{device_id}")
async def get_telemetry(device_id: str, limit: int = 20):
    """Return recent readings + anomaly analysis + business impact for a device."""
    if device_id == "demo":
        raise HTTPException(400, "Use /telemetry/demo/<device_id> to load demo data.")
    readings = _telemetry.get(device_id, [])
    if not readings:
        raise HTTPException(404, "No data for this device. POST readings first or use /telemetry/demo/<device_id>.")
    recent = readings[-limit:]
    latest = recent[-1]

    alerts: list[dict] = []
    temps  = [r["temperature"] for r in recent if "temperature" in r]
    vibs   = [r["vibration"]   for r in recent if "vibration"   in r]
    pows   = [r["power"]       for r in recent if "power"       in r]

    if temps:
        t = temps[-1]
        if t > 90:
            alerts.append({"sensor": "temperature", "value": t, "threshold": 75,
                           "severity": "critical",
                           "message": f"Temperature critical: {t}°C — compressor failure likely within 48 h"})
        elif t > 75:
            alerts.append({"sensor": "temperature", "value": t, "threshold": 75,
                           "severity": "warning",
                           "message": f"Temperature elevated: {t}°C (normal ≤75°C)"})

    if vibs and len(vibs) >= 4:
        baseline = sum(vibs[:4]) / 4
        v = vibs[-1]
        if v > baseline * 2.5:
            alerts.append({"sensor": "vibration", "value": round(v, 2), "threshold": round(baseline * 2, 2),
                           "severity": "warning",
                           "message": f"Vibration spike: {v:.2f} g — {((v/baseline - 1)*100):.0f}% above baseline (bearing wear?)"})

    if pows and len(pows) >= 4:
        baseline_p = sum(pows[:4]) / 4
        p = pows[-1]
        if p > baseline_p * 1.7:
            alerts.append({"sensor": "power", "value": round(p, 0), "threshold": round(baseline_p * 1.5, 0),
                           "severity": "warning",
                           "message": f"Power spike: {p:.0f} W — motor drawing {((p/baseline_p - 1)*100):.0f}% excess current"})

    status = "normal"
    if any(a["severity"] == "critical" for a in alerts):
        status = "critical"
    elif alerts:
        status = "warning"

    return {
        "device_id": device_id,
        "status": status,
        "readings": recent,
        "latest": latest,
        "alerts": alerts,
        "business_impact": _pi_business_impact(device_id, alerts),
    }


@app.get("/telemetry/demo/{device_id}")
async def load_demo_telemetry(device_id: str):
    """Inject a realistic compressor-failure demo scenario into memory."""
    now = datetime.datetime.utcnow()
    readings = []
    for i in range(20):
        t = now - datetime.timedelta(minutes=(19 - i) * 6)
        if i < 10:                          # normal
            temp  = round(62 + random.uniform(-2, 3),  1)
            vib   = round(0.8 + random.uniform(-0.1, 0.15), 2)
            power = round(1180 + random.uniform(-40, 70), 0)
        elif i < 15:                        # rising
            temp  = round(67 + (i - 10) * 3.5 + random.uniform(-1, 2), 1)
            vib   = round(0.9 + (i - 10) * 0.22 + random.uniform(0, 0.1), 2)
            power = round(1260 + (i - 10) * 70  + random.uniform(-30, 50), 0)
        else:                               # critical
            temp  = round(88 + (i - 15) * 2.5 + random.uniform(0, 3), 1)
            vib   = round(2.0 + (i - 15) * 0.45 + random.uniform(0, 0.2), 2)
            power = round(2050 + (i - 15) * 110 + random.uniform(-50, 80), 0)
        readings.append({"temperature": temp, "vibration": vib, "power": power,
                         "timestamp": t.isoformat()})
    _telemetry[device_id] = readings
    return {"ok": True, "device_id": device_id, "readings_loaded": len(readings),
            "message": f"Demo scenario loaded for {device_id} — compressor failure developing"}


_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_ANALYTICS_WORDS = {
    "chart", "pie", "graph", "bar", "count", "total", "summary", "overview",
    "distribution", "breakdown", "how many", "all records", "all deals",
    "every", "by stage", "by owner", "by status", "by sentiment",
    "average", "avg", "highest", "top", "most", "least", "lowest", "best",
    "worst", "rank", "compare", "vs", "versus", "standup", "pipeline",
    "percentage", "percent", "aggregate", "start with", "starts with",
    "begin with", "begins with", "filter", "ending with", "ends with",
    # column/field discovery
    "column", "columns", "field", "fields", "what data", "what information",
    # time/trend questions
    "by month", "monthly", "by year", "yearly", "by week", "weekly",
    "by date", "over time", "trend", "growth", "peak", "activity",
    "when", "timeline", "by quarter", "quarterly",
    # filter / search queries (sidebar prompt clicks)
    "find records", "show records", "find all", "show all", "show me",
    "list all", "filter by", "where is", "records where", "records with",
    "how many unique", "distinct", "unique values", "which label",
    "most common", "least common", "show top", "top 10", "top 5",
    # analytical reasoning — "which X is", "what should", "focus on", risk, opportunity
    "underperform", "overperform", "which stage", "which group", "which category",
    "which owner", "which region", "which product", "which record",
    "what should", "focus on", "should i", "recommend", "suggest", "improve",
    "risk", "at risk", "opportunity", "gap", "concentration", "anomaly",
    "outlier", "unusual", "surprising", "look off", "something wrong",
    "worst performing", "best performing", "leading", "lagging",
    "compare", "versus", "difference between", "more than", "less than",
}

_POSITIVE_WORDS = {"happy", "positive", "excited", "fantastic", "great", "excellent",
                   "thrilled", "amazing", "outstanding", "wonderful", "love",
                   "satisfied", "joyful"}

_NEGATIVE_WORDS = {"sad", "negative", "bad", "terrible", "awful", "angry", "unhappy",
                   "hate", "worst", "horrible", "disappointed", "upset", "frustrated",
                   "concerned", "struggling", "risk", "poor"}


class TFIDFIndex:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, k: int = 5) -> list[str]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_k if scores[i] > 0]


class EmbeddingIndex:
    """Semantic vector search using OpenAI text-embedding-3-small."""

    _BATCH = 96  # stay under API limit

    def __init__(self, documents: list[str]):
        self.documents = documents
        self._embeddings: np.ndarray | None = None
        self._fallback: TFIDFIndex | None = None
        self._build()

    def _embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI()
        all_vecs = []
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i: i + self._BATCH]
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            all_vecs.extend(e.embedding for e in sorted(resp.data, key=lambda x: x.index))
        arr = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def _build(self):
        try:
            self._embeddings = self._embed(self.documents)
        except Exception:
            self._fallback = TFIDFIndex(self.documents)

    def search(self, query: str, k: int = 5) -> list[str]:
        if self._fallback is not None:
            return self._fallback.search(query, k)
        try:
            q_vec = self._embed([query])[0]
            scores = self._embeddings @ q_vec
            top_k = np.argsort(scores)[::-1][:k]
            return [self.documents[i] for i in top_k]
        except Exception:
            return self.documents[:k]


from backend.model_router import get_model


def get_llm():
    return get_model("narrate")


def get_fast_llm():
    return get_model("classify")


def documents_from_csv(file) -> tuple[list[str], pd.DataFrame]:
    df = pd.read_csv(file)

    # Prepend a metadata doc so column/schema questions always get a direct hit
    col_types = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types.append(f"{col} (numeric)")
        elif "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types.append(f"{col} (date)")
        else:
            col_types.append(f"{col} (text)")
    meta = (
        f"DATASET METADATA: {len(df)} total records. "
        f"Columns in this dataset: {', '.join(col_types)}. "
        f"Column names: {', '.join(df.columns)}."
    )

    docs = [meta]
    for _, row in df.iterrows():
        line = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(line)
    return docs, df


def _prepare_image(raw: bytes) -> tuple[bytes, str]:
    """
    Open any image format with Pillow → resize to ≤2048px longest edge
    → convert to JPEG (smaller, universally accepted by vision APIs).
    Supports: JPEG, PNG, WEBP, GIF, BMP, TIFF, AVIF, HEIC/HEIF, ICO,
              JFIF, PPM, PGM, PBM, TGA, PCX, SGI, EPS, and more.
    Returns (jpeg_bytes, "image/jpeg").
    """
    import io as _io
    try:
        # pillow-heif unlocks HEIC/HEIF (iPhone photos) if installed
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass

        from PIL import Image, ImageOps
        img = Image.open(_io.BytesIO(raw))

        # Handle animated GIF / multi-frame TIFF — use first frame
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)

        # Convert palette/RGBA/CMYK → RGB for JPEG compatibility
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Auto-rotate based on EXIF orientation (phone photos)
        img = ImageOps.exif_transpose(img)

        # Resize: keep aspect ratio, longest edge ≤ 2048px
        max_px = 2048
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        buf.seek(0)
        return buf.read(), "image/jpeg"

    except Exception:
        # Pillow failed (truly unrecognised format) — return raw with best-guess MIME
        if raw[:4] == b'\x89PNG':
            return raw, "image/png"
        if raw[:4] == b'RIFF' or raw[8:12] == b'WEBP':
            return raw, "image/webp"
        return raw, "image/jpeg"


def documents_from_image(file) -> tuple[list[str], None]:
    import base64, httpx
    raw = file.read() if hasattr(file, "read") else file

    # Normalise to JPEG + handle all formats / large images
    img_bytes, mime = _prepare_image(raw)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        # Use GPT-4o-mini vision (supports all normalised JPEG/PNG/WEBP)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": (
                        "You are a precise document and image analyst.\n"
                        "1. Extract ALL text verbatim — every word, number, label, heading.\n"
                        "2. Describe tables row-by-row with their values.\n"
                        "3. List every key fact: dates, names, amounts, IDs, addresses.\n"
                        "4. If it is a photo or diagram, describe what is shown in detail.\n"
                        "5. End with a 3-bullet factual summary.\n"
                        "Be exhaustive — a downstream search engine will index your output."
                    )}
                ]
            }],
            "max_tokens": 2500,
            "temperature": 0,
        }
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers  = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        # Fallback: Groq llama-4-scout
        groq_key = os.environ.get("GROQ_API_KEY", "")
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": (
                        "Extract ALL text verbatim and describe every fact, number, name, "
                        "date, and value visible. Be exhaustive."
                    )}
                ]
            }],
            "max_tokens": 2048,
            "temperature": 0,
        }
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        headers  = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}

    try:
        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        description = f"Image analysis failed: {str(e)[:200]}"

    chunk_size, overlap = 800, 100
    chunks, start = [], 0
    while start < len(description):
        chunks.append(description[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks or ["Could not extract content from this image."], None


def _vision_read_page(page_image_bytes: bytes, page_num: int) -> str:
    """Layer 3: Vision AI for image-only or sparse pages — same pipeline as image uploads."""
    import base64, httpx
    b64 = base64.b64encode(page_image_bytes).decode("utf-8")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": (
                    f"Extract ALL text from this document page {page_num} verbatim. "
                    "Pay special attention to tables, booking references, names, dates, flight numbers, times, and any structured data. "
                    "Output as plain text preserving the structure."
                )}
            ]
        }],
        "max_tokens": 1500,
    }
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


def documents_from_pdf(file) -> tuple[list[str], None]:
    """
    Multi-layer PDF extraction (enterprise pattern):
    Layer 1: pypdf  — plain text, fast
    Layer 2: pdfplumber — tables and structured content
    Layer 3: Vision AI — for pages with no readable text (scanned / image PDFs)
    """
    import pdfplumber
    from pypdf import PdfReader
    import io

    raw = file.read() if hasattr(file, "read") else file
    full_text = ""

    # Layer 1 + 2: text and tables via pdfplumber (includes pypdf fallback internally)
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = ""

                # Layer 2: tables first — preserves rows/columns
                try:
                    tables = page.extract_tables()
                    for table in (tables or []):
                        for row in table:
                            row_text = " | ".join(str(c).strip() for c in row if c and str(c).strip())
                            if row_text:
                                page_text += row_text + "\n"
                except Exception:
                    pass

                # Layer 1: plain text
                try:
                    text = page.extract_text() or ""
                    page_text += text + "\n"
                except Exception:
                    pass

                # Layer 3: Vision AI fallback — if page has almost no readable text
                if len(page_text.strip()) < 80:
                    try:
                        img = page.to_image(resolution=150).original
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        vision_text = _vision_read_page(buf.getvalue(), page_num)
                        if vision_text:
                            page_text += f"\n[Vision extracted - Page {page_num}]:\n{vision_text}\n"
                    except Exception:
                        pass

                full_text += page_text + "\n"
    except Exception:
        # Ultimate fallback: pypdf only
        try:
            reader = PdfReader(io.BytesIO(raw))
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            full_text = ""

    chunk_size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(full_text):
        chunks.append(full_text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks or ["Could not extract text from this PDF."], None


def documents_from_text(file) -> tuple[list[str], None]:
    raw = file.read().decode("utf-8")
    chunk_size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(raw):
        chunks.append(raw[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks, None


def documents_from_salesforce(sf_connector, soql_queries: list[str]) -> tuple[list[str], None]:
    docs = []
    for soql in soql_queries:
        for record in sf_connector.query(soql):
            line = " | ".join(f"{k}: {v}" for k, v in record.items() if k != "attributes")
            docs.append(line)
    return docs, None


def build_index(documents: list[str]) -> EmbeddingIndex:
    return EmbeddingIndex(documents)


def _is_analytics_question(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _ANALYTICS_WORDS)


def _is_sentiment_question(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _POSITIVE_WORDS | _NEGATIVE_WORDS)


def _get_text_column(df: pd.DataFrame) -> str | None:
    for candidate in ["Notes", "Note", "Description", "Comments", "Comment",
                       "Subject", "Text", "Body", "Message", "Content"]:
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str)
            if len(sample) > 0 and sample.str.len().mean() > 20:
                return col
    return None


def _sentiment_score(text: str) -> int:
    t = str(text).lower()
    pos = sum(1 for w in ["excited", "fantastic", "great", "excellent", "thrilled",
                           "amazing", "outstanding", "happy", "wonderful", "love", "positive"] if w in t)
    neg = sum(1 for w in ["disappointed", "frustrated", "risk", "concerned", "poor",
                           "negative", "struggling", "terrible", "awful", "angry", "unhappy"] if w in t)
    return pos - neg


def _compute_doc_analytics(df: pd.DataFrame, question: str) -> str:
    """
    Analytics for document-extracted DataFrames (label / value / category / unit).
    Much simpler than CSV analytics — the whole table fits in context.
    """
    q = question.lower()
    doc_type = df.attrs.get("document_type", "document")
    lines = [f"Extracted data from {doc_type}:", ""]

    # Always show the full table — it's small
    lines.append(df[["label", "value", "category", "unit"]].to_string(index=False))
    lines.append("")

    # Category totals
    if "category" in df.columns:
        cat = df.groupby("category")["value"].agg(total="sum", items="count").sort_values("total", ascending=False)
        lines.append("Totals by category:")
        lines.append(cat.to_string())
        lines.append("")

    # Focused subsets based on question keywords
    deduction_kw = ["deduct", "withhold", "tax", "fica", "medicare", "benefit", "insurance", "401"]
    income_kw    = ["income", "gross", "net", "earn", "wage", "pay", "salary"]
    if any(w in q for w in deduction_kw):
        sub = df[df["category"].isin(["deduction", "tax", "expense"])]
        if not sub.empty:
            lines.append("Deductions / taxes:")
            lines.append(sub[["label", "value"]].sort_values("value", ascending=False).to_string(index=False))
    elif any(w in q for w in income_kw):
        sub = df[df["category"].isin(["income"])]
        if not sub.empty:
            lines.append("Income / pay:")
            lines.append(sub[["label", "value"]].to_string(index=False))

    # Top / bottom
    top_kw    = ["highest", "largest", "most", "top", "biggest", "max"]
    bottom_kw = ["lowest", "smallest", "least", "bottom", "min"]
    if any(w in q for w in top_kw):
        lines.append("Top values:")
        lines.append(df[["label","value"]].sort_values("value", ascending=False).head(5).to_string(index=False))
    if any(w in q for w in bottom_kw):
        lines.append("Bottom values:")
        lines.append(df[["label","value"]].sort_values("value").head(5).to_string(index=False))

    return "\n".join(lines)


def _business_summary(df: pd.DataFrame) -> str:
    """One-paragraph business context added to the top of every analytics context block."""
    lines = []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object]

    # Revenue / amount totals
    amt_col = next((c for c in num_cols if any(k in c.lower() for k in
                    ["amount", "revenue", "value", "price", "sales", "total", "pay"])), None)
    if amt_col:
        vals = pd.to_numeric(df[amt_col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        total = vals.sum()
        avg   = vals.mean()
        lines.append(f"Total {amt_col}: ${total:,.0f} across {len(df)} records (avg ${avg:,.0f})")

    # Stage / status distribution
    stage_col = next((c for c in cat_cols if any(k in c.lower() for k in
                      ["stage", "status", "state", "phase"])), None)
    if stage_col:
        dist = df[stage_col].value_counts()
        lines.append(f"{stage_col} distribution: " + ", ".join(f"{v} {k}" for k, v in dist.head(5).items()))

    # Owner / rep leaderboard
    owner_col = next((c for c in cat_cols if any(k in c.lower() for k in
                      ["owner", "rep", "assignee", "account", "manager"])), None)
    if owner_col and amt_col:
        try:
            vals2 = pd.to_numeric(df[amt_col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
            by_owner = df.copy()
            by_owner["_v"] = vals2
            leaderboard = by_owner.groupby(owner_col)["_v"].sum().sort_values(ascending=False)
            lines.append(f"By {owner_col}: " + ", ".join(
                f"{name} ${val:,.0f}" for name, val in leaderboard.head(4).items()))
        except Exception:
            pass

    return "BUSINESS CONTEXT:\n" + "\n".join(lines) + "\n" if lines else ""


def _compute_analytics(df: pd.DataFrame, question: str) -> str:
    # Extracted document DataFrames (PDF/image → label/value/category) get their own path
    if df.attrs.get("is_extracted") and "label" in df.columns and "value" in df.columns:
        return _compute_doc_analytics(df, question)

    lines = [_business_summary(df)]
    lines.append(f"Dataset: {len(df)} total records")
    lines.append(f"Columns: {', '.join(df.columns)}")
    q = question.lower()

    # "Find records where Column is 'Value'" — direct filter
    import re as _re2
    filter_match = _re2.search(
        r"(?:find|show|filter|get|list).*?(?:where|with|having)?\s+['\"]?(\w[\w\s]*?)['\"]?\s+(?:is|=|equals|containing|contains|like)\s+['\"]?([^'\"?]+?)['\"]?\s*$",
        q, _re2.IGNORECASE
    )
    if filter_match:
        col_hint = filter_match.group(1).strip()
        val_hint = filter_match.group(2).strip()
        matched_col = next((c for c in df.columns if col_hint.lower() in c.lower() or c.lower() in col_hint.lower()), None)
        if matched_col:
            mask = df[matched_col].fillna("").astype(str).str.lower().str.contains(val_hint.lower(), regex=False)
            filtered = df[mask]
            lines.append(f"\nDIRECT FILTER: {len(filtered)} records where {matched_col} contains '{val_hint}'")
            for _, row in filtered.head(10).iterrows():
                lines.append("  " + " | ".join(f"{c}: {row[c]}" for c in df.columns))
            return "\n".join(lines)

    # "top 10 records" / "show all records"
    if any(w in q for w in ["top 10", "top 5", "show top", "list all", "show all", "show me all", "first 10"]):
        n = 10 if "10" in q else 5
        lines.append(f"\nDIRECT ANSWER — top {n} records:")
        for _, row in df.head(n).iterrows():
            lines.append("  " + " | ".join(f"{c}: {row[c]}" for c in df.columns))
        return "\n".join(lines)

    # Direct column listing — answer immediately without full stats
    if any(w in q for w in ["column", "columns", "field", "fields", "what data", "what information"]):
        col_details = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_details.append(f"  {col} (numeric)")
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
                col_details.append(f"  {col} (date)")
            else:
                col_details.append(f"  {col} (text, {df[col].nunique()} unique values)")
        lines.append("\nDIRECT ANSWER - Columns in this dataset:\n" + "\n".join(col_details))
        return "\n".join(lines)

    # Direct pandas answers for filter/count questions
    import re as _re
    # "starts with X" / "beginning with X" / "start with letter X"
    sw_match = _re.search(r'start(?:s)?\s+with\s+(?:alphabet\s+|letter\s+)?["\']?([a-z0-9])["\']?', q)
    if sw_match:
        prefix = sw_match.group(1).lower()
        for col in df.columns:
            if df[col].dtype == object:
                cnt = df[col].fillna("").astype(str).str.lower().str.startswith(prefix).sum()
                lines.append(f"\nDIRECT ANSWER: {cnt} records in '{col}' start with '{prefix.upper()}'")
                matching = df[df[col].fillna("").astype(str).str.lower().str.startswith(prefix)][col].value_counts().head(10)
                for val, c in matching.items():
                    lines.append(f"  {val}: {c}")

    for col in df.columns:
        if df[col].dtype == object:
            counts = df[col].value_counts().head(15)
            lines.append(f"\n{col} breakdown ({df[col].nunique()} unique values):")
            for val, cnt in counts.items():
                lines.append(f"  {val}: {cnt}")
        elif pd.api.types.is_numeric_dtype(df[col]):
            clean = pd.to_numeric(df[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
            lines.append(f"\n{col} stats: total={clean.sum():,.0f}, avg={clean.mean():,.0f}, max={clean.max():,.0f}, min={clean.min():,.0f}")
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                monthly = parsed.dt.to_period("M").value_counts().sort_index()
                lines.append(f"\n{col} by month:")
                for period, cnt in monthly.items():
                    lines.append(f"  {period}: {cnt}")
            except Exception:
                pass

    # sentiment on text columns
    text_col = _get_text_column(df)
    if text_col:
        df2 = df.copy()
        df2["_score"] = df2[text_col].fillna("").astype(str).apply(_sentiment_score)
        pos = len(df2[df2["_score"] > 0])
        neg = len(df2[df2["_score"] < 0])
        neu = len(df2) - pos - neg
        lines.append(f"\nSentiment in '{text_col}': Positive={pos}, Negative={neg}, Neutral={neu}")
        if any(w in question.lower() for w in _POSITIVE_WORDS):
            show = df2.nlargest(min(5, max(1, pos)), "_score")
            lines.append("Most positive records:")
            for _, row in show.iterrows():
                lines.append("  " + " | ".join(f"{c}: {row[c]}" for c in df.columns))
        if any(w in question.lower() for w in _NEGATIVE_WORDS):
            show = df2.nsmallest(min(5, max(1, neg)), "_score")
            lines.append("Most negative records:")
            for _, row in show.iterrows():
                lines.append("  " + " | ".join(f"{c}: {row[c]}" for c in df.columns))

    return "\n".join(lines)


_ANALYTICS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Zarva, a senior data analyst embedded in the user's business. Talk like a sharp analyst on a call — direct, specific, no fluff.

RULES:
- Lead with the answer immediately — no preamble, no label, no "Here's the breakdown:"
- Use real names and numbers from the data: "Adobe ($473K)" not "the top account"
- Compare when useful: "that's 2× the average deal size"
- Be precise about risks: name who, how much, and how long
- NEVER start a line or paragraph with: "Key Insight", "Business Meaning", "Next Action", "Important Note", "Summary", "Overview", "Analysis", "Takeaway", "Insight", "Recommendation"
- NEVER give generic advice: "provide training", "conduct reviews", "improve communication", "align stakeholders", "set clear goals" — skip it entirely if you can't be specific
- Do NOT repeat the same sentence structure in consecutive bullets
- NEVER recommend Excel, Tableau, Power BI — Zarva handles charts natively ("ask me to draw a chart")

MEMORY:
{history}

DATA:
{context}"""),
    ("human", "{question}"),
])

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Zarva, a sharp analyst. Answer like you've read the document and know it well — not like you're summarizing it for the first time.

RULES:
- State the answer directly, then add context only if it helps
- Quote exact figures, names, and dates when they exist
- If something isn't in the data, say so briefly — don't pad
- No section headers ("Key Insight", "Next Action", etc.)
- No generic filler recommendations
- NEVER recommend Excel, Tableau, Power BI — Zarva handles charts natively

MEMORY:
{history}

DOCUMENT:
{context}"""),
    ("human", "{question}"),
])


def generate_followups(question: str, answer: str, columns: list) -> list[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate exactly 3 short follow-up questions a business analyst would ask next.
Rules:
- Each on its own line, no bullets, no numbers
- Max 8 words each
- Based on the question and answer given
- Use actual column names if relevant
- Return ONLY the 3 questions"""),
        ("human", f"Columns available: {columns}\nQuestion: {question}\nAnswer: {answer}"),
    ])
    try:
        chain = prompt | get_fast_llm() | StrOutputParser()
        result = chain.invoke({})
        return [s.strip() for s in result.strip().split("\n") if s.strip()][:3]
    except Exception:
        return []


_CHART_WORDS = {
    "chart", "pie", "graph", "bar", "plot", "visual", "visualize", "draw",
    "histogram", "diagram", "donut", "trend", "area chart", "line chart",
    "scatter", "scatter plot", "bubble", "bubble chart",
    "candlestick", "candle", "ohlc", "stock price",
    "waterfall", "cascade", "bridge chart",
    "radar", "spider", "spider chart", "web chart",
    "violin", "violin plot",
    "funnel", "funnel chart",
    "heatmap", "heat map",
    "distribution of", "correlation", "versus", "vs ",
}
_SUMMARY_WORDS = {"summarize", "summary", "overview", "what is this", "what does this", "explain", "tell me about", "describe"}
_COUNT_WORDS = {"how many", "count", "total", "number of"}

# Common autocorrects and abbreviations → canonical forms
_AUTOCORRECT = [
    # chart name typos
    (r'\bpi\b',              'pie'),
    (r'\bbart\b',            'bar'),
    (r'\bpie\s*chat\b',      'pie chart'),
    (r'\bchar\b',            'chart'),
    (r'\bgrapgh\b',          'graph'),
    (r'\bvisualise\b',       'visualize'),
    (r'\bsctter\b',          'scatter'),
    (r'\bscater\b',          'scatter'),
    (r'\bscattr\b',          'scatter'),
    (r'\bscattre\b',         'scatter'),
    (r'\bvolin\b',           'violin'),
    (r'\bvioln\b',           'violin'),
    (r'\bhistagram\b',       'histogram'),
    (r'\bhistorgram\b',      'histogram'),
    (r'\bheatmp\b',          'heatmap'),
    (r'\bcanlde\b',          'candle'),
    # column/value typos
    (r'\bammount\b',         'amount'),
    (r'\bamount\b',          'amount'),
    (r'\brevienue\b',        'revenue'),
    (r'\breveune\b',         'revenue'),
    (r'\bstege\b',           'stage'),
    (r'\bstaeg\b',           'stage'),
    (r'\brecods\b',          'records'),
    (r'\brevune\b',          'revenue'),
    # phrasing
    (r'\bdeducton\b',        'deduction'),
    (r'\bcolumns?\b',        'columns'),
    (r'\bhow much\b',        'what is the total'),
    (r'\bwhat r\b',          'what are'),
    (r'\bshow me\b',         'show'),
    (r'\bal\b',              'all'),
]


def _normalize(q: str) -> str:
    import re as _re
    for pattern, replacement in _AUTOCORRECT:
        q = _re.sub(pattern, replacement, q, flags=_re.IGNORECASE)
    return q


def _detect_intent(question: str) -> str:
    q = _normalize(question.lower())
    if any(w in q for w in _CHART_WORDS):
        return "chart"
    if any(w in q for w in _SUMMARY_WORDS):
        return "summary"
    if any(w in q for w in _COUNT_WORDS):
        return "count"
    if any(w in q for w in _ANALYTICS_WORDS):
        return "analytics"
    return "rag"


def _build_history_context(history: list[dict]) -> str:
    """
    Build a structured memory block from conversation history.
    Recent turns verbatim + extracted key findings from older turns.
    """
    if not history:
        return "No prior conversation."

    pairs = []
    for i in range(0, len(history) - 1, 2):
        if history[i]["role"] == "user" and i + 1 < len(history):
            pairs.append((history[i]["content"], history[i + 1]["content"]))

    if not pairs:
        return "No prior conversation."

    lines = []

    # Summarise older turns (beyond last 3) as a findings list
    if len(pairs) > 3:
        older = pairs[:-3]
        lines.append("Key findings from earlier in this conversation:")
        for q, a in older:
            # Pull first sentence of each answer as the key finding
            first = a.split(".")[0].strip()
            if first:
                lines.append(f"  • [{q[:40]}] → {first}")
        lines.append("")

    # Last 3 turns verbatim
    lines.append("Recent conversation:")
    for q, a in pairs[-3:]:
        lines.append(f"User: {q}")
        lines.append(f"Zarva: {a}")
        lines.append("")

    return "\n".join(lines)


def ask(db: EmbeddingIndex, question: str, history: list[dict] | None = None, df: pd.DataFrame | None = None) -> dict:
    history_text = _build_history_context(history or [])

    intent = _detect_intent(question)

    # Chart request on a document (PDF/image) — check if chart type needs tabular data
    if intent == "chart" and df is None:
        import re as _re2
        q_norm = _normalize(question.lower())
        # Regex patterns survive typos like "sctter", "scater", "volin"
        _NEEDS_DF_RE = [
            r'\bscatt?e?r\b',      # scatter, scater, sctter, scattr
            r'\bviolins?\b',       # violin, violins
            r'\bbubble\b',
            r'\bheat\s*m?a?p\b',   # heatmap, heat map
            r'\bcandl[ei]?\b',     # candle, candli, candlestick
            r'\bohlc\b',
        ]
        needs_df = any(_re2.search(p, q_norm, _re2.IGNORECASE) for p in _NEEDS_DF_RE)
        if needs_df:
            return {
                "answer": (
                    "That chart type needs structured columns to work — it can't be built from a PDF. "
                    "Try uploading a CSV file instead, or ask for a bar chart, pie chart, "
                    "waterfall, radar, or funnel — those all work with document data."
                ),
                "followups": [
                    "Draw a pie chart of all deductions",
                    "Draw a waterfall chart of deductions",
                    "Draw a bar chart of all values",
                ]
            }
        return {
            "answer": "On it — extracting the numbers from your document and building the chart now.",
            "followups": []
        }

    # For extracted document DataFrames (PDF/image → structured), ALL questions route
    # through analytics — this is the semantic data model path
    is_extracted_doc = df is not None and df.attrs.get("is_extracted", False)

    use_analytics = df is not None and (
        is_extracted_doc                                       # always for extracted docs
        or intent in ("analytics", "count", "chart")          # explicit intent
        or _is_sentiment_question(question)                    # sentiment filters
    )

    if use_analytics:
        context = _compute_analytics(df, question)
        # Business reasoning questions always get gpt-4o — cheap model gives generic output
        _REASONING_WORDS = {"why", "should", "recommend", "risk", "underperform", "focus",
                            "strategy", "improve", "compare", "which", "best", "worst",
                            "opportunity", "gap", "at risk", "problem", "issue", "concern"}
        q_lower = question.lower()
        needs_strong = any(w in q_lower for w in _REASONING_WORDS) or len(question) > 60
        llm = get_model("reason") if needs_strong else get_model("narrate")
        chain = _ANALYTICS_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question, "history": history_text})
        followups = generate_followups(question, answer, list(df.columns))
        return {"answer": answer, "followups": followups}

    # For small indexes (PDF/image/text) use ALL chunks as context — no search needed
    if len(db.documents) <= 80:
        context = "\n\n".join(db.documents[:60])
    else:
        docs = db.search(question, k=10)
        context = "\n".join(docs) if docs else "No matching records found."

    _REASONING_WORDS = {"why", "should", "recommend", "risk", "underperform", "focus",
                        "strategy", "improve", "compare", "which", "best", "worst",
                        "opportunity", "gap", "at risk", "problem", "issue", "concern"}
    q_lower = question.lower()
    needs_strong = any(w in q_lower for w in _REASONING_WORDS) or len(question) > 60
    llm = get_model("reason") if needs_strong else get_model("narrate")
    chain = _RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "history": history_text})
    cols = list(df.columns) if df is not None else []
    followups = generate_followups(question, answer, cols)
    return {"answer": answer, "followups": followups}


async def ask_stream(db, question: str, history=None, df=None):
    """Async generator that yields text chunks for streaming responses."""
    history_text = _build_history_context(history or [])
    intent = _detect_intent(question)

    if intent == "chart":
        yield "On it — extracting numbers and building the chart now."
        return

    is_extracted_doc = df is not None and df.attrs.get("is_extracted", False)
    use_analytics = df is not None and (
        is_extracted_doc
        or intent in ("analytics", "count", "chart")
        or _is_sentiment_question(question)
    )

    if use_analytics:
        context = _compute_analytics(df, question)
        _REASONING_WORDS = {"why", "should", "recommend", "risk", "underperform", "focus",
                            "strategy", "improve", "compare", "which", "best", "worst",
                            "opportunity", "gap", "at risk", "problem", "issue", "concern"}
        needs_strong = any(w in question.lower() for w in _REASONING_WORDS) or len(question) > 60
        llm = get_model("reason") if needs_strong else get_model("narrate")
        chain = _ANALYTICS_PROMPT | llm
        async for chunk in chain.astream({"context": context, "question": question, "history": history_text}):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)
        return

    if len(db.documents) <= 80:
        context = "\n\n".join(db.documents[:60])
    else:
        docs = db.search(question, k=10)
        context = "\n".join(docs) if docs else "No matching records found."

    needs_strong = len(question) > 60
    llm = get_model("reason") if needs_strong else get_model("narrate")
    chain = _RAG_PROMPT | llm
    async for chunk in chain.astream({"context": context, "question": question, "history": history_text}):
        yield chunk.content if hasattr(chunk, "content") else str(chunk)

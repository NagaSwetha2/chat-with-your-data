"""
Universal Document → DataFrame extractor.

Converts unstructured documents (PDF, image, text) into a normalized
pandas DataFrame once at upload time. All subsequent questions then hit
pandas, not raw text — giving precise analytics instead of best-guess RAG.

Output schema:  label (str) | value (float) | category (str) | unit (str)
"""
import re
import json
import os
import pandas as pd
from backend.model_router import get_model


def _extract_json_block(text: str) -> str:
    """Pull the first JSON object or array out of an LLM response."""
    for pat in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        m = re.search(pat, text)
        if m:
            return m.group()
    return text


def extract_structured_data(docs: list[str]) -> pd.DataFrame | None:
    """
    Given document text chunks, ask gpt-4o to emit every numeric entity
    as a structured JSON table.  Returns a DataFrame or None.
    Called once per file upload; result is cached in _dataframes[index_id].
    Uses gpt-4o (not mini) — extraction accuracy is the foundation of everything.
    """
    full_text = "\n".join(docs[:40])[:6000]

    prompt = f"""You are a data extraction expert. Your job: extract EVERY numeric fact from this document with full context.

DOCUMENT:
{full_text}

OUTPUT — a single JSON object (no markdown, no explanation, just JSON):
{{
  "document_type": "payroll|invoice|financial_report|sales_report|inventory|telemetry|budget|contract|receipt|medical|other",
  "currency": "USD|EUR|GBP|INR|other",
  "rows": [
    {{
      "label": "Full descriptive name (e.g. 'Federal Income Tax Withheld', not 'FIT')",
      "value": 1234.56,
      "category": "income|deduction|tax|expense|revenue|asset|liability|metric|rate|quantity|other",
      "unit": "USD|EUR|%|hours|days|count|kg|units|other"
    }}
  ]
}}

RULES:
1. Extract EVERY number visible — salaries, taxes, totals, quantities, rates, percentages, dates as numbers
2. "value" must be a plain float (strip $, commas, % — but keep the unit field accurate)
3. Labels must be human-readable and complete — never abbreviate
4. Category mapping:
   income     = gross pay, wages, salary, total earnings, revenue, sales
   deduction  = insurance, 401k, garnishments, benefits taken before net pay
   tax        = federal/state/local withholding, FICA, Medicare, Social Security
   expense    = business costs, operating costs, COGS
   metric     = KPI, percentage, rate, score, utilisation
   quantity   = count of items, units, hours worked
5. If a value appears in multiple forms (e.g. gross AND net), extract both
6. Return ONLY the JSON — no prose, no markdown fences"""

    def _attempt(model_task: str) -> pd.DataFrame | None:
        try:
            resp = get_model(model_task).invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            raw = _extract_json_block(text)
            data = json.loads(raw)

            rows = data.get("rows", [])
            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df[df["value"] != 0]

            for col, default in [("category", "other"), ("unit", ""), ("label", None)]:
                if col not in df.columns:
                    if default is None:
                        return None
                    df[col] = default

            if len(df) < 2:
                return None

            df = df.reset_index(drop=True)
            df.attrs["document_type"] = data.get("document_type", "other")
            df.attrs["currency"]      = data.get("currency", "USD")
            df.attrs["is_extracted"]  = True
            return df
        except Exception:
            return None

    # Try gpt-4o first (best accuracy), fall back to gpt-4o-mini on failure
    result = _attempt("reason")
    if result is None:
        result = _attempt("extract")
    return result

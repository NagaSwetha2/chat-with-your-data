"""
Business Context Engine — 8-stage analytical pipeline

Dataset → Domain Detector → Ontology Builder → Metric Discovery
→ Pattern Engine → Risk Engine → Recommendation Engine → LLM Narrative
"""

from __future__ import annotations
import json, re
import numpy as np
import pandas as pd
from backend.model_router import get_model
from langchain_core.output_parsers import StrOutputParser


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[$,%,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _fmt(v: float, unit: str = "") -> str:
    u = unit.upper()
    if "$" in unit or "USD" in u or "DOLLAR" in u:
        if abs(v) >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"${v:,.0f}"
        return f"${v:,.2f}"
    if "%" in unit:
        return f"{v:.1f}%"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Domain Detector
# ═══════════════════════════════════════════════════════════════════════════════

_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "crm":       ["amount", "stage", "opportunity", "lead", "deal", "pipeline", "owner", "close_date", "account", "contact", "won", "lost"],
    "payroll":   ["net pay", "gross pay", "withholding", "ytd", "pay period", "fica", "deduction", "401k", "pay stub", "paystub"],
    "financial": ["revenue", "expenses", "profit", "loss", "assets", "liabilities", "equity", "income", "ebitda", "margin", "balance sheet"],
    "hr":        ["employee", "salary", "department", "hire date", "tenure", "headcount", "attrition", "performance", "role", "title"],
    "inventory": ["sku", "stock", "quantity", "reorder", "warehouse", "supplier", "unit cost", "inventory", "on hand"],
    "travel":    ["flight", "departure", "arrival", "booking", "airline", "pnr", "seat", "itinerary", "hotel", "layover"],
    "contract":  ["agreement", "contract", "clause", "termination", "obligation", "party", "signed", "effective date", "penalty"],
    "marketing": ["campaign", "clicks", "impressions", "ctr", "conversion", "cpc", "spend", "roas", "channel", "funnel"],
}

_DOMAIN_LABELS = {
    "crm": "Sales CRM",
    "payroll": "Payroll",
    "financial": "Financial Statement",
    "hr": "HR Data",
    "inventory": "Inventory",
    "travel": "Travel / Booking",
    "contract": "Contract",
    "marketing": "Marketing Analytics",
    "general": "General Data",
}


def detect_domain(df: pd.DataFrame | None, docs: list[str] | None, doc_type: str | None) -> dict:
    scores: dict[str, int] = {d: 0 for d in _DOMAIN_SIGNALS}

    # From doc_type attr
    if doc_type:
        dt = doc_type.lower()
        for domain, signals in _DOMAIN_SIGNALS.items():
            if any(s in dt for s in signals):
                scores[domain] += 3

    # From column names
    if df is not None:
        col_text = " ".join(c.lower() for c in df.columns)
        for domain, signals in _DOMAIN_SIGNALS.items():
            scores[domain] += sum(1 for s in signals if s in col_text)

    # From extracted doc labels
    if df is not None and "label" in df.columns:
        label_text = " ".join(str(v).lower() for v in df["label"].dropna())
        for domain, signals in _DOMAIN_SIGNALS.items():
            scores[domain] += sum(2 for s in signals if s in label_text)

    # From document text
    if docs:
        doc_text = " ".join(docs[:5]).lower()[:3000]
        for domain, signals in _DOMAIN_SIGNALS.items():
            scores[domain] += sum(1 for s in signals if s in doc_text)

    best = max(scores, key=scores.get)
    confidence = min(1.0, scores[best] / 10)
    domain = best if confidence > 0.1 else "general"

    return {
        "domain": domain,
        "label": _DOMAIN_LABELS.get(domain, "General Data"),
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Ontology Builder
# ═══════════════════════════════════════════════════════════════════════════════

_COL_ROLES: dict[str, list[str]] = {
    "primary_metric":  ["amount", "revenue", "value", "price", "cost", "sales", "total", "net pay", "gross pay"],
    "entity":          ["name", "company", "account", "contact", "customer", "employee", "deal", "opportunity"],
    "category":        ["stage", "status", "type", "category", "segment", "tier", "phase"],
    "time":            ["date", "time", "month", "year", "quarter", "period", "created", "close", "due"],
    "owner":           ["owner", "rep", "assignee", "manager", "agent", "engineer", "salesperson"],
    "region":          ["region", "territory", "country", "state", "city", "location", "zone"],
    "secondary_metric":["probability", "score", "rating", "rank", "priority", "weight", "pct", "percent"],
}


def build_ontology(df: pd.DataFrame | None, domain: str) -> dict:
    if df is None:
        return {}

    is_extracted = df.attrs.get("is_extracted", False)

    # Extracted doc uses label/value/category/unit schema
    if is_extracted and "label" in df.columns:
        unit_col = df["unit"].dropna().iloc[0] if "unit" in df.columns and not df["unit"].dropna().empty else ""
        return {
            "is_extracted": True,
            "label_col": "label",
            "value_col": "value",
            "category_col": "category" if "category" in df.columns else None,
            "unit": unit_col,
        }

    # CSV / tabular — map column → semantic role
    cols = list(df.columns)
    mapping: dict[str, str | None] = {role: None for role in _COL_ROLES}

    for role, keywords in _COL_ROLES.items():
        for col in cols:
            cl = col.lower()
            if any(kw in cl for kw in keywords):
                mapping[role] = col
                break

    return {"is_extracted": False, **mapping}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Metric Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_metrics(df: pd.DataFrame | None, ontology: dict, domain: str) -> list[dict]:
    if df is None:
        return []

    metrics: list[dict] = []

    if ontology.get("is_extracted"):
        # Extracted doc: group by unit, compute totals per category
        val_col = ontology.get("value_col", "value")
        cat_col = ontology.get("category_col")
        unit_col = "unit" if "unit" in df.columns else None

        if unit_col:
            for unit in df[unit_col].dropna().unique():
                sub = df[df[unit_col] == unit].copy()
                sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
                sub = sub.dropna(subset=[val_col])
                if sub.empty:
                    continue
                total = sub[val_col].sum()
                metrics.append({
                    "name": f"Total ({unit})",
                    "value": total,
                    "formatted": _fmt(total, unit),
                    "unit": unit,
                    "status": "info",
                })
                if cat_col and cat_col in sub.columns:
                    for cat, grp in sub.groupby(cat_col):
                        cat_total = grp[val_col].sum()
                        pct = cat_total / total * 100 if total else 0
                        metrics.append({
                            "name": f"{cat} ({unit})",
                            "value": cat_total,
                            "formatted": f"{_fmt(cat_total, unit)} ({pct:.0f}%)",
                            "unit": unit,
                            "pct": pct,
                            "status": "info",
                        })
        return metrics[:12]

    # Tabular data
    pm = ontology.get("primary_metric")
    cat = ontology.get("category")
    owner = ontology.get("owner")
    time_col = ontology.get("time")
    entity = ontology.get("entity")

    if pm and pm in df.columns:
        vals = _clean_num(df[pm]).dropna()
        if not vals.empty:
            metrics.append({"name": "Total", "value": vals.sum(), "formatted": _fmt(vals.sum(), "$"), "unit": "$", "status": "info"})
            metrics.append({"name": "Average", "value": vals.mean(), "formatted": _fmt(vals.mean(), "$"), "unit": "$", "status": "info"})
            metrics.append({"name": "Records", "value": len(df), "formatted": str(len(df)), "unit": "count", "status": "info"})

            # Top / bottom
            if entity and entity in df.columns:
                top_idx = vals.idxmax()
                bot_idx = vals.idxmin()
                metrics.append({
                    "name": "Largest",
                    "value": vals.max(),
                    "formatted": f"{df.loc[top_idx, entity]} — {_fmt(vals.max(), '$')}",
                    "unit": "$", "status": "positive",
                })
                metrics.append({
                    "name": "Smallest",
                    "value": vals.min(),
                    "formatted": f"{df.loc[bot_idx, entity]} — {_fmt(vals.min(), '$')}",
                    "unit": "$", "status": "warning",
                })

    if cat and cat in df.columns and pm and pm in df.columns:
        vals = _clean_num(df[pm])
        by_cat = df.groupby(cat)[pm].apply(lambda s: _clean_num(s).sum()).sort_values(ascending=False)
        for stage, total in by_cat.head(5).items():
            pct = total / by_cat.sum() * 100 if by_cat.sum() else 0
            won_like = any(w in str(stage).lower() for w in ["won", "closed", "complete", "resolved"])
            metrics.append({
                "name": str(stage),
                "value": total,
                "formatted": f"{_fmt(total, '$')} ({pct:.0f}%)",
                "unit": "$",
                "pct": pct,
                "status": "positive" if won_like else "info",
            })

        # Win rate
        won = df[df[cat].str.contains("won|closed|complete|resolved", case=False, na=False)]
        win_rate = len(won) / len(df) * 100 if len(df) > 0 else 0
        metrics.append({"name": "Win Rate", "value": win_rate, "formatted": f"{win_rate:.0f}%", "unit": "%",
                         "status": "positive" if win_rate >= 30 else "warning"})

    if owner and owner in df.columns and pm and pm in df.columns:
        vals = _clean_num(df[pm])
        by_owner = df.groupby(owner)[pm].apply(lambda s: _clean_num(s).sum()).sort_values(ascending=False)
        if not by_owner.empty:
            top_name = by_owner.index[0]
            metrics.append({
                "name": "Top Performer",
                "value": by_owner.iloc[0],
                "formatted": f"{top_name} — {_fmt(by_owner.iloc[0], '$')}",
                "unit": "$", "status": "positive",
            })

    return metrics[:15]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Pattern Engine
# ═══════════════════════════════════════════════════════════════════════════════

def detect_patterns(df: pd.DataFrame | None, ontology: dict, metrics: list[dict]) -> list[dict]:
    if df is None:
        return []
    patterns: list[dict] = []

    if ontology.get("is_extracted"):
        # For extracted docs, look at value distributions within each unit
        if "value" in df.columns and "unit" in df.columns and "label" in df.columns:
            for unit in df["unit"].dropna().unique():
                sub = df[df["unit"] == unit].copy()
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub.dropna(subset=["value"]).sort_values("value", ascending=False)
                if len(sub) < 2:
                    continue
                total = sub["value"].sum()
                if total == 0:
                    continue
                # Largest item as % of total
                top_row = sub.iloc[0]
                top_pct = top_row["value"] / total * 100
                if top_pct > 60:
                    patterns.append({
                        "type": "concentration",
                        "description": f"{top_row['label']} is {top_pct:.0f}% of all {unit} values",
                        "magnitude": "high",
                        "evidence": f"{_fmt(top_row['value'], unit)} of {_fmt(total, unit)} total",
                    })
        return patterns

    pm = ontology.get("primary_metric")
    cat = ontology.get("category")
    owner = ontology.get("owner")
    time_col = ontology.get("time")

    if pm and pm in df.columns:
        vals = _clean_num(df[pm]).dropna()
        if len(vals) >= 5:
            # Concentration (80/20)
            top20_count = max(1, int(len(vals) * 0.2))
            top20_sum = vals.nlargest(top20_count).sum()
            total = vals.sum()
            if total > 0:
                conc_pct = top20_sum / total * 100
                if conc_pct > 70:
                    patterns.append({
                        "type": "concentration",
                        "description": f"Top 20% of records drive {conc_pct:.0f}% of {pm}",
                        "magnitude": "high" if conc_pct > 80 else "medium",
                        "evidence": f"{_fmt(top20_sum, '$')} out of {_fmt(total, '$')} total",
                    })

            # Outliers (IQR)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            outliers = vals[(vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)]
            if len(outliers) > 0:
                patterns.append({
                    "type": "outlier",
                    "description": f"{len(outliers)} records with unusual {pm} values",
                    "magnitude": "medium",
                    "evidence": f"Range: {_fmt(vals.min(), '$')} – {_fmt(vals.max(), '$')}, median {_fmt(vals.median(), '$')}",
                })

    # Category imbalance
    if cat and cat in df.columns:
        dist = df[cat].value_counts(normalize=True)
        if len(dist) > 1 and dist.iloc[0] > 0.6:
            patterns.append({
                "type": "imbalance",
                "description": f"{dist.index[0]} dominates — {dist.iloc[0]*100:.0f}% of all records",
                "magnitude": "medium",
                "evidence": f"{dist.index[0]}: {int(dist.iloc[0]*len(df))} records vs {len(df)-int(dist.iloc[0]*len(df))} others",
            })

    # Owner concentration
    if owner and owner in df.columns and pm and pm in df.columns:
        vals = _clean_num(df[pm])
        by_owner = df.groupby(owner)[pm].apply(lambda s: _clean_num(s).sum())
        if len(by_owner) > 1:
            top_pct = by_owner.max() / by_owner.sum() * 100
            if top_pct > 50:
                patterns.append({
                    "type": "concentration",
                    "description": f"{by_owner.idxmax()} holds {top_pct:.0f}% of total {pm}",
                    "magnitude": "medium",
                    "evidence": f"{_fmt(by_owner.max(), '$')} out of {_fmt(by_owner.sum(), '$')}",
                })

    return patterns[:6]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Risk Engine
# ═══════════════════════════════════════════════════════════════════════════════

def assess_risks(df: pd.DataFrame | None, ontology: dict, metrics: list[dict], patterns: list[dict]) -> list[dict]:
    if df is None:
        return []
    risks: list[dict] = []

    # Concentration risk
    high_conc = [p for p in patterns if p["type"] == "concentration" and p["magnitude"] == "high"]
    for p in high_conc:
        risks.append({
            "risk": "Concentration Risk",
            "likelihood": "high",
            "impact": "high",
            "evidence": p["evidence"],
            "description": p["description"],
            "mitigation": "Diversify across more entities to reduce single-point dependency",
        })

    cat = ontology.get("category")
    pm = ontology.get("primary_metric")
    time_col = ontology.get("time")

    if not ontology.get("is_extracted") and cat and pm and df is not None:
        # Stagnation risk — large value in early stages
        early_stages = df[df[cat].str.contains("prospect|discovery|new|lead|open", case=False, na=False)]
        if len(early_stages) > 0 and pm in df.columns:
            early_val = _clean_num(early_stages[pm]).sum()
            total_val = _clean_num(df[pm]).sum()
            if total_val > 0 and early_val / total_val > 0.4:
                risks.append({
                    "risk": "Pipeline Stagnation",
                    "likelihood": "medium",
                    "impact": "high",
                    "evidence": f"{_fmt(early_val, '$')} ({early_val/total_val*100:.0f}% of pipeline) stuck in early stages",
                    "description": f"{len(early_stages)} deals not progressing",
                    "mitigation": "Review and action early-stage deals; set stage-advance deadlines",
                })

        # Missing win rate signal
        win_metric = next((m for m in metrics if m["name"] == "Win Rate"), None)
        if win_metric and win_metric["value"] < 20:
            risks.append({
                "risk": "Low Win Rate",
                "likelihood": "high",
                "impact": "high",
                "evidence": f"Win rate is {win_metric['formatted']} — below 20% benchmark",
                "description": "More than 80% of deals are being lost",
                "mitigation": "Analyse lost deals for common objections; improve qualification criteria",
            })

    return risks[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(
    metrics: list[dict],
    patterns: list[dict],
    risks: list[dict],
    ontology: dict,
    domain: str,
) -> list[dict]:
    recs: list[dict] = []

    # Risk-driven recommendations
    for risk in risks[:3]:
        recs.append({
            "priority": "high" if risk["impact"] == "high" else "medium",
            "action": f"Address {risk['risk']}",
            "rationale": risk["description"],
            "evidence": risk["evidence"],
            "mitigation": risk["mitigation"],
        })

    # Pattern-driven recommendations
    for p in patterns:
        if p["type"] == "outlier":
            recs.append({
                "priority": "medium",
                "action": "Investigate outlier records",
                "rationale": p["description"],
                "evidence": p["evidence"],
                "mitigation": "Review outlier causes and determine if they represent errors or opportunities",
            })
        if p["type"] == "imbalance" and len(recs) < 5:
            recs.append({
                "priority": "medium",
                "action": f"Rebalance {p['description'].split(' dominates')[0]}",
                "rationale": p["description"],
                "evidence": p["evidence"],
                "mitigation": "Redistribute focus across underrepresented categories",
            })

    # Metric-driven recommendations
    positive_metric = next((m for m in metrics if m["status"] == "positive"), None)
    if positive_metric and len(recs) < 5:
        recs.append({
            "priority": "low",
            "action": f"Scale what's working — {positive_metric['name']}",
            "rationale": f"{positive_metric['name']} is performing well at {positive_metric['formatted']}",
            "evidence": positive_metric["formatted"],
            "mitigation": "Identify success factors and replicate across similar segments",
        })

    return recs[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — LLM Narrative
# ═══════════════════════════════════════════════════════════════════════════════

def generate_narrative(
    domain_info: dict,
    metrics: list[dict],
    patterns: list[dict],
    risks: list[dict],
    recommendations: list[dict],
    ontology: dict,
    docs: list[str] | None = None,
) -> dict:
    domain_label = domain_info.get("label", "data")

    metrics_text = "\n".join(f"  • {m['name']}: {m['formatted']}" for m in metrics[:10])
    patterns_text = "\n".join(f"  • [{p['type'].upper()}] {p['description']} — {p['evidence']}" for p in patterns)
    risks_text = "\n".join(f"  • [{r['likelihood'].upper()} risk] {r['risk']}: {r['evidence']}" for r in risks)
    recs_text = "\n".join(f"  • [{r['priority'].upper()}] {r['action']} — {r['rationale']}" for r in recommendations)

    doc_excerpt = ""
    if docs:
        doc_excerpt = f"\nDocument context:\n{' '.join(docs[:3])[:600]}"

    prompt = f"""You are a senior analyst writing an executive brief about {domain_label}.

COMPUTED METRICS:
{metrics_text or '  (none computed)'}

DETECTED PATTERNS:
{patterns_text or '  (none detected)'}

RISKS IDENTIFIED:
{risks_text or '  (none identified)'}

RECOMMENDED ACTIONS:
{recs_text or '  (none generated)'}{doc_excerpt}

Write a concise executive narrative with EXACTLY this structure (return as JSON):
{{
  "headline": "One sentence, lead with the single most important number or finding",
  "body": "2 short paragraphs. Para 1: what the data shows with specific figures. Para 2: the main risk or opportunity and its business impact.",
  "key_numbers": ["3-4 bullet strings, each a single key stat: 'Total pipeline: $2.4M'"],
  "next_action": "One specific, time-bound action the reader should take today"
}}

Rules:
- Every sentence must reference a specific number from the computed metrics above
- No generic statements like 'the data shows interesting trends'
- Write for a business executive, not a data scientist
- Do NOT recommend Excel, Power BI, Google Sheets or any external tool
Return only valid JSON."""

    try:
        resp = get_model("reason").invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            data = json.loads(m.group())
            return {
                "headline": data.get("headline", ""),
                "body": data.get("body", ""),
                "key_numbers": data.get("key_numbers", []),
                "next_action": data.get("next_action", ""),
            }
    except Exception:
        pass

    # Fallback
    top_metric = metrics[0] if metrics else None
    return {
        "headline": f"Analysis of {domain_label} — {top_metric['formatted'] if top_metric else 'see details below'}",
        "body": "Metrics and patterns have been computed. Review the findings below.",
        "key_numbers": [m["formatted"] for m in metrics[:4]],
        "next_action": recommendations[0]["action"] if recommendations else "Review the data insights",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — run full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    df: pd.DataFrame | None = None,
    docs: list[str] | None = None,
    doc_type: str | None = None,
) -> dict:
    # Stage 1: Domain
    domain_info = detect_domain(df, docs, doc_type)
    domain = domain_info["domain"]

    # Stage 2: Ontology
    ontology = build_ontology(df, domain)

    # Stage 3: Metrics
    metrics = discover_metrics(df, ontology, domain)

    # Stage 4: Patterns
    patterns = detect_patterns(df, ontology, metrics)

    # Stage 5: Risks
    risks = assess_risks(df, ontology, metrics, patterns)

    # Stage 6: Recommendations
    recs = generate_recommendations(metrics, patterns, risks, ontology, domain)

    # Stage 7: Narrative
    narrative = generate_narrative(domain_info, metrics, patterns, risks, recs, ontology, docs)

    # Build backward-compatible findings list for the existing card UI
    findings = []
    for r in risks:
        findings.append({
            "type": "risk",
            "severity": "critical" if r["impact"] == "high" else "warning",
            "title": r["risk"],
            "value": r["likelihood"].title() + " Likelihood",
            "delta": "",
            "direction": "down",
            "evidence": r["evidence"],
            "action": r["mitigation"],
        })
    for p in patterns:
        findings.append({
            "type": "comparison" if p["type"] in ("concentration", "imbalance") else "anomaly",
            "severity": "warning" if p["magnitude"] == "high" else "info",
            "title": p["description"],
            "value": p["magnitude"].title() + " Impact",
            "delta": "",
            "direction": "flat",
            "evidence": p["evidence"],
            "action": None,
        })
    for rec in recs:
        findings.append({
            "type": "recommendation",
            "severity": "critical" if rec["priority"] == "high" else "info",
            "title": rec["action"],
            "value": rec["priority"].title() + " Priority",
            "delta": "",
            "direction": "up",
            "evidence": rec["evidence"],
            "action": rec["mitigation"],
        })

    critical = sum(1 for f in findings if f["severity"] == "critical")
    warnings = sum(1 for f in findings if f["severity"] == "warning")
    positive = sum(1 for f in findings if f["severity"] == "info")

    return {
        "domain": domain_info,
        "ontology": ontology,
        "metrics": metrics,
        "patterns": patterns,
        "risks": risks,
        "recommendations": recs,
        "narrative": narrative,
        "findings": findings,
        "critical_count": critical,
        "warning_count": warnings,
        "opportunity_count": positive,
        "total": len(findings),
        "summary": (
            f"{domain_info['label']} · {len(findings)} insights · "
            f"{critical} critical · {warnings} warnings"
        ),
    }

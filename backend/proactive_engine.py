"""
Proactive Analysis Engine — Zarva's autonomous analyst.
Runs without user prompting; generates anomalies, trends, predictions,
comparisons, risk signals, and LLM recommendations the moment data is uploaded.
"""
import re
import json
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from backend.model_router import get_model


def _money(v):
    v = float(v)
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.0f}K"
    return f"${v:,.2f}"


def _pct(a, b):
    if not b or b == 0: return None
    return round((a - b) / abs(b) * 100, 1)


def _num_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _cat_cols(df):
    return [c for c in df.columns if df[c].dtype == object and df[c].nunique() <= 25]


def _date_cols(df):
    return [c for c in df.columns
            if "date" in c.lower() or "time" in c.lower()
            or pd.api.types.is_datetime64_any_dtype(df[c])]


def _clean(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


# ── 1. Anomaly detection ──────────────────────────────────────────────────────

def _anomaly_detection(df):
    findings = []
    for col in _num_cols(df):
        s = _clean(df[col]).dropna()
        if len(s) < 8: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue
        upper = q3 + 2.5 * iqr
        lower = q1 - 2.5 * iqr
        outliers = s[(s > upper) | (s < lower)]
        if outliers.empty: continue
        n = len(outliers)
        max_out = outliers.max()
        pct = _pct(max_out, s.median())
        severity = "critical" if n <= 2 else "warning"
        findings.append({
            "type": "anomaly",
            "severity": severity,
            "title": f"{n} outlier{'s' if n > 1 else ''} in {col}",
            "metric": col,
            "value": _money(max_out) if max_out > 100 else f"{max_out:,.2f}",
            "delta": f"{pct:+.0f}% vs median" if pct else "",
            "direction": "up" if max_out > s.median() else "down",
            "evidence": (
                f"Expected {_money(lower)}–{_money(upper)}; "
                f"{n} record{'s' if n > 1 else ''} outside this range (max: {_money(max_out)})"
            ),
            "action": f"Review the {n} {col} outlier{'s' if n > 1 else ''} — may indicate errors or exceptional activity"
        })
        if len(findings) >= 3:
            break
    return findings


# ── 2. Trend detection ────────────────────────────────────────────────────────

def _trend_detection(df):
    findings = []
    date_col = next((c for c in _date_cols(df)), None)
    if not date_col: return []
    num = _num_cols(df)
    if not num: return []

    df2 = df.copy()
    df2["_dt"] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=["_dt"])
    df2["_period"] = df2["_dt"].dt.to_period("M")
    if df2["_period"].nunique() < 3: return []

    for col in num[:4]:
        s = _clean(df2[col])
        df2["_v"] = s
        monthly = df2.dropna(subset=["_v"]).groupby("_period")["_v"].sum()
        if len(monthly) < 3: continue

        # compare last period to first period
        first, last = float(monthly.iloc[0]), float(monthly.iloc[-1])
        pct = _pct(last, first)
        if pct is None or abs(pct) < 5: continue

        direction = "up" if pct > 0 else "down"
        severity = "positive" if pct > 0 else ("warning" if pct < -15 else "info")

        # month-over-month for last 2 periods
        mom = _pct(float(monthly.iloc[-1]), float(monthly.iloc[-2]))
        mom_str = f" (MoM: {mom:+.0f}%)" if mom is not None else ""

        findings.append({
            "type": "trend",
            "severity": severity,
            "title": f"{col} {'↑' if direction == 'up' else '↓'} {abs(pct):.0f}% over {len(monthly)} months",
            "metric": col,
            "value": _money(last) if last > 100 else f"{last:,.2f}",
            "delta": f"{pct:+.1f}%{mom_str}",
            "direction": direction,
            "evidence": (
                f"{monthly.index[0]} → {monthly.index[-1]}: "
                f"{_money(first)} → {_money(last)}"
            ),
            "action": (
                f"{'Sustain momentum in' if pct > 20 else 'Monitor growth in' if pct > 0 else 'Investigate decline in'} "
                f"{col} — {abs(pct):.0f}% shift over the period"
            )
        })
        if len(findings) >= 2:
            break
    return findings


# ── 3. Comparison insights ────────────────────────────────────────────────────

def _comparison_insights(df):
    findings = []
    cats = _cat_cols(df)
    nums = _num_cols(df)
    if not cats or not nums: return []

    # prefer cols that look like segments/stages/categories
    priority_kw = ["stage", "status", "category", "type", "segment", "region", "owner", "group"]
    cat_col = next(
        (c for c in df.columns if any(k in c.lower() for k in priority_kw) and c in cats),
        cats[0]
    )
    amount_kw = ["amount", "revenue", "value", "price", "sales", "total", "pay"]
    num_col = next(
        (c for c in nums if any(k in c.lower() for k in amount_kw)),
        nums[0]
    )

    grp = df.groupby(cat_col).agg(
        count=(num_col, "count"),
        avg=(num_col, lambda x: _clean(x).mean()),
        total=(num_col, lambda x: _clean(x).sum())
    ).dropna()
    grp = grp[grp["count"] >= 2].sort_values("avg", ascending=False)
    if len(grp) < 2: return []

    top_lbl, bot_lbl = grp.index[0], grp.index[-1]
    top_avg, bot_avg = grp.loc[top_lbl, "avg"], grp.loc[bot_lbl, "avg"]
    ratio = round(top_avg / bot_avg, 1) if bot_avg and bot_avg > 0 else None

    if ratio and ratio >= 1.5:
        findings.append({
            "type": "comparison",
            "severity": "info",
            "title": f"'{top_lbl}' outperforms '{bot_lbl}' by {ratio}×",
            "metric": f"{num_col} by {cat_col}",
            "value": f"{ratio}× gap",
            "delta": f"+{_pct(top_avg, bot_avg):.0f}%" if _pct(top_avg, bot_avg) else "",
            "direction": "up",
            "evidence": (
                f"Best: {top_lbl} avg {_money(top_avg)} · "
                f"Worst: {bot_lbl} avg {_money(bot_avg)} across {cat_col}"
            ),
            "action": (
                f"Study what drives '{top_lbl}' — replicate across other {cat_col} segments"
            )
        })

    # top 3 vs bottom 3
    if len(grp) >= 6:
        top3_avg = grp["avg"].iloc[:3].mean()
        bot3_avg = grp["avg"].iloc[-3:].mean()
        ratio3 = round(top3_avg / bot3_avg, 1) if bot3_avg > 0 else None
        if ratio3 and ratio3 >= 2:
            findings.append({
                "type": "comparison",
                "severity": "warning",
                "title": f"Top 3 {cat_col}s earn {ratio3}× bottom 3",
                "metric": f"{num_col} concentration",
                "value": f"{ratio3}× spread",
                "delta": "",
                "direction": "flat",
                "evidence": (
                    f"Top 3 avg: {_money(top3_avg)} · Bottom 3 avg: {_money(bot3_avg)} — "
                    f"significant performance gap"
                ),
                "action": f"Address performance gap — bottom {cat_col} segments need targeted support"
            })
    return findings[:2]


# ── 4. Predictions ────────────────────────────────────────────────────────────

def _prediction_insights(df):
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return []

    findings = []
    date_col = next((c for c in _date_cols(df)), None)
    nums = _num_cols(df)
    if not date_col or not nums: return []

    amount_kw = ["amount", "revenue", "value", "price", "sales", "total"]
    num_col = next((c for c in nums if any(k in c.lower() for k in amount_kw)), nums[0])

    df2 = df.copy()
    df2["_dt"] = pd.to_datetime(df2[date_col], errors="coerce")
    df2["_v"] = _clean(df2[num_col])
    df2["_p"] = df2["_dt"].dt.to_period("M")
    monthly = df2.dropna(subset=["_dt", "_v"]).groupby("_p")["_v"].sum()
    if len(monthly) < 4: return []

    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly.values.astype(float)
    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    if r2 < 0.45: return []

    next_val = float(model.predict([[len(monthly)]])[0])
    pct = _pct(next_val, float(monthly.iloc[-1]))
    direction = "up" if next_val > monthly.iloc[-1] else "down"

    findings.append({
        "type": "prediction",
        "severity": "positive" if direction == "up" else "warning",
        "title": f"{num_col} forecast: {_money(next_val)} next period",
        "metric": num_col,
        "value": _money(next_val),
        "delta": f"{pct:+.1f}%" if pct else "",
        "direction": direction,
        "evidence": (
            f"Linear model trained on {len(monthly)} months (R²={r2:.2f}) — "
            f"current: {_money(float(monthly.iloc[-1]))} → predicted: {_money(next_val)}"
        ),
        "action": (
            f"Plan for {_money(next_val)} in {num_col} next month — "
            f"{'prepare resources for growth' if direction == 'up' else 'review pipeline to prevent decline'}"
        )
    })
    return findings


# ── 5. Risk scoring ───────────────────────────────────────────────────────────

def _risk_scoring(df):
    findings = []
    nums = _num_cols(df)

    signals = []
    for col in nums:
        s = _clean(df[col]).dropna()
        if len(s) < 5: continue
        cv = float(s.std() / s.mean()) if s.mean() != 0 else 0
        if cv > 1.2:
            signals.append(f"{col} is highly volatile (CV={cv:.1f})")
        miss = float(df[col].isna().mean() * 100)
        if miss > 15:
            signals.append(f"{col} has {miss:.0f}% missing values")

    # Concentration risk
    cats = _cat_cols(df)
    if cats and nums:
        num_col = nums[0]
        cat_col = cats[0]
        s = _clean(df[num_col])
        grp = df.groupby(cat_col).apply(lambda g: _clean(g[num_col]).sum())
        total = s.sum()
        if total > 0:
            top_share = float(grp.max() / total * 100)
            if top_share > 60:
                signals.append(
                    f"Top {cat_col} accounts for {top_share:.0f}% of {num_col} — concentration risk"
                )

    if signals:
        severity = "critical" if len(signals) >= 3 else "warning" if len(signals) >= 2 else "info"
        findings.append({
            "type": "risk",
            "severity": severity,
            "title": f"{len(signals)} risk signal{'s' if len(signals) > 1 else ''} detected",
            "metric": "Data & Business Risk",
            "value": f"{len(signals)} signals",
            "delta": "",
            "direction": "flat",
            "evidence": " · ".join(signals[:3]),
            "action": "Address risk signals before making strategic decisions from this data"
        })
    return findings


# ── 6. LLM Recommendations ───────────────────────────────────────────────────

def _llm_recommendations(findings, df, docs, doc_type=None):
    try:
        context_parts = []
        is_extracted = df.attrs.get("is_extracted", False) if df is not None else False
        resolved_type = doc_type or (df.attrs.get("document_type", "") if df is not None else "") or "data"

        if is_extracted and df is not None and "label" in df.columns and "value" in df.columns:
            # For extracted docs: give the LLM the actual labelled rows
            rows = df[["label", "value", "unit", "category"]].head(20).to_dict("records") if "unit" in df.columns else df[["label", "value"]].head(20).to_dict("records")
            context_parts.append(f"Document type: {resolved_type}")
            context_parts.append("Document data:\n" + "\n".join(
                f"  {r.get('label','')}: {r.get('value','')} {r.get('unit','')}" for r in rows
            ))
        else:
            if df is not None:
                nums = _num_cols(df)
                cats = _cat_cols(df)
                context_parts.append(
                    f"Dataset type: {resolved_type} — {len(df)} rows, {len(df.columns)} cols. "
                    f"Numeric: {nums[:6]}. Categorical: {cats[:6]}."
                )
                for col in nums[:3]:
                    s = _clean(df[col]).dropna()
                    if not s.empty:
                        context_parts.append(
                            f"{col}: mean={_money(s.mean())}, max={_money(s.max())}, min={_money(s.min())}"
                        )

        if findings:
            context_parts.append("Already detected: " + "; ".join(f["title"] for f in findings[:6]))
        if docs and not is_extracted:
            context_parts.append("Document excerpt: " + " ".join(docs[:2])[:400])

        context = "\n".join(context_parts) or "Limited data available."

        prompt = f"""You are an expert analyst. A user uploaded a document and you must give 3 specific insights.

{context}

Generate exactly 3 insights as a JSON array. Each insight must:
1. Be specific to THIS document — reference exact figures, names, or items from the data above
2. Be relevant to the document type ("{resolved_type}") — flight booking → travel insights, payroll → tax/deduction insights, recipe → nutrition/ingredient insights, contract → obligation/deadline insights
3. Be actionable — tell the user what to do or what to watch for

[
  {{
    "title": "Short action-verb phrase under 8 words",
    "evidence": "Cite the exact figure or fact from the data",
    "action": "One specific next step",
    "priority": "high|medium|low",
    "impact": "revenue|risk|efficiency|quality"
  }}
]

Rules:
- NEVER recommend Excel, Google Sheets, Power BI or any external tool
- NEVER write generic advice — every insight must cite a specific number or name
- Match the domain: don't give business KPI advice for a flight booking or recipe
Return only valid JSON."""

        resp = get_model("reason").invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m: return []

        recs = json.loads(m.group())
        priority_sev = {"high": "critical", "medium": "warning", "low": "info"}
        impact_icons = {"revenue": "💰", "risk": "⚠️", "efficiency": "⚡", "quality": "✅"}

        result = []
        for r in recs[:3]:
            impact = r.get("impact", "efficiency")
            result.append({
                "type": "recommendation",
                "severity": priority_sev.get(r.get("priority", "medium"), "info"),
                "title": r.get("title", "Review data"),
                "metric": f"{impact_icons.get(impact, '💡')} {impact.title()} opportunity",
                "value": r.get("priority", "medium").title() + " Priority",
                "delta": "",
                "direction": "up",
                "evidence": r.get("evidence", ""),
                "action": r.get("action", "")
            })
        return result
    except Exception:
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

def _doc_insights(df: pd.DataFrame) -> list[dict]:
    """
    Insights for extracted documents (label/value/category/unit schema).
    Compares within the same unit to avoid mixing $ amounts with percentages.
    """
    findings = []
    doc_type = df.attrs.get("document_type", "document")

    # Group by category, summarise totals
    if "category" in df.columns and "value" in df.columns and "unit" in df.columns:
        for unit in df["unit"].unique():
            sub = df[df["unit"] == unit].copy()
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub.dropna(subset=["value"])
            if len(sub) < 2:
                continue

            for cat, grp in sub.groupby("category"):
                total = grp["value"].sum()
                if total == 0:
                    continue
                top = grp.nlargest(1, "value").iloc[0]
                findings.append({
                    "type": "comparison",
                    "severity": "info",
                    "title": f"{cat.title()} total: {_money(total) if unit in ('USD','EUR','GBP') else f'{total:,.2f} {unit}'}",
                    "metric": f"{cat} ({unit})",
                    "value": _money(total) if unit in ("USD","EUR","GBP") else f"{total:,.2f} {unit}",
                    "delta": "",
                    "direction": "flat",
                    "evidence": f"Largest item: {top['label']} = {_money(top['value']) if unit in ('USD','EUR','GBP') else f'{top[chr(118)]:,.2f} {unit}'}",
                    "action": f"Review {cat} items in this {doc_type}"
                })
            if len(findings) >= 4:
                break

    return findings[:4]


def run_proactive_analysis(df=None, docs=None, doc_type=None):
    findings = []

    if df is not None and len(df) >= 5:
        is_extracted = df.attrs.get("is_extracted", False)

        if is_extracted:
            # Extracted docs (PDF/image): use unit-aware doc insights only
            # Never run generic anomaly/trend detection — it mixes units and produces nonsense
            findings += _doc_insights(df)
        else:
            # CSV / structured data: full analytics suite
            findings += _anomaly_detection(df)
            findings += _trend_detection(df)
            findings += _comparison_insights(df)
            findings += _prediction_insights(df)
            findings += _risk_scoring(df)

    findings += _llm_recommendations(findings, df, docs, doc_type=doc_type)

    critical = sum(1 for f in findings if f["severity"] == "critical")
    warnings_ = sum(1 for f in findings if f["severity"] == "warning")
    positive  = sum(1 for f in findings if f["severity"] in ("positive", "info"))

    return {
        "findings": findings,
        "critical_count": critical,
        "warning_count": warnings_,
        "opportunity_count": positive,
        "total": len(findings),
        "summary": (
            f"Found {len(findings)} insight{'s' if len(findings) != 1 else ''} · "
            f"{critical} critical · {warnings_} warnings · {positive} opportunities"
        )
    }

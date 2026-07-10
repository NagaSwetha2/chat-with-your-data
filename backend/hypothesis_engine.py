"""
Hypothesis Engine — Zarva's business reasoning core.
Proactively forms hypotheses, tests them with real data, explains evidence.
No LLM for numbers — pandas only. LLM only writes the plain-English narrative.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.model_router import get_model


def _fast_llm():
    # Narration: just verbalising pre-computed evidence — cheap model is fine
    return get_model("narrate")


def _find_col(df, *keywords):
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    return None


def _numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


_SENTIMENT_POS = ["excited","fantastic","great","excellent","thrilled","amazing","outstanding","happy","wonderful","positive","love","satisfied"]
_SENTIMENT_NEG = ["disappointed","frustrated","risk","concerned","poor","negative","struggling","terrible","awful","angry","unhappy","issue","problem","escalat","cancel","loss","delayed"]

def _sentiment(text):
    t = str(text).lower()
    p = sum(1 for w in _SENTIMENT_POS if w in t)
    n = sum(1 for w in _SENTIMENT_NEG if w in t)
    return (p - n) / (p + n) if (p + n) > 0 else 0.0


def _narrate(hypothesis, evidence_lines, action):
    """LLM writes a 2-sentence plain-English explanation of the evidence."""
    prompt = ChatPromptTemplate.from_messages([("system", """You are Zarva. Write exactly 2 short sentences explaining this business finding.
Be direct. Use the numbers given. No filler. No "based on the data". Sound like a sharp analyst."""),
        ("human", f"Hypothesis: {hypothesis}\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence_lines) + f"\nAction: {action}")
    ])
    try:
        return (prompt | _fast_llm() | StrOutputParser()).invoke({})
    except Exception:
        return " ".join(evidence_lines[:2])


# ── Individual hypothesis tests ────────────────────────────────────────────

def _h_owner_performance(df):
    """Who is significantly underperforming and why?"""
    owner_col = _find_col(df, "owner", "rep", "assignee", "engineer", "agent")
    amt_col   = _find_col(df, "amount", "revenue", "value")
    stage_col = _find_col(df, "stage", "status", "state")
    if not owner_col or not stage_col:
        return None

    owners = df[owner_col].dropna()
    if owners.nunique() < 2:
        return None

    # win rate per owner
    won_mask = df[stage_col].fillna("").str.lower().str.contains("won|resolved|complete|closed won", regex=True)
    grp = df.groupby(owner_col)
    win_rates = grp.apply(lambda g: won_mask.loc[g.index].mean()).sort_values()
    if len(win_rates) < 2:
        return None

    bottom = win_rates.index[0]
    top    = win_rates.index[-1]
    bottom_rate = round(win_rates.iloc[0] * 100, 1)
    top_rate    = round(win_rates.iloc[-1] * 100, 1)
    team_avg    = round(won_mask.mean() * 100, 1)
    gap = top_rate - bottom_rate
    if gap < 15:
        return None

    bottom_count = int(grp.size()[bottom])
    evidence = [
        f"{bottom} has a {bottom_rate}% win rate vs team average of {team_avg}%",
        f"{top} leads at {top_rate}% — a {gap:.0f}pt gap",
        f"{bottom} has {bottom_count} deals",
    ]

    if amt_col:
        bottom_amt = round(_numeric(df[df[owner_col] == bottom][amt_col]).mean(), 0)
        team_amt   = round(_numeric(df[amt_col]).mean(), 0)
        evidence.append(f"{bottom}'s avg deal size ${bottom_amt:,.0f} vs team ${team_amt:,.0f}")

    action = f"Review {bottom}'s last 5 lost deals — identify if it's deal selection, negotiation, or support gap"
    narrative = _narrate(f"{bottom} is significantly underperforming peers", evidence, action)

    return {
        "title": f"{bottom} is underperforming — {gap:.0f}pt win rate gap",
        "hypothesis": f"One owner has a significantly lower win rate than peers, suggesting a coachable pattern.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": min(95, 60 + int(gap)),
        "impact": "high" if gap > 30 else "medium",
        "action": action,
        "category": "Performance Gap",
    }


def _h_stage_stall(df):
    """Are deals stuck in a particular stage?"""
    stage_col = _find_col(df, "stage", "status", "state")
    name_col  = _find_col(df, "account", "name", "company", "subject", "title")
    if not stage_col:
        return None

    stages = df[stage_col].fillna("unknown")
    # exclude terminal stages
    open_mask = ~stages.str.lower().str.contains("won|lost|resolved|complete|closed|cancelled", regex=True)
    open_df = df[open_mask]
    if len(open_df) < 5:
        return None

    stage_counts = open_df[stage_col].value_counts()
    dominant_stage = stage_counts.index[0]
    dominant_count = int(stage_counts.iloc[0])
    dominant_pct   = round(dominant_count / len(open_df) * 100, 1)

    if dominant_pct < 35:
        return None

    # sentiment in dominant stage
    text_col = _find_col(df, "notes", "note", "description", "comments", "subject")
    neg_pct = None
    if text_col:
        stuck = open_df[open_df[stage_col] == dominant_stage]
        sentiments = stuck[text_col].fillna("").astype(str).apply(_sentiment)
        neg_pct = round((sentiments < 0).mean() * 100, 1)

    evidence = [
        f"{dominant_pct}% of open deals ({dominant_count} of {len(open_df)}) are stuck in '{dominant_stage}'",
        f"This suggests a bottleneck — deals are entering but not progressing",
    ]
    if neg_pct is not None:
        evidence.append(f"{neg_pct}% of '{dominant_stage}' records show negative sentiment")

    if name_col:
        examples = open_df[open_df[stage_col] == dominant_stage][name_col].dropna().head(3).tolist()
        if examples:
            evidence.append(f"Examples: {', '.join(str(e) for e in examples)}")

    action = f"Audit all '{dominant_stage}' deals — set a 2-week exit deadline or escalate"
    narrative = _narrate(f"Deals are bottlenecking in {dominant_stage}", evidence, action)

    return {
        "title": f"Pipeline bottleneck — {dominant_pct}% of deals stuck in '{dominant_stage}'",
        "hypothesis": f"A disproportionate share of open deals are stalled in one stage, indicating a process or resource bottleneck.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": min(93, 55 + int(dominant_pct)),
        "impact": "high" if dominant_pct > 50 else "medium",
        "action": action,
        "category": "Stage Bottleneck",
    }


def _h_sentiment_outcome(df):
    """Do negative notes actually predict lost deals?"""
    stage_col = _find_col(df, "stage", "status", "state")
    text_col  = _find_col(df, "notes", "note", "description", "comments", "subject")
    if not stage_col or not text_col:
        return None

    df2 = df.copy()
    df2["_sent"] = df2[text_col].fillna("").astype(str).apply(_sentiment)
    df2["_won"]  = df2[stage_col].fillna("").str.lower().str.contains("won|resolved|complete", regex=True).astype(int)
    df2["_lost"] = df2[stage_col].fillna("").str.lower().str.contains("lost|cancelled|rejected", regex=True).astype(int)

    closed = df2[df2["_won"] | df2["_lost"].astype(bool)]
    if len(closed) < 10:
        return None

    won_sent  = round(closed[closed["_won"] == 1]["_sent"].mean(), 2)
    lost_sent = round(closed[closed["_lost"] == 1]["_sent"].mean(), 2)
    diff = won_sent - lost_sent

    if abs(diff) < 0.1:
        return None

    open_neg = df2[~(df2["_won"].astype(bool) | df2["_lost"].astype(bool)) & (df2["_sent"] < -0.2)]
    at_risk_count = len(open_neg)

    evidence = [
        f"Won deals average sentiment score: {won_sent:+.2f}",
        f"Lost deals average sentiment score: {lost_sent:+.2f} — a {diff:.2f} gap",
        f"{at_risk_count} currently open deals have negative sentiment matching the 'lost' pattern",
    ]

    action = f"Review the {at_risk_count} open deals with negative notes — intervene before they close lost"
    narrative = _narrate("Negative sentiment in notes reliably predicts deal loss", evidence, action)

    return {
        "title": f"Negative notes predict lost deals — {at_risk_count} open deals at risk",
        "hypothesis": "Sentiment in deal notes is a leading indicator of outcome — negative language appears before losses.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": min(90, 50 + int(abs(diff) * 100)),
        "impact": "high" if at_risk_count > 5 else "medium",
        "action": action,
        "category": "Sentiment Signal",
    }


def _h_deal_size_risk(df):
    """Do large deals close at lower rates?"""
    amt_col   = _find_col(df, "amount", "revenue", "value")
    stage_col = _find_col(df, "stage", "status", "state")
    if not amt_col or not stage_col:
        return None

    df2 = df.copy()
    df2["_amt"] = _numeric(df2[amt_col])
    df2["_won"] = df2[stage_col].fillna("").str.lower().str.contains("won|resolved|complete", regex=True)
    df2 = df2.dropna(subset=["_amt"])

    if len(df2) < 10:
        return None

    median_amt = df2["_amt"].median()
    large = df2[df2["_amt"] > median_amt * 1.5]
    small = df2[df2["_amt"] <= median_amt * 1.5]

    if len(large) < 3 or len(small) < 3:
        return None

    large_win = round(large["_won"].mean() * 100, 1)
    small_win = round(small["_won"].mean() * 100, 1)
    gap = small_win - large_win

    if gap < 10:
        return None

    large_open_amt = large[~large["_won"]]["_amt"].sum()

    evidence = [
        f"Deals above ${median_amt * 1.5:,.0f} win at {large_win}% vs {small_win}% for smaller deals",
        f"{gap:.0f}pt win rate gap — large deals are significantly harder to close",
        f"${large_open_amt:,.0f} in large open deals currently at higher risk",
    ]

    action = "Assign senior AE or executive sponsor to all deals above the median size threshold"
    narrative = _narrate("Large deals underperform smaller ones in win rate", evidence, action)

    return {
        "title": f"Large deals win {gap:.0f}% less often — ${large_open_amt:,.0f} at risk",
        "hypothesis": "Deal size is inversely correlated with win rate, suggesting larger deals need different handling.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": min(88, 55 + int(gap)),
        "impact": "high",
        "action": action,
        "category": "Deal Size Risk",
    }


def _h_time_trend(df):
    """Is performance improving or declining over time?"""
    date_col  = _find_col(df, "closedate", "close_date", "createddate", "created_date", "date")
    stage_col = _find_col(df, "stage", "status", "state")
    amt_col   = _find_col(df, "amount", "revenue", "value")
    if not date_col:
        return None

    df2 = df.copy()
    df2["_date"] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=["_date"])
    if len(df2) < 10:
        return None

    df2["_month"] = df2["_date"].dt.to_period("M")
    monthly = df2.groupby("_month").size()
    if len(monthly) < 3:
        return None

    # linear trend
    x = np.arange(len(monthly))
    slope = np.polyfit(x, monthly.values, 1)[0]
    direction = "growing" if slope > 0 else "declining"

    recent_2 = monthly.iloc[-2:].mean()
    earlier_2 = monthly.iloc[:2].mean()
    change_pct = round((recent_2 - earlier_2) / (earlier_2 + 1e-9) * 100, 1)

    peak_month = str(monthly.idxmax())
    peak_count = int(monthly.max())

    evidence = [
        f"Record volume is {direction} — slope of {slope:+.1f} records/month",
        f"Recent 2-month avg: {recent_2:.0f} records vs earliest 2-month avg: {earlier_2:.0f}",
        f"Peak activity was {peak_month} with {peak_count} records",
    ]

    if amt_col and stage_col:
        won_mask = df2[stage_col].fillna("").str.lower().str.contains("won|resolved|complete", regex=True)
        recent_mask = df2["_date"] >= df2["_date"].quantile(0.75)
        recent_win = round(won_mask[recent_mask].mean() * 100, 1)
        overall_win = round(won_mask.mean() * 100, 1)
        evidence.append(f"Recent win rate {recent_win}% vs overall {overall_win}%")

    action = "Focus pipeline review on the declining months — identify what changed"
    narrative = _narrate(f"Activity is {direction} over time", evidence, action)

    return {
        "title": f"Pipeline activity is {direction} — {abs(change_pct):.0f}% {'increase' if change_pct > 0 else 'decline'}",
        "hypothesis": f"Record volume shows a clear {'upward' if slope > 0 else 'downward'} trend over the observed period.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": 75,
        "impact": "medium",
        "action": action,
        "category": "Trend Analysis",
    }


def _h_priority_neglect(df):
    """Are high-priority records being resolved slower or at lower rates?"""
    pri_col   = _find_col(df, "priority")
    stage_col = _find_col(df, "stage", "status", "state")
    if not pri_col or not stage_col:
        return None

    df2 = df.copy()
    high_mask = df2[pri_col].fillna("").str.lower().str.contains("high|critical|urgent", regex=True)
    resolved_mask = df2[stage_col].fillna("").str.lower().str.contains("resolved|won|complete|closed", regex=True)

    if high_mask.sum() < 3:
        return None

    high_res = round(resolved_mask[high_mask].mean() * 100, 1)
    low_res  = round(resolved_mask[~high_mask].mean() * 100, 1)
    gap = low_res - high_res

    if gap < 10:
        return None

    open_high = int((high_mask & ~resolved_mask).sum())

    evidence = [
        f"High-priority records resolve at {high_res}% vs {low_res}% for others",
        f"{gap:.0f}pt gap — high-priority items are being deprioritized in practice",
        f"{open_high} high-priority items currently unresolved",
    ]

    action = f"Escalate all {open_high} open high-priority items — assign dedicated owner with 48hr SLA"
    narrative = _narrate("High-priority records paradoxically resolve slower", evidence, action)

    return {
        "title": f"High-priority items resolve {gap:.0f}% less often — {open_high} open",
        "hypothesis": "Despite being flagged as high-priority, these records have worse resolution rates than normal ones.",
        "narrative": narrative,
        "evidence": evidence,
        "confidence": min(90, 55 + int(gap)),
        "impact": "high",
        "action": action,
        "category": "Priority Neglect",
    }


# ── Document mode (PDF / image / text — no DataFrame) ──────────────────────

def run_doc_hypotheses(docs: list) -> dict:
    """Generate insights from raw document text chunks using LLM reasoning."""
    full_text = "\n".join(docs[:30])[:6000]

    prompt = ChatPromptTemplate.from_messages([("system", """You are Zarva, a sharp document analyst.
Read the document content and generate 3 key findings/insights.

For each finding return a JSON object with these exact fields:
- title: short finding title (max 10 words)
- narrative: 2 sentences explaining the finding with specific facts from the document
- evidence: list of 3-4 specific bullet points with actual numbers/dates/names from the document
- action: one actionable recommendation
- confidence: number 65-90
- impact: "high" or "medium"
- category: topic area (e.g. "Financial Summary", "Risk Flag", "Key Date", "Tax Analysis", "Payroll", "Travel", "Contract Term")

Return ONLY a valid JSON array of 3 objects. No markdown, no explanation."""),
        ("human", f"Document content:\n{full_text}")
    ])

    try:
        raw = (get_model("hypothesize") | StrOutputParser()).invoke(
            prompt.format_messages()
        )
        import json, re
        raw = re.sub(r"```json|```", "", raw).strip()
        findings = json.loads(raw)
        if isinstance(findings, list):
            return {"hypotheses": findings[:3]}
    except Exception as e:
        pass

    return {"hypotheses": []}


# ── Main entry ─────────────────────────────────────────────────────────────

def run_hypotheses(df: pd.DataFrame) -> dict:
    if len(df) < 8:
        return {"hypotheses": [], "error": "Need more records to form hypotheses"}

    tests = [
        _h_owner_performance,
        _h_stage_stall,
        _h_sentiment_outcome,
        _h_deal_size_risk,
        _h_time_trend,
        _h_priority_neglect,
    ]

    results = []
    for test in tests:
        try:
            h = test(df)
            if h:
                results.append(h)
        except Exception:
            continue

    # sort by confidence desc, cap at 4
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {"hypotheses": results[:4]}

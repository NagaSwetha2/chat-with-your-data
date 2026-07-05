import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textblob import TextBlob


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def _sentiment(text: str) -> str:
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    return "Neutral"


COLORS = {"Positive": "#16a34a", "Neutral": "#94a3b8", "Negative": "#ef4444"}
BAR_COLOR = "#16a34a"
ACCENT = "#1a6e3c"


def generate_charts(df: pd.DataFrame) -> dict:
    charts = {}

    # pick text column for sentiment
    text_col = next((c for c in df.columns if df[c].dtype == object), None)
    if text_col:
        df["_sentiment"] = df[text_col].apply(_sentiment)

        # 1. Sentiment pie chart
        counts = df["_sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=counts.index,
               colors=[COLORS.get(l, "#94a3b8") for l in counts.index],
               autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Sentiment Distribution", fontweight="bold", color=ACCENT)
        charts["sentiment_pie"] = _to_base64(fig)

    # 2. Revenue / Amount bar by sentiment
    amount_col = next((c for c in df.columns if "amount" in c.lower() or "revenue" in c.lower() or "price" in c.lower()), None)
    if amount_col and "_sentiment" in df.columns:
        try:
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
            avg = df.groupby("_sentiment")[amount_col].mean().reindex(["Positive", "Neutral", "Negative"]).dropna()
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(avg.index, avg.values,
                          color=[COLORS.get(l, "#94a3b8") for l in avg.index],
                          edgecolor="white", linewidth=1.5)
            ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
            ax.set_title(f"Avg {amount_col} by Sentiment", fontweight="bold", color=ACCENT)
            ax.set_ylabel(amount_col)
            ax.spines[["top", "right"]].set_visible(False)
            charts["revenue_by_sentiment"] = _to_base64(fig)
        except Exception:
            pass

    # 3. Records by Stage / Status
    stage_col = next((c for c in df.columns if "stage" in c.lower() or "status" in c.lower()), None)
    if stage_col:
        counts = df[stage_col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color=BAR_COLOR, edgecolor="white")
        ax.bar_label(bars, padding=4, fontsize=9)
        ax.set_title(f"Records by {stage_col}", fontweight="bold", color=ACCENT)
        ax.spines[["top", "right"]].set_visible(False)
        charts["by_stage"] = _to_base64(fig)

    # 4. Top owners / assignees
    owner_col = next((c for c in df.columns if "owner" in c.lower() or "assignee" in c.lower() or "rep" in c.lower()), None)
    if owner_col:
        counts = df[owner_col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#1a6e3c", edgecolor="white")
        ax.bar_label(bars, padding=4, fontsize=9)
        ax.set_title(f"Records by {owner_col}", fontweight="bold", color=ACCENT)
        ax.spines[["top", "right"]].set_visible(False)
        charts["by_owner"] = _to_base64(fig)

    return charts

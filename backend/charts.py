"""
Chart engine — dark-themed, full chart-type library.
Supports: bar, pie, line, area, scatter, bubble, candlestick,
          waterfall, radar, violin, funnel, heatmap, histogram.
"""
import io
import re as _re
import base64
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection
import os
import warnings
warnings.filterwarnings("ignore")

# ── dark theme ────────────────────────────────────────────────────────────
BG      = "#0d0d0d"
CARD_BG = "#141414"
GRID_BG = "#1a1a1a"
GREEN   = "#22c55e"
GREEN2  = "#16a34a"
TEXT    = "#e5e7eb"
MUTED   = "#6b7280"
RED     = "#ef4444"
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
PURPLE  = "#a78bfa"
PALETTE = ["#22c55e","#3b82f6","#f59e0b","#ef4444","#a78bfa",
           "#06b6d4","#f97316","#ec4899","#84cc16","#14b8a6",
           "#8b5cf6","#f43f5e","#10b981","#6366f1","#fbbf24"]


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=145,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _dark(fig, ax):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(GRID_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(GREEN)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")
    ax.spines[["top","right"]].set_visible(False)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color="#2a2a2a", linewidth=0.5, linestyle="--", alpha=0.6)


# ─────────────────────────── primitive charts ────────────────────────────

def _bar_chart(series: pd.Series, title: str, xlabel="", ylabel="Count") -> str:
    top = series.head(12)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(top) * 0.46)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1],
                   height=0.65, edgecolor="none")
    ax.bar_label(bars, fmt="%g", padding=5, fontsize=8.5, color=TEXT)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel(ylabel, fontsize=9)
    _dark(fig, ax); _grid(ax, "x")
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _pie_chart(series: pd.Series, title: str) -> str:
    top = series.head(8)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(CARD_BG)
    wedges, _, autotexts = ax.pie(
        top.values, labels=None, colors=PALETTE[:len(top)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": CARD_BG, "linewidth": 2},
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
    ax.legend(wedges, top.index, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, fontsize=8.5, frameon=False, labelcolor=TEXT, facecolor=CARD_BG)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12, color=GREEN)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _line_chart(x, y, title: str, xlabel="", ylabel="") -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color=GREEN, linewidth=2.5, marker="o", markersize=5,
            markerfacecolor=GREEN2, markeredgecolor=CARD_BG, markeredgewidth=1.5)
    ax.fill_between(range(len(y)), y, alpha=0.12, color=GREEN)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel(ylabel, fontsize=9)
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _area_chart(x, y, title: str, ylabel="") -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(range(len(y)), y, alpha=0.35, color=GREEN)
    ax.plot(range(len(y)), y, color=GREEN, linewidth=2)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel(ylabel, fontsize=9)
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _scatter_chart(x: pd.Series, y: pd.Series, title: str,
                   xlabel="", ylabel="", size_col: pd.Series = None) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = None
    if size_col is not None:
        s_clean = pd.to_numeric(size_col, errors="coerce").fillna(size_col.median())
        sizes = ((s_clean - s_clean.min()) / (s_clean.max() - s_clean.min() + 1e-9) * 250 + 20)
    ax.scatter(x, y, s=sizes if sizes is not None else 40,
               color=GREEN, alpha=0.6, edgecolors=CARD_BG, linewidths=0.4)
    # trend line
    try:
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, color=AMBER, linewidth=1.5,
                linestyle="--", alpha=0.8, label="Trend")
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    except Exception:
        pass
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    _dark(fig, ax); _grid(ax, "both")
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _bubble_chart(x: pd.Series, y: pd.Series, size: pd.Series, title: str,
                  xlabel="", ylabel="") -> str:
    """Bubble = scatter where bubble area encodes a third numeric variable."""
    fig, ax = plt.subplots(figsize=(7, 5))
    s_norm = (size - size.min()) / (size.max() - size.min() + 1e-9) * 600 + 20
    sc = ax.scatter(x, y, s=s_norm, c=s_norm, cmap="YlGn", alpha=0.7,
                    edgecolors="#2a2a2a", linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    _dark(fig, ax); _grid(ax, "both")
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _waterfall_chart(labels: list, values: list, title: str) -> str:
    """Running-total waterfall — great for payroll gross → deductions → net."""
    running = 0
    bottoms, heights, colors_w = [], [], []
    for v in values:
        bottoms.append(running if v >= 0 else running + v)
        heights.append(abs(v))
        colors_w.append(GREEN if v >= 0 else RED)
        running += v

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.75), 5))
    fig.patch.set_facecolor(CARD_BG)
    bars = ax.bar(range(len(labels)), heights, bottom=bottoms,
                  color=colors_w, edgecolor=CARD_BG, linewidth=0.5, width=0.65)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ypos = bar.get_y() + bar.get_height() + (ax.get_ylim()[1] * 0.01)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{'−' if v < 0 else '+'}${abs(v):,.0f}",
                ha="center", va="bottom", fontsize=7.5, color=TEXT, fontweight="bold")
    # connector lines
    cum = 0
    for i, v in enumerate(values[:-1]):
        cum += v
        ax.plot([i + 0.32, i + 0.68], [cum, cum],
                color=MUTED, linewidth=0.8, linestyle="--")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(l)[:14] for l in labels], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Amount ($)", fontsize=9)

    # legend
    pos_patch = mpatches.Patch(color=GREEN, label="Addition")
    neg_patch = mpatches.Patch(color=RED, label="Deduction")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, frameon=False,
              labelcolor=TEXT, loc="upper right")
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _candlestick_chart(df: pd.DataFrame, open_c, high_c, low_c, close_c,
                       date_c, title: str) -> str:
    """Pure-matplotlib candlestick — no external lib needed."""
    df2 = df[[date_c, open_c, high_c, low_c, close_c]].copy()
    df2[date_c] = pd.to_datetime(df2[date_c], errors="coerce")
    df2 = df2.dropna().sort_values(date_c).tail(60)
    if len(df2) < 2:
        return ""

    for col in [open_c, high_c, low_c, close_c]:
        df2[col] = pd.to_numeric(
            df2[col].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce"
        )
    df2 = df2.dropna()

    fig, ax = plt.subplots(figsize=(max(8, len(df2) * 0.2), 5))
    fig.patch.set_facecolor(CARD_BG)

    for i, (_, row) in enumerate(df2.iterrows()):
        o, h, l, c = row[open_c], row[high_c], row[low_c], row[close_c]
        color = GREEN if c >= o else RED
        # wick
        ax.plot([i, i], [l, h], color=color, linewidth=1, alpha=0.8)
        # body
        body_h = abs(c - o) or (h - l) * 0.05
        body_b = min(c, o)
        ax.add_patch(Rectangle((i - 0.35, body_b), 0.7, body_h,
                                facecolor=color, edgecolor=CARD_BG, linewidth=0.3))

    n = len(df2)
    tick_step = max(1, n // 8)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(
        [str(df2[date_c].iloc[i])[:10] for i in range(0, n, tick_step)],
        rotation=35, ha="right", fontsize=7.5
    )
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Price", fontsize=9)

    g_patch = mpatches.Patch(color=GREEN, label="Bullish")
    r_patch = mpatches.Patch(color=RED,   label="Bearish")
    ax.legend(handles=[g_patch, r_patch], fontsize=8, frameon=False,
              labelcolor=TEXT, loc="upper left")
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _radar_chart(categories: list, values: list, title: str) -> str:
    """Spider/radar chart for multi-metric comparison."""
    n = len(categories)
    if n < 3:
        return ""
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    vals = [float(v) for v in values]
    mx = max(vals) or 1
    norm = [v / mx for v in vals]
    norm += norm[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(GRID_BG)
    ax.plot(angles, norm, color=GREEN, linewidth=2)
    ax.fill(angles, norm, color=GREEN, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([str(c)[:12] for c in categories],
                       fontsize=8, color=TEXT)
    ax.set_yticklabels([]); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(color="#2a2a2a", linewidth=0.6, linestyle="--")
    ax.spines["polar"].set_edgecolor("#2a2a2a")
    ax.set_title(title, fontweight="bold", fontsize=13, pad=18, color=GREEN)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _violin_chart(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> str:
    groups = df[cat_col].value_counts().head(8).index.tolist()
    data = [pd.to_numeric(df[df[cat_col] == g][num_col], errors="coerce").dropna().values
            for g in groups]
    data = [d for d in data if len(d) >= 3]
    if not data:
        return ""
    fig, ax = plt.subplots(figsize=(max(5, len(data) * 0.9), 5))
    fig.patch.set_facecolor(CARD_BG)
    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.5)
        pc.set_edgecolor("#2a2a2a")
    parts["cmedians"].set_color(GREEN); parts["cmedians"].set_linewidth(2)
    parts["cmins"].set_color(MUTED);   parts["cmaxes"].set_color(MUTED)
    parts["cbars"].set_color(MUTED);   parts["cbars"].set_linewidth(0.8)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([str(g)[:12] for g in groups[:len(data)]],
                       rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel(num_col, fontsize=9)
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _funnel_chart(labels: list, values: list, title: str) -> str:
    vals = [float(v) for v in values]
    mx = max(vals) or 1
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(labels) * 0.65)))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(GRID_BG)
    for i, (label, val) in enumerate(zip(labels, vals)):
        pct = val / mx
        color = PALETTE[i % len(PALETTE)]
        # centred horizontal bar
        ax.barh(len(labels) - 1 - i, pct * 2, left=1 - pct,
                height=0.7, color=color, edgecolor=CARD_BG, linewidth=0.5, alpha=0.85)
        ax.text(1, len(labels) - 1 - i, f"  {label}: {val:,.0f}",
                ha="left", va="center", fontsize=9, color=TEXT, fontweight="500")
    ax.set_xlim(0, 2)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12, color=GREEN)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _heatmap_chart(df: pd.DataFrame, row_col: str, col_col: str,
                   val_col: str, title: str) -> str:
    pivot = df.pivot_table(values=val_col, index=row_col, columns=col_col,
                           aggfunc="sum", fill_value=0)
    pivot = pivot.iloc[:12, :12]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gd", [BG, GREEN2, GREEN], N=256)
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 0.7),
                                    max(4, len(pivot) * 0.55)))
    fig.patch.set_facecolor(CARD_BG)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([str(c)[:10] for c in pivot.columns],
                       rotation=35, ha="right", fontsize=7.5, color=MUTED)
    ax.set_yticklabels([str(r)[:14] for r in pivot.index],
                       fontsize=7.5, color=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=10, color=GREEN)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(colors=MUTED, labelsize=7)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


def _histogram_chart(series: pd.Series, title: str) -> str:
    clean = pd.to_numeric(series.astype(str).str.replace(r"[$,]","",regex=True),
                          errors="coerce").dropna()
    if len(clean) < 3:
        return ""
    fig, ax = plt.subplots(figsize=(7, 4))
    n_bins = min(30, max(10, len(clean)//5))
    ax.hist(clean, bins=n_bins, color=GREEN, edgecolor=CARD_BG, linewidth=0.4, alpha=0.85)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(clean, bw_method="scott")
        xr = np.linspace(clean.min(), clean.max(), 200)
        counts, edges = np.histogram(clean, bins=n_bins)
        bw = edges[1] - edges[0]
        ax.plot(xr, kde(xr) * len(clean) * bw, color="#4ade80", linewidth=2, zorder=5)
    except Exception:
        pass
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    _dark(fig, ax); _grid(ax)
    plt.tight_layout(pad=1.5)
    return _to_b64(fig)


# ─────────────────────── NLP routing helpers ────────────────────────────

_ANSWER_PATTERNS = [
    # "Adobe : $473,154" or "Oracle : $450,000" — dollar amounts without decimals
    _re.compile(r"\*{0,2}([A-Za-z][A-Za-z0-9\s\-/&]{1,35}?)\*{0,2}\s*:\s*\$\s*([\d,]+)(?:\.\d{1,2})?\b", _re.IGNORECASE),
    # "January 2026: 73 opportunities (7.3%)" — counts with optional suffix/pct
    _re.compile(r"\*{0,2}([A-Za-z][A-Za-z0-9\s\-/]{1,30}?)\*{0,2}\s*:\s*([\d,]+)\s*(?:records?|items?|cases?|entries?|occurrences?|responses?|users?|customers?|deals?|opportunities?)?\s*(?:\([\d.]+%\))?(?:\s*[-–]\s*\w+)?", _re.IGNORECASE),
    # "Label $1,234.56" — dollar inline
    _re.compile(r"([A-Za-z][A-Za-z0-9\s\-/&]{2,35}?)\s+\$\s*([\d,]+(?:\.\d{1,2})?)\b"),
    # plain decimal "Label 123.45"
    _re.compile(r"([A-Za-z][A-Za-z0-9\s\-/]{3,35}?)\s+([\d,]+\.\d{2})\b"),
]

_NOISE_LABELS = {
    "total", "sum", "count", "average", "avg", "mean", "max", "min",
    "gross", "net", "ytd", "year to date", "grand total", "overall",
}


def _extract_answer_context(question: str) -> dict:
    """Parse label→value pairs from the answer-context block the frontend appends."""
    marker = "The assistant already computed this data:"
    idx = question.find(marker)
    if idx == -1:
        return {}
    context_text = question[idx + len(marker):]
    # Stop before previous-conversation context to avoid bleeding in unrelated numbers
    stop = context_text.find("Context from previous conversation:")
    if stop != -1:
        context_text = context_text[:stop]
    candidates: dict[str, float] = {}
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(context_text):
            label = m.group(1).strip().title()
            val_str = m.group(2).replace(",", "")
            if label.lower() in _NOISE_LABELS or len(label) < 2:
                continue
            try:
                val = float(val_str)
                if val > 0:
                    if label not in candidates or val > candidates[label]:
                        candidates[label] = val
            except ValueError:
                pass
    return candidates if len(candidates) >= 2 else {}


def _normalize_q(question: str) -> str:
    """Return normalised first-line only — ignore context appended by frontend."""
    q = question.split('\n')[0].lower()
    for pat, rep in [
        (r'\bpi\b',       'pie'),
        (r'\bbart\b',     'bar'),
        (r'\bchar\b',     'chart'),
        (r'\bvisualise\b','visualize'),
        (r'\bscaters\b',  'scatter'),
        (r'\bcandles\b',  'candlestick'),
        (r'\bwaterfal\b', 'waterfall'),
    ]:
        q = _re.sub(pat, rep, q)
    return q


def _detect_chart_type(q: str) -> str:
    # Explicit chart type keywords take highest priority
    if any(w in q for w in ["pie chart","pie graph","donut chart","pie"]):
        return "pie"
    if any(w in q for w in ["bar chart","bar graph","horizontal bar","vertical bar"]):
        return "bar"
    if any(w in q for w in ["line chart","line graph","line plot"]):
        return "line"
    if any(w in q for w in ["area chart","area graph","stacked area","filled line"]):
        return "area"
    # Specialised chart types
    if any(w in q for w in ["candlestick","candle","ohlc","stock price"]):
        return "candlestick"
    if any(w in q for w in ["waterfall","cascade","bridge","gross to net","net from gross"]):
        return "waterfall"
    if any(w in q for w in ["scatter","correlation","relationship between","vs ","versus"]):
        return "scatter"
    if any(w in q for w in ["bubble"]):
        return "bubble"
    if any(w in q for w in ["radar","spider","web chart","multi-axis"]):
        return "radar"
    if any(w in q for w in ["violin","density","kde by","distribution by","distribution per"]):
        return "violin"
    if any(w in q for w in ["funnel","conversion","pipeline stages","stage funnel"]):
        return "funnel"
    if any(w in q for w in ["heatmap","heat map","calendar heatmap","crosstab"]):
        return "heatmap"
    if any(w in q for w in ["histogram","frequency","bins","distribution of"]):
        return "histogram"
    # Context hints (lower priority — only when no explicit type given)
    if any(w in q for w in ["trend","over time","monthly","by month","by year","growth","by week","by quarter"]):
        return "line"
    if any(w in q for w in ["proportion","share","breakdown","split","donut"]):
        return "pie"
    return "bar"  # default


# ─────────────────── CSV chart (DataFrame path) ─────────────────────────

def _col_relevance(df: pd.DataFrame, col: str, q_words: set) -> int:
    """Score how relevant a column is to the question — purely from the data itself."""
    score = 0
    col_words = set(_re.split(r'[\s_\-]+', col.lower()))
    # Column name matches question words
    score += len(col_words & q_words) * 3
    # Partial substring match (e.g. "stage" matches "stagename")
    for w in q_words:
        if len(w) >= 4 and w in col.lower():
            score += 1
    # Values in the column match question words (finds "proposal" inside Stage column)
    if df[col].dtype == object:
        sample = df[col].dropna().astype(str).str.lower().unique()[:30]
        for val in sample:
            val_words = set(_re.split(r'[\s_\-]+', val))
            score += len(val_words & q_words) * 2
    return score


def generate_dynamic_chart(df: pd.DataFrame, question: str) -> dict:
    q = _normalize_q(question)
    chart_type = _detect_chart_type(q)

    # Classify columns purely from data shape — no hardcoded names
    cat_cols  = [c for c in df.columns
                 if df[c].dtype == object or df[c].nunique() <= max(10, len(df) // 50)]
    num_cols  = [c for c in df.columns
                 if pd.api.types.is_numeric_dtype(df[c]) and c not in cat_cols]
    date_cols = [c for c in df.columns
                 if pd.api.types.is_datetime64_any_dtype(df[c]) or
                 (df[c].dtype == object and "date" in c.lower())]

    # Score every column by relevance to the question
    q_words = set(_re.split(r'\W+', q)) - {"the","a","an","of","in","for","by","to","is","are","per","how","many","show","me","what","give","draw","chart","graph","plot","pie","bar","line"}
    cat_scores = {c: _col_relevance(df, c, q_words) for c in cat_cols}
    num_scores = {c: _col_relevance(df, c, q_words) for c in num_cols}

    # Pick best columns — highest relevance score wins, no hardcoded fallback lists
    best_cat = max(cat_scores, key=cat_scores.get) if cat_scores else None
    best_num = max(num_scores, key=num_scores.get) if num_scores else None

    # If a column name is literally in the question, that always wins
    q_nospace = q.replace(" ", "")
    for c in df.columns:
        if c.lower() in q or c.lower().replace(" ","") in q_nospace:
            if c in cat_cols:
                best_cat = c
            elif c in num_cols:
                best_num = c

    def clean_num(col):
        return pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce"
        )

    # ── candlestick ────────────────────────────────────────────────────
    if chart_type == "candlestick":
        ohlc = {}
        for role, keywords in [
            ("open",  ["open","opening"]),
            ("high",  ["high","max","maximum"]),
            ("low",   ["low","min","minimum"]),
            ("close", ["close","closing","last","price"]),
        ]:
            ohlc[role] = next((c for c in num_cols
                               if any(k in c.lower() for k in keywords)), None)
        date_c = date_cols[0] if date_cols else None
        if all(ohlc.values()) and date_c:
            b64 = _candlestick_chart(df, ohlc["open"], ohlc["high"],
                                     ohlc["low"], ohlc["close"], date_c,
                                     "Candlestick Chart")
            if b64:
                return {"chart": b64}
        # fallback to line if no OHLC
        chart_type = "line"

    # ── waterfall ──────────────────────────────────────────────────────
    if chart_type == "waterfall":
        if best_num and best_cat:
            grp = df.groupby(best_cat)[best_num].sum().head(10)
            if len(grp) >= 2:
                return {"chart": _waterfall_chart(list(grp.index), list(grp.values),
                                                 f"Waterfall — {best_num} by {best_cat}")}
        chart_type = "bar"

    # ── scatter / bubble ───────────────────────────────────────────────
    if chart_type in ("scatter", "bubble"):
        vs_m = _re.search(r'([\w\s]+?)\s+vs\.?\s+([\w\s]+?)(?:\s+(?:chart|plot|graph)|\s*$)', q)
        asked_num = []
        if vs_m:
            for part in [vs_m.group(1).strip(), vs_m.group(2).strip()]:
                hit = next((c for c in df.columns
                            if c.lower() == part or c.lower().replace(" ","") == part.replace(" ","")), None)
                if hit and hit in num_cols:
                    asked_num.append(hit)
        if len(num_cols) >= 2:
            x_col = asked_num[0] if asked_num else best_num or num_cols[0]
            y_col = asked_num[1] if len(asked_num) >= 2 else next((c for c in num_cols if c != x_col), num_cols[1])
            sx = clean_num(x_col).fillna(0)
            sy = clean_num(y_col).fillna(0)
            mask = sx.notna() & sy.notna()
            if chart_type == "bubble" and len(num_cols) >= 3:
                sz_col = next((c for c in num_cols if c not in [x_col, y_col]), None)
                if sz_col:
                    sz = clean_num(sz_col).fillna(0)
                    return {"chart": _bubble_chart(sx[mask], sy[mask], sz[mask],
                                                   f"Bubble — {x_col} / {y_col} / {sz_col}", x_col, y_col)}
            return {"chart": _scatter_chart(sx[mask], sy[mask], f"{x_col} vs {y_col}", x_col, y_col)}

    # ── radar ──────────────────────────────────────────────────────────
    if chart_type == "radar":
        if num_cols:
            means = df[num_cols[:8]].mean()
            return {"chart": _radar_chart(list(means.index), list(means.values), "Radar — Avg by Feature")}

    # ── violin ─────────────────────────────────────────────────────────
    if chart_type == "violin":
        if best_cat and best_num:
            return {"chart": _violin_chart(df, best_cat, best_num, f"Violin — {best_num} by {best_cat}")}

    # ── funnel ─────────────────────────────────────────────────────────
    if chart_type == "funnel":
        if best_cat:
            counts = df[best_cat].value_counts().head(8)
            return {"chart": _funnel_chart(list(counts.index), list(counts.values), f"Funnel — {best_cat}")}

    # ── heatmap ────────────────────────────────────────────────────────
    if chart_type == "heatmap":
        if len(cat_cols) >= 2 and num_cols:
            c1 = max(cat_scores, key=cat_scores.get) if cat_scores else cat_cols[0]
            c2 = next((c for c in sorted(cat_cols, key=lambda x: cat_scores.get(x,0), reverse=True) if c != c1), cat_cols[-1])
            return {"chart": _heatmap_chart(df, c1, c2, best_num or num_cols[0], f"Heatmap — {best_num or num_cols[0]}")}

    # ── histogram ──────────────────────────────────────────────────────
    if chart_type == "histogram":
        if best_num:
            return {"chart": _histogram_chart(df[best_num], f"Distribution of {best_num}")}

    # ── area / line ────────────────────────────────────────────────────
    if chart_type in ("area", "line"):
        date_col = next((c for c in date_cols), None)
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
            monthly = parsed.dt.to_period("M").value_counts().sort_index()
            if len(monthly) > 1:
                fn = _area_chart if chart_type == "area" else _line_chart
                return {"chart": fn([str(p) for p in monthly.index], list(monthly.values),
                                    f"Records Over Time ({date_col})", ylabel="Count")}

    # ── pie ────────────────────────────────────────────────────────────
    if chart_type == "pie":
        if best_cat:
            vc = df[best_cat].value_counts()
            if len(vc) >= 2:
                return {"chart": _pie_chart(vc, f"{best_cat} Distribution")}
        if num_cols:
            means = df[num_cols[:8]].mean().sort_values(ascending=False)
            if len(means) >= 2:
                return {"chart": _pie_chart(means, "Numeric Breakdown")}

    # ── default bar ────────────────────────────────────────────────────
    if best_cat:
        if best_num and any(w in q for w in ["by","per","group","each","sum","total","amount","revenue","value"]):
            grouped = df.groupby(best_cat)[best_num].sum().sort_values(ascending=False).head(12)
            return {"chart": _bar_chart(grouped, f"{best_num} by {best_cat}", ylabel=best_num)}
        # Default: count per category (handles "how many X per stage", "bar by status", etc.)
        return {"chart": _bar_chart(df[best_cat].value_counts().head(12), f"Records by {best_cat}")}

    if best_num:
        return {"chart": _histogram_chart(df[best_num], f"Distribution of {best_num}")}

    return {}


# ─────────────────── PDF / text chart (no DataFrame) ────────────────────

def generate_dynamic_chart_from_text(docs: list, question: str) -> dict:
    """Extract numbers from doc text and render the best chart type."""
    # Check for pre-computed answer context first (works for PDF path too)
    answer_data = _extract_answer_context(question)
    if answer_data:
        q0 = _normalize_q(question)
        ct = _detect_chart_type(q0)
        series = pd.Series(answer_data).sort_values(ascending=False)
        title = question.split('\n')[0].strip().title()
        if ct == "pie" or len(series) <= 6:
            return {"chart": _pie_chart(series, title)}
        return {"chart": _bar_chart(series, title, ylabel="Count")}

    q = _normalize_q(question)
    chart_type = _detect_chart_type(q)

    full_text = "\n".join(docs[:40])

    # Extract label→value pairs — dollar amounts, decimals, and integer counts
    patterns = [
        _re.compile(r"([A-Za-z][A-Za-z\s\-/]{3,35}?)\s*\$\s*([\d,]+\.\d{2})\b"),         # explicit $
        _re.compile(r"([A-Za-z][A-Za-z\s\-/]{3,35}?)\s+([\d,]+\.\d{2})\b"),                 # plain .XX decimal
        _re.compile(r"([A-Za-z][A-Za-z\s\-/]{2,30}?)\s*[:\-]\s*([\d,]{1,9})\s*(?:records?|items?|cases?|entries?|occurrences?|units?|times?|\()", _re.IGNORECASE),  # "Label: 365 records"
        _re.compile(r"^\s*([A-Za-z][A-Za-z\s\-/]{2,30}?)\s*[:\-]\s*([\d,]{2,9})\s*$", _re.MULTILINE),  # "Label: 365" on its own line
    ]
    candidates = {}
    for pat in patterns:
        for m in pat.finditer(full_text):
            label, val_str = m.group(1).strip().title(), m.group(2)
            if label.lower() in _NOISE_LABELS or len(label) < 2:
                continue
            try:
                val = float(val_str.replace(",", ""))
                if 0.01 < val < 10_000_000 and len(label) > 2:
                    if label not in candidates or val > candidates[label]:
                        candidates[label] = val
            except Exception:
                pass

    if not candidates:
        return {}

    # Keyword filters for deduction/tax queries
    deduction_kw = ["deduct","withhold","benefit","contribution","insurance",
                    "retire","401","fica","oasdi","medicare","federal","social"]
    # Exclude gross/income-base/aggregate rows — clean substrings only (no regex patterns)
    exclude_kw   = [
        "taxable wages", "taxable wage",
        "gross", "ytd", "year to date",
        "total wages", "total earnings",
        "total tax", "annual sa", "payroll address",
    ]

    def _is_excluded(label: str) -> bool:
        lo = label.lower().strip()
        if lo in ("wages","wage","salary","taxes","tax","earnings","income","gross pay","gross"):
            return True
        return any(w in lo for w in exclude_kw)

    ask_deductions = any(w in q for w in deduction_kw + ["deduction","tax"])
    if ask_deductions:
        filtered = {k: v for k, v in candidates.items()
                    if any(w in k.lower() for w in deduction_kw)
                    and not _is_excluded(k)}
        if filtered:
            candidates = filtered
        else:
            # fallback: at least strip the pure-gross rows
            candidates = {k: v for k, v in candidates.items() if not _is_excluded(k)}

    # Chart types that need DataFrame columns — can't work from extracted text numbers
    _NEEDS_DF = {"scatter", "bubble", "violin", "heatmap", "candlestick"}
    if chart_type in _NEEDS_DF:
        return {}  # rag_engine already returned a helpful text message; don't render a bar chart

    series = pd.Series(candidates).sort_values(ascending=False).head(12)
    labels = list(series.index)
    values = list(series.values)
    title  = "Deductions Breakdown" if ask_deductions else "Breakdown"

    # Waterfall is the richest view for payroll deductions
    if chart_type == "waterfall" or (ask_deductions and chart_type not in ("pie","bar","radar")):
        if len(labels) >= 2:
            wf_vals = [-v for v in values]
            if len(wf_vals) >= 2:
                return {"chart": _waterfall_chart(labels, wf_vals, f"Waterfall — {title}")}

    if chart_type == "radar" and len(labels) >= 3:
        return {"chart": _radar_chart(labels[:8], values[:8], title)}

    if chart_type == "funnel":
        return {"chart": _funnel_chart(labels, values, title)}

    if chart_type == "pie":
        if len(series) >= 2:
            return {"chart": _pie_chart(series, title)}
        return {"chart": _bar_chart(series, title, ylabel="Amount ($)")}

    # default bar
    return {"chart": _bar_chart(series, title, ylabel="Amount ($)")}


# ─────────────── pre-built dashboard charts (CSV upload) ────────────────

def generate_charts(df: pd.DataFrame) -> dict:
    charts = {}

    def _sentiment(text):
        t = str(text).lower()
        pos = sum(1 for w in ["excited","fantastic","great","excellent","thrilled",
                               "amazing","outstanding","happy","wonderful","love"] if w in t)
        neg = sum(1 for w in ["disappointed","frustrated","risk","concerned","poor",
                               "negative","struggling","terrible","awful","angry"] if w in t)
        return "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"

    text_col = next(
        (c for c in df.columns if c.lower() in ["notes","note","description","comments","text","body"]),
        next((c for c in df.columns if df[c].dtype == object
              and df[c].str.len().mean() > 20), None)
    )
    if text_col:
        df2 = df.copy()
        df2["_sent"] = df2[text_col].apply(_sentiment)
        charts["sentiment_pie"] = _pie_chart(df2["_sent"].value_counts(), "Sentiment Distribution")

    stage_col = next((c for c in df.columns if "stage" in c.lower() or "status" in c.lower()), None)
    if stage_col:
        charts["by_stage"] = _bar_chart(df[stage_col].value_counts(), f"Records by {stage_col}")

    owner_col = next((c for c in df.columns if "owner" in c.lower() or "assignee" in c.lower()), None)
    if owner_col:
        charts["by_owner"] = _bar_chart(df[owner_col].value_counts(), f"Records by {owner_col}")

    return charts

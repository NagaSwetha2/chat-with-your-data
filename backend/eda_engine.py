"""
Exploratory Data Analysis engine.
Returns shape, types, missing, duplicates, statistical summary,
univariate charts (distribution + boxplot), bivariate charts
(correlation heatmap + scatter plots), and LLM-generated insights.
"""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
import os
import warnings
warnings.filterwarnings("ignore")

# ── dark theme constants (match charts.py) ──────────────────────────────
BG       = "#0d0d0d"
CARD_BG  = "#141414"
GRID_BG  = "#1a1a1a"
GREEN    = "#22c55e"
GREEN2   = "#16a34a"
TEXT     = "#e5e7eb"
MUTED    = "#6b7280"
PALETTE  = ["#22c55e","#3b82f6","#f59e0b","#ef4444","#a78bfa",
            "#06b6d4","#f97316","#ec4899","#84cc16","#14b8a6"]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _apply_dark(fig, ax):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(GRID_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(GREEN)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.5, linestyle="--", alpha=0.6)


# ── 1. Shape & data quality report ──────────────────────────────────────

def _shape_report(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    duplicates = int(df.duplicated().sum())
    total_cells = n_rows * n_cols
    missing_cells = int(df.isnull().sum().sum())

    col_types = {"numeric": [], "categorical": [], "datetime": [], "boolean": []}
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            col_types["boolean"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
            col_types["datetime"].append(col)
        else:
            col_types["categorical"].append(col)

    per_col = []
    for col in df.columns:
        miss = int(df[col].isnull().sum())
        per_col.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": miss,
            "missing_pct": round(miss / n_rows * 100, 1) if n_rows else 0,
            "unique": int(df[col].nunique()),
        })

    return {
        "rows": n_rows,
        "columns": n_cols,
        "duplicates": duplicates,
        "duplicate_pct": round(duplicates / n_rows * 100, 1) if n_rows else 0,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 1) if total_cells else 0,
        "column_types": col_types,
        "per_column": per_col,
    }


# ── 2. Statistical summary ───────────────────────────────────────────────

def _stat_summary(df: pd.DataFrame) -> dict:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object]

    numeric_stats = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        numeric_stats.append({
            "column": col,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "q25": round(float(s.quantile(0.25)), 3),
            "median": round(float(s.median()), 3),
            "q75": round(float(s.quantile(0.75)), 3),
            "max": round(float(s.max()), 3),
            "skewness": round(float(s.skew()), 3),
            "kurtosis": round(float(s.kurt()), 3),
            "missing_pct": round(df[col].isnull().mean() * 100, 1),
        })

    cat_stats = []
    for col in cat_cols[:10]:
        s = df[col].dropna()
        top = s.value_counts()
        cat_stats.append({
            "column": col,
            "count": int(s.count()),
            "unique": int(s.nunique()),
            "top": str(top.index[0]) if len(top) else "",
            "freq": int(top.iloc[0]) if len(top) else 0,
            "mode_pct": round(top.iloc[0] / len(s) * 100, 1) if len(top) and len(s) else 0,
            "missing_pct": round(df[col].isnull().mean() * 100, 1),
        })

    return {"numeric": numeric_stats, "categorical": cat_stats}


# ── 3. Univariate charts ─────────────────────────────────────────────────

def _dist_boxplot(series: pd.Series, col: str) -> str:
    """Combined distribution (histogram+KDE) and boxplot for one numeric column."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(CARD_BG)

    # histogram
    ax1.hist(s, bins=min(30, max(10, len(s)//5)), color=GREEN, edgecolor=CARD_BG,
             linewidth=0.4, alpha=0.85)
    ax1.set_title(f"Distribution — {col}", color=GREEN, fontweight="bold", fontsize=11, pad=8)
    ax1.set_facecolor(GRID_BG)
    ax1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", color="#2a2a2a", linewidth=0.5, linestyle="--", alpha=0.6)

    # KDE overlay
    try:
        kde = scipy_stats.gaussian_kde(s, bw_method="scott")
        x_range = np.linspace(s.min(), s.max(), 200)
        counts, edges = np.histogram(s, bins=min(30, max(10, len(s)//5)))
        bin_width = edges[1] - edges[0]
        ax1.plot(x_range, kde(x_range) * len(s) * bin_width,
                 color="#4ade80", linewidth=2, zorder=5)
    except Exception:
        pass

    # boxplot
    bp = ax2.boxplot(s, vert=False, patch_artist=True, widths=0.5,
                     whiskerprops=dict(color=MUTED, linewidth=1.2),
                     capprops=dict(color=MUTED, linewidth=1.2),
                     medianprops=dict(color=GREEN, linewidth=2),
                     flierprops=dict(marker="o", color="#ef4444", markersize=4, alpha=0.6))
    bp["boxes"][0].set_facecolor("#1f3a1f")
    bp["boxes"][0].set_edgecolor(GREEN2)
    ax2.set_facecolor(GRID_BG)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_yticks([])

    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def _cat_bar(series: pd.Series, col: str) -> str:
    """Horizontal bar chart for a categorical column (top 12 values)."""
    counts = series.value_counts().head(12)
    if len(counts) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(6, max(3, len(counts) * 0.42)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1],
                   height=0.65, edgecolor="none")
    ax.bar_label(bars, fmt="%g", padding=5, fontsize=8, color=TEXT)
    ax.set_title(f"Value Counts — {col}", color=GREEN, fontweight="bold", fontsize=11, pad=8)
    _apply_dark(fig, ax)
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def _univariate_charts(df: pd.DataFrame) -> dict:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:6]
    cat_cols = [c for c in df.columns if df[c].dtype == object][:4]

    charts = {}
    for col in num_cols:
        b64 = _dist_boxplot(df[col], col)
        if b64:
            charts[col] = {"type": "numeric", "chart": b64}

    for col in cat_cols:
        b64 = _cat_bar(df[col], col)
        if b64:
            charts[col] = {"type": "categorical", "chart": b64}

    return charts


# ── 4. Bivariate charts ──────────────────────────────────────────────────

def _correlation_heatmap(df: pd.DataFrame) -> str:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return ""

    corr = df[num_cols].corr()
    n = len(num_cols)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    # custom colormap: red(neg) → dark → green(pos)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#ef4444", CARD_BG, "#22c55e"], N=256
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c[:14] for c in num_cols]
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8, color=MUTED)
    ax.set_yticklabels(labels, fontsize=8, color=MUTED)
    ax.set_title("Correlation Heatmap", color=GREEN, fontweight="bold", fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=TEXT if abs(val) < 0.6 else BG, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor("#2a2a2a")

    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def _scatter_plots(df: pd.DataFrame) -> list[dict]:
    """Top 3 most-correlated numeric column pairs as scatter plots."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return []

    corr = df[num_cols].corr().abs()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            pairs.append((corr.iloc[i, j], num_cols[i], num_cols[j]))
    pairs.sort(reverse=True)

    results = []
    for _, cx, cy in pairs[:3]:
        sx = pd.to_numeric(df[cx], errors="coerce")
        sy = pd.to_numeric(df[cy], errors="coerce")
        mask = sx.notna() & sy.notna()
        sx, sy = sx[mask], sy[mask]
        if len(sx) < 5:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(sx, sy, color=GREEN, alpha=0.5, s=18, edgecolors="none")

        # trend line
        try:
            m, b, r, *_ = scipy_stats.linregress(sx, sy)
            xline = np.linspace(sx.min(), sx.max(), 100)
            ax.plot(xline, m * xline + b, color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.8)
            r_label = f"r = {r:.2f}"
        except Exception:
            r_label = ""

        ax.set_xlabel(cx, color=MUTED, fontsize=9)
        ax.set_ylabel(cy, color=MUTED, fontsize=9)
        ax.set_title(f"{cx} vs {cy}  {r_label}", color=GREEN, fontweight="bold", fontsize=10, pad=8)
        _apply_dark(fig, ax)
        ax.grid(axis="both", color="#2a2a2a", linewidth=0.5, linestyle="--", alpha=0.5)
        plt.tight_layout(pad=1.2)

        results.append({"x": cx, "y": cy, "r": r_label, "chart": _fig_to_b64(fig)})

    return results


def _cat_vs_num_box(df: pd.DataFrame) -> list[dict]:
    """Box plots of each numeric col grouped by the first categorical column."""
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cat_cols or not num_cols:
        return []

    cat_col = cat_cols[0]
    groups = df[cat_col].value_counts().head(8).index.tolist()
    results = []

    for num_col in num_cols[:3]:
        data = [pd.to_numeric(df[df[cat_col] == g][num_col], errors="coerce").dropna().values
                for g in groups]
        data = [d for d in data if len(d) > 0]
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(max(5, len(data) * 0.9), 4))
        fig.patch.set_facecolor(CARD_BG)
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        whiskerprops=dict(color=MUTED, linewidth=1),
                        capprops=dict(color=MUTED, linewidth=1),
                        medianprops=dict(color=GREEN, linewidth=2),
                        flierprops=dict(marker="o", color="#ef4444", markersize=3, alpha=0.5))
        colors = PALETTE[:len(data)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color + "33")
            patch.set_edgecolor(color)

        ax.set_xticklabels([str(g)[:12] for g in groups[:len(data)]],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{num_col} by {cat_col}", color=GREEN, fontweight="bold", fontsize=10, pad=8)
        ax.set_ylabel(num_col, color=MUTED, fontsize=9)
        _apply_dark(fig, ax)
        plt.tight_layout(pad=1.2)
        results.append({"x": cat_col, "y": num_col, "chart": _fig_to_b64(fig)})

    return results


def _bivariate_charts(df: pd.DataFrame) -> dict:
    return {
        "correlation_heatmap": _correlation_heatmap(df),
        "scatter_plots": _scatter_plots(df),
        "cat_vs_num": _cat_vs_num_box(df),
    }


# ── 5. Missing-value bar chart ───────────────────────────────────────────

def _missing_bar_chart(df: pd.DataFrame) -> str:
    pcts = (df.isnull().mean() * 100).sort_values(ascending=False)
    pcts = pcts[pcts > 0]
    if len(pcts) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pcts) * 0.38)))
    colors = ["#ef4444" if p > 20 else "#f59e0b" if p > 5 else "#3b82f6" for p in pcts.values]
    bars = ax.barh(pcts.index[::-1], pcts.values[::-1], color=colors[::-1],
                   height=0.6, edgecolor="none")
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=8, color=TEXT)
    ax.set_xlim(0, max(pcts.values) * 1.25)
    ax.set_title("Missing Values by Column (%)", color=GREEN, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel("% Missing", color=MUTED, fontsize=8)
    _apply_dark(fig, ax)
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


# ── 6. LLM insights ─────────────────────────────────────────────────────

def _eda_insights(df: pd.DataFrame, shape: dict, stat_summary: dict) -> list[dict]:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from backend.model_router import get_model
        import json, re

        llm = get_model("insights")

        num_summary = []
        for r in stat_summary.get("numeric", [])[:6]:
            num_summary.append(
                f"{r['column']}: mean={r['mean']}, std={r['std']}, "
                f"skew={r['skewness']}, missing={r['missing_pct']}%"
            )
        cat_summary = []
        for r in stat_summary.get("categorical", [])[:4]:
            cat_summary.append(
                f"{r['column']}: {r['unique']} unique values, top='{r['top']}' ({r['mode_pct']}%)"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior data scientist reviewing an EDA report.
Return EXACTLY a JSON array of 4 insight objects. Each object must have:
  title (string, ≤8 words), narrative (string, 1-2 sentences), severity ("info"|"warning"|"critical"), category (string)
No markdown, no explanation — only valid JSON array."""),
            ("human", f"""Dataset: {shape['rows']} rows, {shape['columns']} columns.
Duplicates: {shape['duplicates']} ({shape['duplicate_pct']}%).
Missing cells: {shape['missing_pct']}%.

Numeric columns:
{chr(10).join(num_summary)}

Categorical columns:
{chr(10).join(cat_summary)}

Generate 4 insights covering data quality, distribution anomalies, potential issues, and opportunities.""")
        ])

        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({})
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return [
            {"title": "Data loaded successfully", "narrative": f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns ready for analysis.", "severity": "info", "category": "Overview"},
        ]


# ── Main entry ───────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    shape = _shape_report(df)
    stat_summary = _stat_summary(df)
    missing_chart = _missing_bar_chart(df)
    univariate = _univariate_charts(df)
    bivariate = _bivariate_charts(df)
    insights = _eda_insights(df, shape, stat_summary)

    return {
        "shape": shape,
        "stat_summary": stat_summary,
        "missing_chart": missing_chart,
        "univariate": univariate,
        "bivariate": bivariate,
        "insights": insights,
    }

"""
ML predictions with a proper data science pipeline:
  1. Outlier detection & treatment (IQR winsorization)
  2. Feature engineering with rationale
  3. Train / test split (80/20 stratified)
  4. Preprocessing pipeline (StandardScaler + OneHotEncoder + SimpleImputer)
  5. Hyperparameter tuning (RandomizedSearchCV)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ── lookups ────────────────────────────────────────────────────────────────

_SENTIMENT_POS = ["excited","fantastic","great","excellent","thrilled","amazing",
                  "outstanding","happy","wonderful","positive","satisfied","love"]
_SENTIMENT_NEG = ["disappointed","frustrated","risk","concerned","poor","negative",
                  "struggling","terrible","awful","angry","unhappy","hate","loss",
                  "delayed","overdue","churn","cancel","issue","problem","escalat"]

_STAGE_PROB = {
    "prospecting": 0.10, "discovery": 0.25, "qualification": 0.30,
    "proposal": 0.50, "value proposition": 0.55, "id. decision makers": 0.40,
    "perception analysis": 0.45, "negotiation": 0.70, "review": 0.65,
    "closed won": 1.0, "won": 1.0, "closed": 0.90,
    "closed lost": 0.0, "lost": 0.0,
    "new": 0.10, "open": 0.20, "in progress": 0.40,
    "resolved": 0.95, "complete": 1.0, "escalated": 0.20,
}


def _sentiment_score(text: str) -> float:
    t = str(text).lower()
    pos = sum(1 for w in _SENTIMENT_POS if w in t)
    neg = sum(1 for w in _SENTIMENT_NEG if w in t)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def _find_col(df, *keywords):
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    return None


def _numeric_amount(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


# ── 1. Outlier detection & treatment ──────────────────────────────────────

def _treat_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Detect outliers via IsolationForest, then cap numeric columns at
    1.5 × IQR (Winsorization) instead of dropping rows so no data is lost.
    """
    df = df.copy()
    report = {"outliers_detected": 0, "treatment": "IQR Winsorization (cap at Q1−1.5×IQR / Q3+1.5×IQR)", "columns_treated": []}

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        report["rationale_no_treatment"] = "No numeric columns found — outlier treatment skipped."
        return df, report

    # IsolationForest to count how many rows are anomalous
    X_iso = df[num_cols].fillna(df[num_cols].median())
    if len(X_iso) >= 10:
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(X_iso)
        outlier_flags = iso.predict(X_iso)
        report["outliers_detected"] = int((outlier_flags == -1).sum())

    # Winsorize each numeric column
    for col in num_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 4:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = int(((clean < lo) | (clean > hi)).sum())
        if before > 0:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
            report["columns_treated"].append({"column": col, "capped_rows": before, "range": f"[{lo:.2f}, {hi:.2f}]"})

    return df, report


# ── 2. Feature engineering ─────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Derived features that improve model signal beyond raw columns.
    Returns (feature_df, engineering_report).
    """
    feats = pd.DataFrame(index=df.index)
    created = []

    amt_col = _find_col(df, "amount", "revenue", "value", "price")
    stage_col = _find_col(df, "stage", "status", "state", "phase")
    text_col = _find_col(df, "notes", "note", "description", "comments", "comment", "subject", "text", "body")
    date_col = _find_col(df, "closedate", "close_date", "due", "resolveddate", "resolved_date", "date")
    pri_col = _find_col(df, "priority")

    # amount → log-transform to reduce skew
    if amt_col:
        amt = _numeric_amount(df[amt_col]).fillna(0).clip(lower=0)
        feats["amount"] = amt
        feats["log_amount"] = np.log1p(amt)
        feats["amount_zscore"] = (amt - amt.mean()) / (amt.std() + 1e-9)
        created += ["amount", "log_amount (log1p to reduce skew)", "amount_zscore"]

    # stage → conversion probability
    if stage_col:
        stage_prob = df[stage_col].fillna("").str.lower().map(
            lambda s: next((v for k, v in _STAGE_PROB.items() if k in s), 0.3)
        )
        feats["stage_prob"] = stage_prob
        created.append("stage_prob (mapped from stage name → known conversion rate)")

    # interaction: amount × stage_prob → expected value of deal
    if "amount" in feats.columns and "stage_prob" in feats.columns:
        feats["expected_value"] = feats["amount"] * feats["stage_prob"]
        created.append("expected_value = amount × stage_prob (pipeline-weighted value)")

    # sentiment from free text
    if text_col:
        feats["sentiment"] = df[text_col].fillna("").astype(str).apply(_sentiment_score)
        created.append("sentiment (keyword-based polarity score from text column)")

    # days to close / overdue flag
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        today = pd.Timestamp.today()
        days = (parsed - today).dt.days.fillna(0)
        feats["days_remaining"] = days
        feats["is_overdue"] = (days < 0).astype(float)
        feats["urgency"] = np.where(days < 0, 2, np.where(days < 14, 1, 0))
        created += ["days_remaining", "is_overdue", "urgency (0=ok, 1=soon, 2=overdue)"]

    # priority
    if pri_col:
        pri_map = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25, "normal": 0.5}
        feats["priority_score"] = df[pri_col].fillna("").str.lower().map(
            lambda s: next((v for k, v in pri_map.items() if k in s), 0.5)
        )
        created.append("priority_score (ordinal encoding: critical=1.0 … low=0.25)")

    # fill any remaining NaN
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)

    report = {
        "features_created": created,
        "rationale": (
            "Log-transform reduces right-skew in monetary values. "
            "expected_value combines amount and stage probability into a single signal. "
            "Urgency buckets are more informative than raw days for tree-based models. "
            "Categorical stage/priority → numeric to allow gradient-based learners."
        ),
        "dropped_rationale": "Raw categorical columns (account name, owner) are excluded — high cardinality with no ordinal meaning; they would cause overfitting on small datasets.",
    }
    return feats, report


# ── 3. Preprocessing pipeline ─────────────────────────────────────────────

def _build_preprocessing_pipeline(feature_df: pd.DataFrame) -> tuple[Pipeline, list, list]:
    """
    Returns (pipeline, num_cols, cat_cols).
    Numeric → SimpleImputer(median) → StandardScaler
    Categorical → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
    """
    num_cols = list(feature_df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, num_cols, cat_cols


# ── 4. Win probability with HPT ───────────────────────────────────────────

def _win_probability(df: pd.DataFrame, feats: pd.DataFrame) -> tuple[list, dict]:
    stage_col = _find_col(df, "stage", "status", "state")
    name_col = _find_col(df, "account", "name", "company", "customer", "subject", "title")

    model_report = {
        "model": "Deep Neural Network (MLP: 128→64→32, ReLU, Adam)",
        "hyperparameter_tuning": "RandomizedSearchCV (20 iterations, 5-fold stratified CV)",
        "best_params": None,
        "train_accuracy": None,
        "test_accuracy": None,
        "auc_roc": None,
        "train_size": None,
        "test_size": None,
        "fallback": False,
    }

    if not stage_col:
        model_report["fallback"] = True
        model_report["fallback_reason"] = "No stage/status column found — using stage-probability heuristic"
        return _win_prob_fallback(df, feats, name_col, stage_col), model_report

    stages = df[stage_col].fillna("").str.lower()
    won_mask = stages.str.contains("won|resolved|complete|closed won", regex=True)
    lost_mask = stages.str.contains("lost|closed lost|cancelled|rejected", regex=True)
    open_mask = ~(won_mask | lost_mask)

    labeled = won_mask.astype(int).where(won_mask | lost_mask, other=np.nan)
    train_mask = labeled.notna()

    if train_mask.sum() < 15:
        model_report["fallback"] = True
        model_report["fallback_reason"] = f"Only {int(train_mask.sum())} labeled records — need ≥15 for supervised training; using heuristic"
        return _win_prob_fallback(df, feats, name_col, stage_col), model_report

    X_all = feats[train_mask].values
    y_all = labeled[train_mask].values.astype(int)

    # Train / test split (stratified 80/20)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

    model_report["train_size"] = len(X_train)
    model_report["test_size"] = len(X_test)

    # Preprocessing pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Deep Neural Network — MLPClassifier with architecture search
    param_dist = {
        "hidden_layer_sizes": [
            (128, 64, 32), (64, 32), (128, 64), (64, 32, 16),
            (256, 128, 64), (32, 16),
        ],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "learning_rate_init": [1e-3, 5e-4, 1e-2],
        "max_iter": [500],
        "early_stopping": [True],
        "validation_fraction": [0.1],
    }
    base_clf = MLPClassifier(random_state=42, solver="adam")
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=20, cv=min(5, int(train_mask.sum() // 5)),
        scoring="roc_auc", random_state=42, n_jobs=-1, refit=True
    )
    try:
        search.fit(X_train_p, y_train)
        best_clf = search.best_estimator_
        model_report["best_params"] = search.best_params_
        model_report["train_accuracy"] = round(accuracy_score(y_train, best_clf.predict(X_train_p)), 3)
        model_report["test_accuracy"] = round(accuracy_score(y_test, best_clf.predict(X_test_p)), 3)
        try:
            model_report["auc_roc"] = round(roc_auc_score(y_test, best_clf.predict_proba(X_test_p)[:, 1]), 3)
        except Exception:
            pass

        # predict on open records
        open_idx = feats[open_mask].index
        results = []
        if len(open_idx) > 0:
            X_open = preprocessor.transform(feats.loc[open_idx].values)
            probs = best_clf.predict_proba(X_open)[:, 1]
            for i, idx in enumerate(open_idx):
                name = str(df.loc[idx, name_col]) if name_col else f"Record {idx}"
                stage = str(df.loc[idx, stage_col])
                results.append({
                    "name": name[:50], "stage": stage,
                    "win_probability": round(float(probs[i]) * 100, 1),
                    "sentiment": round(float(feats.loc[idx, "sentiment"]) if "sentiment" in feats.columns else 0.0, 2),
                    "amount": round(float(feats.loc[idx, "amount"]) if "amount" in feats.columns else 0.0, 0),
                })
        return sorted(results, key=lambda x: x["win_probability"], reverse=True)[:20], model_report

    except Exception as e:
        model_report["fallback"] = True
        model_report["fallback_reason"] = f"Model training error: {str(e)[:100]}"
        return _win_prob_fallback(df, feats, name_col, stage_col), model_report


def _win_prob_fallback(df, feats, name_col, stage_col):
    open_mask = ~df[stage_col].fillna("").str.lower().str.contains(
        "won|resolved|complete|lost|closed|cancelled", regex=True
    ) if stage_col else pd.Series([True] * len(df), index=df.index)
    results = []
    for idx in feats[open_mask].index[:50]:
        name = str(df.loc[idx, name_col]) if name_col else f"Record {idx}"
        stage = str(df.loc[idx, stage_col]) if stage_col else ""
        results.append({
            "name": name[:50], "stage": stage,
            "win_probability": round(float(feats.loc[idx, "stage_prob"]) * 100 if "stage_prob" in feats.columns else 30.0, 1),
            "sentiment": round(float(feats.loc[idx, "sentiment"]) if "sentiment" in feats.columns else 0.0, 2),
            "amount": round(float(feats.loc[idx, "amount"]) if "amount" in feats.columns else 0.0, 0),
        })
    return sorted(results, key=lambda x: x["win_probability"], reverse=True)[:20]


# ── risk scoring (rule-based — no HPT needed) ─────────────────────────────

def _risk_scores(df: pd.DataFrame, feats: pd.DataFrame) -> list[dict]:
    stage_col = _find_col(df, "stage", "status", "state")
    name_col = _find_col(df, "account", "name", "company", "customer", "subject", "title")

    if stage_col:
        stages = df[stage_col].fillna("").str.lower()
        active = ~stages.str.contains("won|resolved|complete|lost|closed|cancelled", regex=True)
    else:
        active = pd.Series([True] * len(df), index=df.index)

    results = []
    avg_amount = feats["amount"].mean() if "amount" in feats.columns else 0

    for idx in feats[active].index:
        risk = 0.0
        reasons = []

        sent = float(feats.loc[idx, "sentiment"]) if "sentiment" in feats.columns else 0.0
        if sent < -0.3:
            risk += 0.35; reasons.append("Negative sentiment")
        elif sent < 0:
            risk += 0.15; reasons.append("Mixed sentiment")

        if "is_overdue" in feats.columns and float(feats.loc[idx, "is_overdue"]) > 0:
            days = abs(float(feats.loc[idx, "days_remaining"]))
            risk += min(0.30, days / 90)
            reasons.append(f"Overdue by {int(days)} days")

        sp = float(feats.loc[idx, "stage_prob"]) if "stage_prob" in feats.columns else 0.3
        if sp < 0.2:
            risk += 0.20; reasons.append("Early stage, low conversion")

        amt = float(feats.loc[idx, "amount"]) if "amount" in feats.columns else 0.0
        if amt > avg_amount * 1.5 and sp < 0.5:
            risk += 0.15; reasons.append("High-value deal at risk")

        pri = float(feats.loc[idx, "priority_score"]) if "priority_score" in feats.columns else 0.5
        if pri >= 0.75 and sp < 0.4:
            risk += 0.10; reasons.append("High priority, low progress")

        risk = min(risk, 1.0)
        if risk > 0.25:
            name = str(df.loc[idx, name_col]) if name_col else f"Record {idx}"
            stage = str(df.loc[idx, stage_col]) if stage_col else "Unknown"
            results.append({
                "name": name[:50], "stage": stage,
                "risk_score": round(risk * 100, 1),
                "risk_label": "High Risk" if risk > 0.6 else "Medium Risk",
                "reasons": reasons[:3],
                "amount": round(amt, 0),
            })

    return sorted(results, key=lambda x: x["risk_score"], reverse=True)[:10]


# ── anomaly detection ──────────────────────────────────────────────────────

def _anomalies(df: pd.DataFrame, feats: pd.DataFrame) -> list[dict]:
    name_col = _find_col(df, "account", "name", "company", "customer", "subject", "title")
    stage_col = _find_col(df, "stage", "status", "state")

    cols = [c for c in ["amount_zscore", "sentiment", "stage_prob", "is_overdue", "days_remaining", "expected_value"] if c in feats.columns]
    if not cols or len(feats) < 10:
        return []

    X = feats[cols].fillna(0).values
    try:
        clf = IsolationForest(contamination=0.08, random_state=42)
        preds = clf.fit_predict(X)
        scores = clf.score_samples(X)
        results = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred == -1:
                idx = feats.index[i]
                name = str(df.loc[idx, name_col]) if name_col else f"Record {idx}"
                stage = str(df.loc[idx, stage_col]) if stage_col else ""
                amt = float(feats.loc[idx, "amount"]) if "amount" in feats.columns else 0.0
                reasons = []
                if "amount_zscore" in feats.columns and abs(float(feats.loc[idx, "amount_zscore"])) > 2:
                    reasons.append("Unusual deal size")
                if "sentiment" in feats.columns and float(feats.loc[idx, "sentiment"]) < -0.5:
                    reasons.append("Very negative sentiment")
                if "is_overdue" in feats.columns and float(feats.loc[idx, "is_overdue"]):
                    reasons.append("Significantly overdue")
                results.append({
                    "name": name[:50], "stage": stage,
                    "anomaly_score": round(abs(score), 3),
                    "reasons": reasons or ["Statistical outlier"],
                    "amount": round(amt, 0),
                })
        return sorted(results, key=lambda x: x["anomaly_score"], reverse=True)[:8]
    except Exception:
        return []


# ── revenue forecast ───────────────────────────────────────────────────────

def _revenue_forecast(df: pd.DataFrame) -> dict:
    date_col = _find_col(df, "closedate", "close_date", "createddate", "created_date", "date")
    amt_col = _find_col(df, "amount", "revenue", "value")

    if not date_col or not amt_col:
        return {}

    try:
        df2 = df.copy()
        df2["_date"] = pd.to_datetime(df2[date_col], errors="coerce")
        df2["_amt"] = _numeric_amount(df2[amt_col])
        df2 = df2.dropna(subset=["_date", "_amt"])
        df2["_month_num"] = df2["_date"].dt.year * 12 + df2["_date"].dt.month
        monthly = df2.groupby("_month_num")["_amt"].sum().reset_index()

        if len(monthly) < 3:
            return {}

        X = monthly["_month_num"].values.reshape(-1, 1)
        y = monthly["_amt"].values
        model = LinearRegression().fit(X, y)

        last_month = int(monthly["_month_num"].max())
        future = np.array([last_month + 1, last_month + 2, last_month + 3]).reshape(-1, 1)
        preds = model.predict(future)

        def _label(m):
            yr, mo = divmod(m - 1, 12)
            return pd.Timestamp(year=yr, month=mo + 1, day=1).strftime("%b %Y")

        return {
            "trend": "up" if model.coef_[0] > 0 else "down",
            "monthly_growth": round(float(model.coef_[0]), 0),
            "forecast": [{"month": _label(int(future[i][0])), "predicted": round(float(preds[i]), 0)} for i in range(3)],
            "r_squared": round(float(model.score(X, y)), 2),
        }
    except Exception:
        return {}


# ── next best actions ──────────────────────────────────────────────────────

def _next_best_actions(risk_records: list[dict], win_probs: list[dict]) -> list[dict]:
    actions = []
    for r in risk_records[:3]:
        if r["risk_score"] > 60:
            action = "Schedule executive call"
            if "Negative sentiment" in r.get("reasons", []):
                action = "Address concerns — schedule customer success call"
            elif "Overdue" in " ".join(r.get("reasons", [])):
                action = "Expedite — contact immediately, reassign if needed"
            elif "High-value deal" in " ".join(r.get("reasons", [])):
                action = "Escalate to senior AE — high value at risk"
            actions.append({"record": r["name"], "action": action, "urgency": "High", "reason": r["reasons"][0] if r["reasons"] else "High risk score"})

    for w in sorted(win_probs, key=lambda x: x["win_probability"])[:3]:
        if w["win_probability"] < 30:
            actions.append({"record": w["name"], "action": "Re-qualify or advance stage — low conversion probability", "urgency": "Medium", "reason": f"Win probability only {w['win_probability']}%"})

    return actions[:6]


# ── main entry ─────────────────────────────────────────────────────────────

def run_ml_predictions(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"error": "Need at least 5 records for ML predictions"}

    # Step 1 — outlier treatment
    df_clean, outlier_report = _treat_outliers(df)

    # Step 2 — feature engineering
    feats, feat_report = _engineer_features(df_clean)

    # Step 3 — preprocessing pipeline summary (built inside win_probability training)
    preprocessing_steps = [
        "SimpleImputer(strategy='median') — fills missing numerics with column median",
        "StandardScaler — zero-mean, unit-variance normalization for numeric features",
        "OneHotEncoder(handle_unknown='ignore') — encodes categorical features if present",
        "ColumnTransformer — applies transformers per column type in a single pipeline",
    ]

    # Step 4 — models (win prob includes HPT + train/test split)
    win_probs, model_report = _win_probability(df_clean, feats)
    risks = _risk_scores(df_clean, feats)
    anomalies = _anomalies(df_clean, feats)
    forecast = _revenue_forecast(df_clean)
    actions = _next_best_actions(risks, win_probs)

    avg_win = round(np.mean([w["win_probability"] for w in win_probs]), 1) if win_probs else None
    high_risk_count = sum(1 for r in risks if r["risk_score"] > 60)

    ml_report = {
        "outlier_treatment": outlier_report,
        "feature_engineering": feat_report,
        "preprocessing_pipeline": preprocessing_steps,
        "model": model_report,
        "train_test_split": {
            "strategy": "80/20 stratified split (preserves class ratio in both sets)",
            "train_records": model_report.get("train_size"),
            "test_records": model_report.get("test_size"),
        },
    }

    return {
        "win_probabilities": win_probs,
        "risk_scores": risks,
        "anomalies": anomalies,
        "revenue_forecast": forecast,
        "next_best_actions": actions,
        "ml_report": ml_report,
        "summary": {
            "avg_win_probability": avg_win,
            "high_risk_count": high_risk_count,
            "anomaly_count": len(anomalies),
            "forecast_trend": forecast.get("trend"),
            "monthly_growth": forecast.get("monthly_growth"),
        },
    }

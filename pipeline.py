"""Steps 1-3 of the project: load data -> feature engineering -> split + preprocessing (all NumPy/Pandas)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "raw" / "IT_customer_churn.csv"
TARGET = "Churn"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED = ["NumServices", "AvgMonthlySpend", "ChargePerService"]


# ------------------------------------------------------------------ 1. load
def load_data(path=DATA_PATH):
    """Return (X, y). TotalCharges has blank strings in the raw file -> NaN. y: Yes=1, No=0."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df[TARGET].str.strip() == "Yes").astype(int).to_numpy()
    return df.drop(columns=[TARGET]).reset_index(drop=True), y


# ------------------------------------------------------------------ 2. feature engineering
def add_features(X):
    """Create 3 new numeric columns from existing ones (works for any number of rows)."""
    X = X.copy()
    total = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X["NumServices"] = (X[SERVICES] == "Yes").sum(axis=1).astype(float)
    # average bill actually paid per month; a brand-new customer (tenure 0) has no history -> use current bill
    X["AvgMonthlySpend"] = np.where(X["tenure"] > 0, total / X["tenure"].clip(lower=1), X["MonthlyCharges"])
    X["ChargePerService"] = X["MonthlyCharges"] / (X["NumServices"] + 1)
    return X


def stratified_split(y, fractions=(0.70, 0.15, 0.15), seed=42):
    """Shuffle each class separately and cut it, so every part keeps the same churn share."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fractions]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cuts = (np.cumsum(fractions)[:-1] * len(idx)).round().astype(int)
        for part, chunk in zip(parts, np.split(idx, cuts)):
            part.extend(chunk.tolist())
    return [np.sort(np.array(p)) for p in parts]


# ------------------------------------------------------------------ 3. preprocessing
class Preprocessor:
    """median-impute + standardise numbers, mode-impute + one-hot encode categories.
    Everything is learned in fit() on TRAINING rows only, then re-used unchanged."""

    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = list(numeric), list(categorical)

    def fit(self, df):
        num = df[self.numeric].apply(pd.to_numeric, errors="coerce")
        self.median = num.median().to_dict()
        filled = num.fillna(self.median)
        self.mean = filled.mean().to_dict()
        self.std = {c: (s if s > 1e-12 else 1.0) for c, s in filled.std(ddof=0).items()}
        self.categories = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.categorical}
        self.mode = {c: df[c].mode()[0] for c in self.categorical}
        self.min = filled.min().to_dict()
        self.max = filled.max().to_dict()
        self.feature_names = self.numeric + [f"{c}={v}" for c in self.categorical for v in self.categories[c]]
        return self

    def transform(self, df):
        num = df[self.numeric].apply(pd.to_numeric, errors="coerce").fillna(self.median)
        blocks = [((num[c] - self.mean[c]) / self.std[c]).to_numpy()[:, None] for c in self.numeric]
        for c in self.categorical:
            col = df[c].astype(str).where(df[c].notna(), self.mode[c]).to_numpy()
            blocks.append((col[:, None] == np.array(self.categories[c])[None, :]).astype(float))
        return np.hstack(blocks)

    def groups(self):
        """original feature -> its column indices after encoding (to group one-hot columns)."""
        g = {}
        for i, name in enumerate(self.feature_names):
            g.setdefault(name.split("=")[0], []).append(i)
        return g


def explain(predict_proba, prep, record, top=6):
    """Per-customer 'what pushes the risk': reset ONE feature at a time to the typical value
    (training median / mode) and measure how much the probability changes."""
    base = float(predict_proba(prep.transform(record))[0])
    rows = []
    for c in prep.numeric + prep.categorical:
        typical = record.copy()
        typical[c] = prep.median[c] if c in prep.numeric else prep.mode[c]
        delta = base - float(predict_proba(prep.transform(typical))[0])
        rows.append({"feature": c, "value": record[c].iloc[0], "effect on risk": delta})
    rows.sort(key=lambda r: -abs(r["effect on risk"]))
    return rows[:top]

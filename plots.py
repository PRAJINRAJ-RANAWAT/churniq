"""All figures (matplotlib / seaborn). Each function returns a figure that Streamlit displays."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from metrics import auc, roc_curve
from pipeline import CATEGORICAL

CHURN, STAY, GREY, ORANGE = "#E4572E", "#2E86AB", "#6C757D", "#F2A541"
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})


# ------------------------------------------------------------------ EDA
def class_distribution(y):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    ax.bar(["Stayed (0)", "Churned (1)"], counts, color=[STAY, CHURN])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n({c / len(y):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, max(counts) * 1.2); ax.set_ylabel("Customers"); ax.set_title("Class balance")
    return fig


def numeric_distributions(X, y):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        for lab, color, name in [(0, STAY, "Stayed"), (1, CHURN, "Churned")]:
            ax.hist(X.loc[y == lab, c].dropna(), bins=30, alpha=.65, color=color, label=name, density=True)
        ax.set_title(c); ax.set_ylabel("Density")
    axes[0].legend()
    fig.tight_layout()
    return fig


def churn_rate_by_category(X, y):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharey=True)
    for ax, c in zip(axes.ravel(), CATEGORICAL):
        rate = pd.Series(y).groupby(X[c].values).mean().sort_values(ascending=False)
        ax.bar(range(len(rate)), rate.values, color=CHURN)
        ax.axhline(y.mean(), color=GREY, ls="--", lw=1)
        ax.set_xticks(range(len(rate))); ax.set_xticklabels([str(i)[:12] for i in rate.index], rotation=30, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10)
    for ax in axes.ravel()[len(CATEGORICAL):]:
        ax.axis("off")
    fig.suptitle("Churn rate per category (dashed = overall rate)", fontweight="bold")
    fig.tight_layout()
    return fig


def correlation_heatmap(X, y, columns):
    d = X[columns].copy(); d["Churn"] = y
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation")
    return fig


def churn_rate_binned(X, y, col, bins=12):
    edges = np.unique(np.quantile(X[col].dropna(), np.linspace(0, 1, bins + 1)))
    g = pd.DataFrame({"bin": pd.cut(X[col], edges, include_lowest=True), "y": y}).groupby("bin", observed=True)["y"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([iv.mid for iv in g.index], g.values, "o-", color=CHURN)
    ax.axhline(y.mean(), color=GREY, ls="--", label="overall rate")
    ax.set_xlabel(col); ax.set_ylabel("Churn rate"); ax.set_title(f"Churn rate vs {col}"); ax.legend()
    return fig


def churn_by_contract(X, y):
    rate = pd.Series(y).groupby(X["Contract"].values).agg(["mean", "size"]).loc[["Month-to-month", "One year", "Two year"]]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.bar(rate.index, rate["mean"], color=[CHURN, ORANGE, STAY])
    for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
        ax.text(i, m, f"{m:.1%}\n(n={n})", ha="center", va="bottom")
    ax.set_ylim(0, rate["mean"].max() * 1.3); ax.set_ylabel("Churn rate"); ax.set_title("Churn by contract")
    return fig


# ------------------------------------------------------------------ models
def loss_curve(model):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(model.history, color=STAY, label="train loss")
    if model.val_history:
        ax.plot(model.val_history, color=CHURN, label="validation loss")
    ax.set_xlabel("Gradient-descent iteration"); ax.set_ylabel("Binary cross-entropy"); ax.set_title("Logistic regression learning curve"); ax.legend()
    return fig


def confusion_matrix_plot(cm, title):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14},
                xticklabels=["Pred: stay", "Pred: churn"], yticklabels=["Actual: stay", "Actual: churn"])
    ax.set_title(title)
    return fig


def roc_plot(y, probas):
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for name, p in probas.items():
        fpr, tpr = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)"); ax.set_title("ROC curve"); ax.legend(loc="lower right", fontsize=8)
    return fig


def threshold_plot(rows, chosen):
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for key, c in [("precision", STAY), ("recall", CHURN), ("f1", ORANGE)]:
        ax.plot(t, [r[key] for r in rows], color=c, label=key)
    ax.axvline(chosen, color=GREY, ls="--", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_title("Precision / recall / F1 vs threshold"); ax.legend()
    return fig


def importance_plot(names, values, title, top=12):
    order = np.argsort(-np.abs(values))[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh([names[i] for i in order], [values[i] for i in order], color=[CHURN if values[i] > 0 else STAY for i in order])
    ax.axvline(0, color="k", lw=.8); ax.set_title(title)
    return fig


def comparison_plot(rows):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        ax.bar(np.arange(len(metrics)) + i * w, [r[m] for m in metrics], w, label=r["model"])
    ax.set_xticks(np.arange(len(metrics)) + 0.4 - w / 2); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_title("Model comparison (test set)"); ax.legend(fontsize=8, ncol=2)
    return fig


# ------------------------------------------------------------------ segments
def elbow_plot(ks, inertia, sil):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ks, inertia, "o-", color=STAY); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow curve")
    axes[1].plot(ks, sil, "o-", color=ORANGE); axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette"); axes[1].set_title("Silhouette vs k")
    return fig


def pca_scatter(coords, labels, ratio, colors=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    pal = colors or sns.color_palette("Set2", int(labels.max()) + 1)
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=.5, color=pal[k], label=f"Cluster {k}")
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.1%})"); ax.set_title("Segments in 2-D (PCA)"); ax.legend(markerscale=2)
    return fig


def cluster_profile_plot(profile):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (col, t) in zip(axes, [("tenure", "Mean tenure"), ("MonthlyCharges", "Mean monthly charge"),
                                   ("TotalCharges", "Mean total charge"), ("churn_rate", "Churn rate (after clustering)")]):
        ax.bar([f"C{k}" for k in profile.index], profile[col], color=CHURN if col == "churn_rate" else STAY); ax.set_title(t, fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ restored extra plots
def missing_summary(X):
    miss = X.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if len(miss):
        ax.barh(miss.index, miss.values, color=ORANGE)
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v}", va="center")
    else:
        ax.text(.5, .5, "No missing values", ha="center"); ax.axis("off")
    ax.set_title("Missing values (blank TotalCharges, tenure = 0)")
    return fig


def categorical_distributions(X):
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), CATEGORICAL):
        vc = X[c].value_counts()
        ax.bar(range(len(vc)), vc.values, color=STAY)
        ax.set_xticks(range(len(vc))); ax.set_xticklabels([str(i)[:12] for i in vc.index], rotation=30, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10)
    for ax in axes.ravel()[len(CATEGORICAL):]:
        ax.axis("off")
    fig.suptitle("Categorical feature counts", fontweight="bold")
    fig.tight_layout()
    return fig


def sweep_curves(xs, tr_auc, va_auc, tr_err, va_err, xlabel, leaves=None):
    n = 3 if leaves else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.6))
    x = range(len(xs))
    axes[0].plot(x, tr_auc, "o-", color=STAY, label="train"); axes[0].plot(x, va_auc, "o-", color=CHURN, label="validation")
    axes[0].set_title("ROC-AUC"); axes[0].legend()
    axes[1].plot(x, tr_err, "o-", color=STAY, label="train"); axes[1].plot(x, va_err, "o-", color=CHURN, label="validation")
    axes[1].set_title("Error rate (threshold 0.5)"); axes[1].legend()
    if leaves:
        axes[2].plot(x, leaves, "o-", color=ORANGE); axes[2].set_yscale("log"); axes[2].set_title("Number of leaves")
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels([str(v) for v in xs], fontsize=8); ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def generalization_plot(rows):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - .2, [r["train_auc"] for r in rows], .4, label="Train AUC", color=STAY)
    ax.bar(x + .2, [r["val_auc"] for r in rows], .4, label="Validation AUC", color=CHURN)
    for i, r in enumerate(rows):
        ax.text(i, max(r["train_auc"], r["val_auc"]) + .01, f"gap {r['train_auc'] - r['val_auc']:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in rows], fontsize=8); ax.set_ylim(.5, 1.05)
    ax.set_title("Train vs validation AUC (generalisation gap)"); ax.legend(loc="lower right")
    return fig


def risk_band_plot(y, p):
    bands = np.where(p < .25, "Low", np.where(p < .5, "Medium", "High"))
    names = ["Low", "Medium", "High"]
    rate = [y[bands == b].mean() if (bands == b).any() else 0 for b in names]
    n = [int((bands == b).sum()) for b in names]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.bar(names, rate, color=[STAY, ORANGE, CHURN])
    for i in range(3):
        ax.text(i, rate[i], f"{rate[i]:.1%}\n(n={n[i]})", ha="center", va="bottom")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Observed churn rate"); ax.set_title("Observed churn by predicted risk band (test)")
    return fig

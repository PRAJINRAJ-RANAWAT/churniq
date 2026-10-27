"""ChurnIQ - simple end-to-end churn project. Run:  streamlit run app.py
Tabs follow the ML pipeline: data/EDA -> feature engineering -> model -> predict -> segments."""
import numpy as np
import pandas as pd
import streamlit as st

import metrics as M
import plots
from models import MODEL_NAMES, KMeans, PCA, build_model, silhouette_score
from pipeline import (CATEGORICAL, ENGINEERED, NUMERIC, Preprocessor, add_features, explain, load_data,
                      stratified_split)

plots.plt.close("all")   # free figures from the previous rerun
st.set_page_config(page_title="ChurnIQ", page_icon="📡", layout="wide")
st.title("📡 ChurnIQ - Telecom Churn Prediction")
st.caption("Every model and metric is written from scratch with NumPy. Change parameters in the sidebar - the app retrains instantly.")


# ============================================================ cached pipeline steps
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def prepare(use_fe, seed):
    """Split first, then fit the preprocessor on TRAIN only (no leakage)."""
    X, y = get_data()
    if use_fe:
        X = add_features(X)
    numeric = NUMERIC + (ENGINEERED if use_fe else [])
    tr, va, te = stratified_split(y, seed=seed)
    prep = Preprocessor(numeric, CATEGORICAL).fit(X.iloc[tr])
    d = {"X": X, "y": y, "prep": prep, "idx": (tr, va, te)}
    for name, i in zip(("tr", "va", "te"), (tr, va, te)):
        d["X_" + name], d["y_" + name] = prep.transform(X.iloc[i]), y[i]
    return d


@st.cache_resource
def train(name, params, use_fe, seed):
    d = prepare(use_fe, seed)
    model = build_model(name, dict(params))
    if name == "logistic_regression":
        model.fit(d["X_tr"], d["y_tr"], d["X_va"], d["y_va"])
    else:
        model.fit(d["X_tr"], d["y_tr"])
    return model, model.predict_proba(d["X_va"]), model.predict_proba(d["X_te"])


@st.cache_data
def compare_all(use_fe, seed):
    defaults = {"logistic_regression": {"n_iter": 1500}, "knn": {"k": 31}, "decision_tree": {"max_depth": 4}, "gaussian_nb": {}}
    d, rows, probas = prepare(use_fe, seed), [], {}
    for name, params in defaults.items():
        _, pva, pte = train(name, tuple(params.items()), use_fe, seed)
        t = M.best_f1_threshold(d["y_va"], pva)
        r = M.report(d["y_te"], pte >= t)
        mdl = train(name, tuple(params.items()), use_fe, seed)[0]
        rows.append({"model": MODEL_NAMES[name], "threshold": t, **{k: r[k] for k in ("accuracy", "precision", "recall", "f1")},
                     "auc": M.roc_auc_score(d["y_te"], pte), "train_auc": M.roc_auc_score(d["y_tr"], mdl.predict_proba(d["X_tr"])),
                     "val_auc": M.roc_auc_score(d["y_va"], pva), "cm": M.confusion_matrix(d["y_te"], pte >= t)})
        probas[MODEL_NAMES[name]] = pte
    return pd.DataFrame(rows), probas


@st.cache_data
def sweep(name, params, use_fe, seed):
    """Train/validation AUC while one hyper-parameter changes (over- vs under-fitting)."""
    d = prepare(use_fe, seed)
    grids = {"logistic_regression": ("l2", [0, 10, 100, 500, 2000, 8000]), "knn": ("k", [1, 5, 11, 21, 31, 51, 101, 201]),
             "decision_tree": ("max_depth", [1, 2, 3, 4, 6, 8, 12, 20]), "gaussian_nb": ("var_smoothing", [1e-9, 1e-6, 1e-3, 1e-1, 1.0])}
    key, values = grids[name]
    tr_auc, va_auc, tr_err, va_err, leaves = [], [], [], [], []
    for v in values:
        p = {**dict(params), key: v}
        m = build_model(name, p).fit(d["X_tr"], d["y_tr"])
        ptr, pva = m.predict_proba(d["X_tr"]), m.predict_proba(d["X_va"])
        tr_auc.append(M.roc_auc_score(d["y_tr"], ptr)); va_auc.append(M.roc_auc_score(d["y_va"], pva))
        tr_err.append(float(((ptr >= .5) != d["y_tr"]).mean())); va_err.append(float(((pva >= .5) != d["y_va"]).mean()))
        if name == "decision_tree":
            leaves.append(sum(f < 0 for f in m.feature))
    return key, values, tr_auc, va_auc, tr_err, va_err, leaves


@st.cache_data
def segments(use_fe, seed, k):
    d = prepare(use_fe, seed)
    X = d["X_tr"]
    km = KMeans(k, seed=seed).fit(X)
    sample = np.random.default_rng(seed).choice(len(X), 1000, replace=False)
    sil = silhouette_score(X[sample], km.labels_[sample])
    pca = PCA(2).fit(X)
    return km, sil, pca.transform(X), pca.explained_variance_ratio_


@st.cache_data
def elbow(use_fe, seed):
    X, ks = prepare(use_fe, seed)["X_tr"], list(range(2, 8))
    sample = np.random.default_rng(seed).choice(len(X), 800, replace=False)
    fits = [KMeans(k, n_init=3, seed=seed).fit(X) for k in ks]
    return ks, [f.inertia_ for f in fits], [silhouette_score(X[sample], f.predict(X[sample])) for f in fits]


def permutation_importance(model, X, y, prep, n=600, seed=0):
    """Importance of an original feature = drop in AUC when its columns are shuffled together."""
    rng = np.random.default_rng(seed)
    X, y = X[:n], y[:n]
    base = M.roc_auc_score(y, model.predict_proba(X))
    out = {}
    for feat, cols in prep.groups().items():
        Xp = X.copy()
        Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
        out[feat] = base - M.roc_auc_score(y, model.predict_proba(Xp))
    return out


# ============================================================ sidebar: parameters
X_raw, y_raw = get_data()
st.sidebar.header("1 · Features & split")
use_fe = st.sidebar.checkbox("Add engineered features", True, help=", ".join(ENGINEERED))
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

st.sidebar.header("2 · Model & parameters")
name = st.sidebar.selectbox("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
n_train = int(len(y_raw) * 0.70)
if name == "logistic_regression":
    params = {"learning_rate": st.sidebar.select_slider("Learning rate", [0.01, 0.03, 0.1, 0.3, 0.5], 0.1),
              "n_iter": st.sidebar.slider("Iterations", 100, 4000, 1500, 100),
              "l2": float(st.sidebar.select_slider("L2 strength (λ)", [0, 10, 100, 500, 2000, 8000], 0))}
elif name == "knn":
    params = {"k": st.sidebar.slider("k (neighbours)", 1, min(301, n_train), 31, 2)}
elif name == "decision_tree":
    params = {"max_depth": st.sidebar.slider("Max depth", 1, 20, 4),
              "min_samples_leaf": st.sidebar.slider("Min samples per leaf", 1, 200, 1)}
else:
    params = {"var_smoothing": float(st.sidebar.select_slider("Variance smoothing", [1e-9, 1e-6, 1e-3, 1e-1, 1.0], 1e-6))}

st.sidebar.header("3 · Decision constraint")
mode = st.sidebar.radio("How to pick the churn threshold", ["Best F1 (validation)", "Must catch ≥ X% of churners", "Manual"])
min_recall = st.sidebar.slider("Minimum recall", 0.30, 0.95, 0.80, 0.01) if mode.startswith("Must") else None
manual_t = st.sidebar.slider("Threshold", 0.02, 0.98, 0.50, 0.01) if mode == "Manual" else None

# ---- train with the chosen parameters
d = prepare(use_fe, seed)
with st.spinner("Training..."):
    model, p_val, p_test = train(name, tuple(params.items()), use_fe, seed)
if mode == "Manual":
    thr = manual_t
elif mode.startswith("Must"):
    thr = M.threshold_for_min_recall(d["y_va"], p_val, min_recall)
    if thr is None:
        st.sidebar.warning("No threshold reaches that recall; falling back to best F1.")
        thr = M.best_f1_threshold(d["y_va"], p_val)
else:
    thr = M.best_f1_threshold(d["y_va"], p_val)   # chosen on validation, never on test
st.sidebar.success(f"Threshold used: **{thr:.2f}**")

tab_eda, tab_fe, tab_model, tab_pred, tab_seg = st.tabs(
    ["1 · Data & EDA", "2 · Feature engineering", "3 · Model & results", "4 · Predict a customer", "5 · Segments"])

# ============================================================ tab 1: EDA
with tab_eda:
    c = st.columns(4)
    c[0].metric("Customers", f"{len(y_raw):,}")
    c[1].metric("Features", X_raw.shape[1])
    c[2].metric("Churn rate", f"{y_raw.mean():.1%}")
    c[3].metric("Missing TotalCharges", int(X_raw["TotalCharges"].isna().sum()))
    st.dataframe(X_raw.head(8), hide_index=True)
    a, b = st.columns(2)
    a.pyplot(plots.class_distribution(y_raw))
    a.caption("How many customers stayed vs churned. The classes are imbalanced, so plain accuracy is misleading - we also track precision, recall and F1.")
    b.pyplot(plots.churn_by_contract(X_raw, y_raw))
    b.caption("Churn rate for each contract type. Month-to-month customers leave far more often than 1- or 2-year contracts.")
    st.pyplot(plots.numeric_distributions(X_raw, y_raw))
    st.caption("Distribution of tenure, monthly charges and total charges, split by stayed/churned. Where the two shapes differ, the feature carries signal.")
    a, b, c3 = st.columns(3)
    a.pyplot(plots.churn_rate_binned(X_raw, y_raw, "tenure"))
    a.caption("Churn rate by tenure bucket: risk is highest in the first months and falls as customers stay longer.")
    b.pyplot(plots.churn_rate_binned(X_raw, y_raw, "MonthlyCharges"))
    b.caption("Churn rate by monthly bill bucket: higher bills tend to mean higher churn.")
    c3.pyplot(plots.correlation_heatmap(X_raw, y_raw, NUMERIC))
    c3.caption("Pairwise correlation (-1 to +1). Strong pairs (e.g. tenure and total charges) carry overlapping information.")
    a, b = st.columns([1, 2])
    a.pyplot(plots.missing_summary(X_raw))
    a.caption("Columns with missing values. These are filled (median/mode) during preprocessing.")
    b.markdown("**Categorical feature counts**")
    st.pyplot(plots.categorical_distributions(X_raw))
    st.caption("How many customers fall in each category of every text feature - shows rare categories and class balance.")
    st.pyplot(plots.churn_rate_by_category(X_raw, y_raw))
    st.caption("Churn rate inside each category. Bars far above the overall churn rate mark high-risk groups.")
    st.caption("Churn is concentrated in new customers on month-to-month contracts with high monthly bills.")

# ============================================================ tab 2: features
with tab_fe:
    tr, va, te = d["idx"]
    st.markdown(f"**Split (stratified):** train {len(tr):,} · validation {len(va):,} · test {len(te):,} - each keeps ≈{y_raw.mean():.1%} churn.")
    st.markdown("**Preprocessing, fitted on train only:** median-fill numbers → standardise `(x−mean)/std` → mode-fill categories → one-hot encode.")
    st.markdown(f"**Result:** {len(d['prep'].numeric)} numeric + {len(CATEGORICAL)} categorical columns → **{d['X_tr'].shape[1]}** model inputs.")
    if use_fe:
        st.markdown("**Engineered features:** `NumServices` = count of subscribed services · `AvgMonthlySpend` = TotalCharges ÷ tenure · "
                    "`ChargePerService` = MonthlyCharges ÷ (NumServices+1)")
        st.dataframe(d["X"][["tenure", "MonthlyCharges", "TotalCharges"] + ENGINEERED].head(8), hide_index=True)
        st.pyplot(plots.correlation_heatmap(d["X"], y_raw, ["tenure", "MonthlyCharges", "TotalCharges"] + ENGINEERED))
        st.caption("Correlation of the new features with the originals and with churn. A new feature is useful if it relates to churn "
                   "without being a pure copy of an existing column.")
    st.markdown("**Encoded matrix (first rows, first 10 columns):**")
    st.dataframe(pd.DataFrame(d["X_tr"][:6, :10], columns=d["prep"].feature_names[:10]).round(2), hide_index=True)

# ============================================================ tab 3: model
with tab_model:
    st.subheader(MODEL_NAMES[name])
    yt = d["y_te"]
    r = M.report(yt, p_test >= thr)
    c = st.columns(6)
    for col, (lab, v) in zip(c, [("ROC-AUC", M.roc_auc_score(yt, p_test)), ("Accuracy", r["accuracy"]), ("Precision", r["precision"]),
                                 ("Recall", r["recall"]), ("Specificity", r["specificity"]), ("F1", r["f1"])]):
        col.metric(lab, f"{v:.3f}")
    st.caption(f"Test-set results (threshold {thr:.2f} chosen on validation). Validation AUC = {M.roc_auc_score(d['y_va'], p_val):.3f}.")
    a, b, c3 = st.columns(3)
    a.pyplot(plots.confusion_matrix_plot(M.confusion_matrix(yt, p_test >= thr), "Confusion matrix (test)"))
    a.caption("Counts of right and wrong predictions on the test set: true stays/churners (correct) vs false alarms and missed churners. "
              "Moving the threshold shifts customers between these boxes.")
    b.pyplot(plots.roc_plot(yt, {MODEL_NAMES[name]: p_test}))
    b.caption("ROC curve: catch rate (recall) vs false-alarm rate across every possible threshold. Closer to the top-left is better; "
              "the diagonal is random guessing. AUC summarises it in one number.")
    c3.pyplot(plots.threshold_plot(M.threshold_sweep(d["y_va"], p_val), thr))
    c3.caption("How precision, recall and F1 change as the threshold moves (validation data). The marker is the threshold currently used: "
               "a lower threshold catches more churners but raises false alarms.")

    names = d["prep"].feature_names
    st.markdown("##### What drives the model")
    if name == "logistic_regression":
        a, b = st.columns(2)
        a.pyplot(plots.loss_curve(model))
        a.caption("Training loss per iteration of gradient descent. It should fall and flatten; if it is still dropping, add iterations "
                  "or raise the learning rate; if it zig-zags, lower the learning rate.")
        b.pyplot(plots.importance_plot(names, model.w, "Coefficients (red raises churn risk)"))
        b.caption("Learned weight per input (inputs are standardised, so sizes are comparable). Red pushes churn risk up, the other colour pushes it down.")
    elif name == "decision_tree":
        st.caption(f"Tree depth {max(model.depth_of)}, {sum(f < 0 for f in model.feature)} leaves.")
        st.pyplot(plots.importance_plot(names, model.feature_importances_, "Gini importance"))
        st.caption("Gini importance: how much each feature reduced impurity across the tree's splits. Bigger bar = used for more of the decision. "
                   "It is never negative and does not show direction.")
    else:
        with st.spinner("Permutation importance..."):
            imp = permutation_importance(model, d["X_va"], d["y_va"], d["prep"])
        st.pyplot(plots.importance_plot(list(imp), np.array(list(imp.values())), "AUC drop when feature is shuffled"))
        st.caption("Each feature is randomly shuffled and we measure how much AUC drops. A big drop means the model relies on that feature; "
                   "near zero means it barely matters. This model has no built-in weights, so this is how we explain it.")

    a, b = st.columns(2)
    a.pyplot(plots.risk_band_plot(yt, p_test))
    a.caption("Actual churn rate of test customers grouped by predicted risk band.")
    b.caption("Bands: Low < 0.25, Medium < 0.50, High ≥ 0.50. A useful model shows churn rising from Low to High.")

    st.markdown("##### Over/under-fitting: one hyper-parameter changing (others as in sidebar)")
    with st.spinner("Running sweep..."):
        key, values, tr_auc, va_auc, tr_err, va_err, leaves = sweep(name, tuple(params.items()), use_fe, seed)
    st.pyplot(plots.sweep_curves(values, tr_auc, va_auc, tr_err, va_err, key, leaves or None))
    sweep_note = {
        "logistic_regression": "L2 strength λ: 0 = no penalty (can overfit); very large λ shrinks weights to zero (underfits).",
        "knn": "k: small k memorises noise (train AUC near 1, validation low = overfit); very large k averages too many neighbours (underfits).",
        "decision_tree": "Max depth: deeper trees fit training data almost perfectly but validation drops (overfit); depth 1-2 is too simple (underfit). "
                         "The leaves line shows tree size.",
        "gaussian_nb": "Variance smoothing: adds a floor to each feature's variance. Too small = over-confident on rare values; too large = all features look alike.",
    }[name]
    st.caption(f"Left: AUC on train vs validation. Right: error rate. {sweep_note} "
               "Train ≫ validation = overfitting. Both poor = underfitting. Choose the setting where validation AUC peaks.")

    st.markdown("##### All four models side by side (default parameters, same split)")
    with st.spinner("Training all four models..."):
        table, probas = compare_all(use_fe, seed)
    a, b = st.columns(2)
    a.dataframe(table.drop(columns="cm").round(3), hide_index=True)
    b.pyplot(plots.roc_plot(yt, probas))
    b.caption("ROC curves of all four models on the same test set. The curve nearest the top-left corner ranks churners best.")
    st.pyplot(plots.comparison_plot(table.to_dict("records")))
    st.caption("Accuracy, precision, recall and F1 for each model, each at its own best-F1 threshold. "
               "Pick by what matters: recall to catch churners, precision to avoid wasted offers.")
    cols = st.columns(4)
    for col, r in zip(cols, table.to_dict("records")):
        col.pyplot(plots.confusion_matrix_plot(r["cm"], r["model"]))
        col.caption("Test-set confusion matrix.")
    st.pyplot(plots.generalization_plot(table.to_dict("records")))
    st.caption("Train vs validation vs test AUC per model. A big gap between train and the others means the model memorised the training data; "
               "similar bars mean it generalises.")

# ============================================================ tab 4: predict
with tab_pred:
    st.subheader("Predict a single customer")
    prep = d["prep"]
    st.caption("Inputs are constrained to what the model has seen in training (ranges shown) and to combinations that exist in the data.")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input("Tenure (months)", 0, int(prep.max["tenure"]), 12, help=f"Training range 0-{int(prep.max['tenure'])}")
        monthly = st.number_input("Monthly charges", float(prep.min["MonthlyCharges"]), float(prep.max["MonthlyCharges"]), 70.0,
                                  help=f"Training range {prep.min['MonthlyCharges']:.0f}-{prep.max['MonthlyCharges']:.0f}")
        manual_total = st.checkbox("Enter total charges manually")
        total = st.number_input("Total charges", 0.0, float(prep.max["TotalCharges"]), float(tenure * monthly)) if manual_total else float(tenure * monthly)
        if not manual_total:
            st.caption(f"Total charges = tenure × monthly = {total:,.0f}")
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
        gender = st.selectbox("Gender", prep.categories["gender"])
        partner = st.selectbox("Partner", prep.categories["Partner"])
        dependents = st.selectbox("Dependents", prep.categories["Dependents"])
    with c2:
        contract = st.selectbox("Contract", prep.categories["Contract"])
        payment = st.selectbox("Payment method", prep.categories["PaymentMethod"])
        paperless = st.selectbox("Paperless billing", prep.categories["PaperlessBilling"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["Yes", "No"] if phone == "Yes" else ["No phone service"])
        internet = st.selectbox("Internet service", prep.categories["InternetService"], index=prep.categories["InternetService"].index("Fiber optic"))
    with c3:
        addon_opts = ["No", "Yes"] if internet != "No" else ["No internet service"]
        if internet == "No":
            st.info("Internet = No → add-ons fixed to 'No internet service' (as in the data).")
        addons = {a: st.selectbox(a, addon_opts) for a in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}
    rec = pd.DataFrame([{"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
                         "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, **addons, "Contract": contract,
                         "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total}])
    if use_fe:
        rec = add_features(rec)
    p = float(model.predict_proba(prep.transform(rec))[0])
    band = "High" if p >= 0.5 else "Medium" if p >= 0.25 else "Low"
    c = st.columns(4)
    c[0].metric("Churn probability", f"{p:.1%}")
    c[1].metric("Prediction", "Will churn" if p >= thr else "Will stay")
    c[2].metric("Risk band", band)
    c[3].metric("Threshold", f"{thr:.2f}")
    st.markdown("##### Factors moving this customer's risk")
    ex = pd.DataFrame(explain(model.predict_proba, prep, rec))
    ex["value"] = ex["value"].astype(str)
    ex["effect on risk"] = ex["effect on risk"].map(lambda v: f"{v:+.1%}")
    st.dataframe(ex, hide_index=True)
    st.caption("Each factor is reset to the typical training value; the change in probability is its effect. "
               "These are associations in historical data, not guaranteed causes.")

# ============================================================ tab 5: segments
with tab_seg:
    st.subheader("Customer segments (K-means, no labels used)")
    k = st.slider("Number of clusters k", 2, 7, 3)
    km, sil, coords, ratio = segments(use_fe, seed, k)
    Xtr = d["X"].iloc[d["idx"][0]]
    prof = Xtr[["tenure", "MonthlyCharges", "TotalCharges"]].fillna(Xtr["TotalCharges"].median()).groupby(km.labels_).mean()
    prof["churn_rate"] = pd.Series(d["y_tr"]).groupby(km.labels_).mean()
    prof["customers"] = np.bincount(km.labels_)
    st.metric("Silhouette (sampled)", f"{sil:.3f}")
    a, b = st.columns(2)
    a.pyplot(plots.pca_scatter(coords, km.labels_, ratio))
    a.caption("All customers squashed to 2 dimensions with PCA and coloured by cluster. Well-separated colours = distinct segments; "
              "the axis labels show how much of the data's variation each axis keeps.")
    b.dataframe(prof.round(2).rename_axis("cluster"))
    b.caption("Average tenure, charges and churn rate per cluster. Silhouette runs from -1 to 1; higher means tighter, better separated clusters.")
    st.pyplot(plots.cluster_profile_plot(prof))
    st.caption("The same cluster averages as bars, so you can name each segment (e.g. 'new, high-bill, high-churn').")
    st.pyplot(plots.elbow_plot(*elbow(use_fe, seed)))
    st.caption("Helps choose k. Inertia (spread inside clusters) always falls as k grows - look for the 'elbow' where gains flatten. "
               "Silhouette should peak at a good k.")

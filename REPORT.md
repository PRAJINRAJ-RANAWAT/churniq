# ChurnIQ — Project Report
**Telecom customer churn prediction with machine-learning algorithms written from scratch (NumPy)**

> The *why* behind every formula is in [`MATH_INTUITION.md`](MATH_INTUITION.md). This report covers the *what* and the *results*.

## 1. Problem
A telecom operator loses revenue when customers leave, and winning a new customer costs more than keeping one. Given a customer's account data, we estimate the **probability of churn**, flag high-risk customers, and describe customer segments.
Task type: **binary classification** (`Churn` = Yes/No) + unsupervised **clustering**.

## 2. Data
* Kaggle *Telecom Customer Churn* (`data/raw/IT_customer_churn.csv`): **7,043 customers, 19 features + target**.
* 4 numeric (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) and 15 categorical features.
* **Churn rate 26.5%** → imbalanced (a "never churn" model scores 73.5% accuracy but is useless).
* 11 blank `TotalCharges`, all with `tenure = 0` (new customers) → median-imputed.

## 3. Pipeline
```
load CSV → EDA → feature engineering → stratified 70/15/15 split → preprocessing (fit on train)
        → train model → pick threshold on validation → evaluate once on test → predict new customers
                                                         └→ K-means + PCA segments
```
| Stage | File | Notes |
|---|---|---|
| Load, features, split, preprocessing | `pipeline.py` | median/mode impute, z-score, one-hot → 48 inputs |
| Models | `models.py` | Logistic Regression, KNN, Gaussian NB, Decision Tree, K-means, PCA — **all from scratch** |
| Metrics | `metrics.py` | confusion matrix, precision, recall, F1, ROC, AUC, threshold search |
| Figures | `plots.py` | every graph shown in the app |
| UI | `app.py` | Streamlit: sidebar parameters, constraints, prediction |

## 4. EDA findings
* **Contract:** month-to-month churn **42.7%**, one-year 11.3%, two-year 2.8%.
* **Tenure:** churn falls steadily with tenure (r = −0.35); the first months are the riskiest.
* **Internet:** fibre-optic customers churn **41.9%** vs DSL 19.0% and no-internet 7.4%.
* **Payment:** electronic-check users churn **45.3%** vs ≈15–19% for the other methods.
* `tenure` and `TotalCharges` are strongly correlated (0.83) — expected, since total ≈ tenure × monthly.
These are *associations*; they do not prove that changing a feature causes customers to stay.

## 5. Feature engineering
Three features were added: `NumServices`, `AvgMonthlySpend`, `ChargePerService` (can be switched off in the sidebar to compare). In the depth-4 tree, `ChargePerService` is the 2nd most important feature (0.22 importance) and `AvgMonthlySpend` also appears, showing the engineered ratios carry signal the raw columns do not expose directly.

## 6. Models and results (test set, 1,056 customers, threshold chosen on validation by best F1)

| Model | Parameters | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | η=0.1, 1500 iter, λ=0 | 0.28 | 0.761 | 0.534 | 0.789 | 0.637 | **0.849** |
| K-Nearest Neighbors | k=31 | 0.33 | 0.748 | 0.516 | 0.804 | 0.628 | 0.839 |
| Gaussian Naive Bayes | smoothing 1e-6 | 0.98 | 0.724 | 0.488 | 0.811 | 0.609 | 0.824 |
| Decision Tree | depth 4 | 0.29 | 0.731 | 0.495 | 0.768 | 0.602 | 0.811 |

(Reproduce in the app: **3 · Model & results → Compare all four models**. Different seeds shift each number by about ±0.01–0.02.)

**Reading the table**
* Logistic regression is best on AUC and F1 and is also the most interpretable → it is the default model.
* The gaps between models are small; with ~1,000 test rows, differences of ~0.01 AUC are within noise.
* Recall ≈ 0.8 means roughly 4 of 5 churners are caught; precision ≈ 0.5 means about half the flagged customers would actually have left — the price of high recall.
* Naive Bayes needs a threshold of 0.98 because its independence assumption makes probabilities over-confident; it ranks customers reasonably (AUC 0.82) but its raw probabilities should not be read literally.

## 7. Over-/under-fitting (train AUC vs validation AUC)
| Model / setting | Train AUC | Validation AUC | Verdict |
|---|---|---|---|
| Tree depth 1 | 0.728 | 0.730 | underfit (too simple) |
| Tree depth 4 | 0.835 | 0.812 | good |
| Tree depth 8 | 0.905 | 0.791 | overfitting |
| Tree depth 20 | 1.000 | 0.669 | memorises training data |
| KNN k=1 | 0.997 | 0.656 | severe overfit |
| KNN k=31 | 0.854 | 0.817 | good |
| KNN k=201 | 0.837 | 0.811 | slightly underfit |
| Logistic λ=0 | 0.849 | 0.829 | good (small gap) |
| Logistic λ=8000 | 0.828 | 0.810 | over-regularised |

The app's **"Run over/under-fitting sweep"** button draws these curves for the selected model.

## 8. Drivers of churn (logistic regression, standardised features)
Strongest weights: `tenure` (−0.84), `Contract = Two year` (−0.71), `Contract = Month-to-month` (+0.56), plus phone/billing indicators. Longer tenure and long contracts lower risk; month-to-month contracts raise it — consistent with the EDA.

## 9. Decision constraint (the "constraints" in the app)
Instead of a fixed 0.5 cut-off, the sidebar offers three threshold rules:
1. **Best F1** on validation (balanced).
2. **Minimum recall** — e.g. "catch ≥ 80% of churners": the app picks the highest threshold that meets it (logistic regression: threshold 0.23, i.e. flag anyone with ≥ 23% risk).
3. **Manual** slider with live precision/recall/F1 curves.

Input constraints on the prediction form: numeric fields are limited to the training range, and impossible service combinations are blocked/auto-fixed.

## 10. Customer segments (K-means, k=3 default)
Fitted on features only (no label). Cluster sizes on the training set: 1,085 / 1,712 / 2,133. Churn rate is computed *afterwards* to describe each segment; PCA gives a 2-D picture and silhouette/elbow help choose k. Segments are descriptive, not "true" customer types.

## 11. Limitations
* One dataset, one time period — behaviour drifts, so the model needs re-training/monitoring.
* Probabilities are not calibrated (especially Naive Bayes and shallow trees); treat them as risk scores.
* 40 exact duplicate rows exist and can fall in different splits (no customer ID to tell them apart).
* Feature effects are associations, not causes; actions suggested from them are hypotheses to test (A/B test an offer).
* Keep a human in the loop for any consequential action.

## 12. How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

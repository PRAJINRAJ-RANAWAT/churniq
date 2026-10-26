# ChurnIQ — Simple Telecom Churn Prediction (ML from scratch)

```bash
pip install -r requirements.txt
streamlit run app.py
```

| File | Pipeline step |
|---|---|
| `pipeline.py` | load → feature engineering → stratified split → preprocessing |
| `models.py` | Logistic Regression, KNN, Naive Bayes, Decision Tree, K-means, PCA (NumPy only) |
| `metrics.py` | confusion matrix, precision/recall/F1, ROC-AUC, threshold search |
| `plots.py` | all graphs |
| `app.py` | Streamlit app (sidebar = parameters & constraints; tabs = pipeline steps) |
| `MATH_INTUITION.md` | the math and intuition behind every step |
| `REPORT.md` | results and analysis |

Dataset: `data/raw/IT_customer_churn.csv` ([Kaggle](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn)).

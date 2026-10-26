# Mathematical Intuition Behind Every Step

Notation: `n` customers, `x` = one customer's feature vector, `y ∈ {0,1}` (1 = churned), `p = P(y=1 | x)`.
Each section says **what the code does → the math → why it works**, and names the file where it lives.

---

## Step 1 — Load & clean (`pipeline.load_data`)
`TotalCharges` is text because 11 rows are blank. They are all customers with `tenure = 0` (they have not been billed yet), so blank → `NaN` → filled later.
`Churn` is mapped `Yes → 1`, `No → 0`, turning the problem into **binary classification**: learn a function `f(x) → p`.

---

## Step 2 — EDA (`plots.py`)

| Plot | Math behind it | What to read |
|---|---|---|
| Class balance | Base rate `π = mean(y) ≈ 0.265` | A model that always says "stay" gets 73.5% accuracy — so **accuracy alone is misleading**; we need precision/recall/AUC. |
| Histograms (density) | Estimate the distribution `P(x \| y)` for each class | If the two histograms differ, the feature carries information about churn. |
| Churn rate by category | Conditional probability `P(y=1 \| category) = churners in group / group size` | Compare each bar to the dashed overall rate `π`. Month-to-month ≈ 43% vs two-year ≈ 3%. |
| Churn rate vs tenure (equal-count bins) | Same conditional rate on quantile bins, so each point has the same sample size | A falling curve = loyalty grows with time. |
| Correlation heatmap | Pearson `r = Cov(a,b) / (σ_a σ_b)` ∈ [−1, 1] | Detects redundant features (tenure ↔ TotalCharges ≈ 0.83). Only measures **linear** association. |

---

## Step 3 — Feature engineering (`pipeline.add_features`, `Preprocessor`)

### 3.1 Stratified split — 70 / 15 / 15
Each class is shuffled and cut separately so train, validation and test all keep `π ≈ 26.5%`.
**Why three sets?** Training data fits the parameters; validation data chooses hyper-parameters and the threshold; test data is touched **once** to give an honest estimate of future performance. If you tune on the test set, the test score becomes optimistic (you have fitted to it).

### 3.2 Imputation — median / mode
Median minimises `Σ|x_i − c|` so it is robust to outliers; mode is the most frequent category. Both are computed **on train only**.

### 3.3 Standardisation (z-score)
`z = (x − μ) / σ` (μ, σ from train).
**Why:** distance-based and gradient-based models are dominated by large-scale columns. `TotalCharges` (0–8000) would swamp `SeniorCitizen` (0/1) in a Euclidean distance. After scaling every column has mean 0 and variance 1, so each contributes comparably. Gradient descent also converges faster because the loss surface is rounder.

### 3.4 One-hot encoding
A category with `m` levels becomes `m` binary columns. Ordinal codes (Month=0, Year=1, Two-year=2) would impose a fake ordering and spacing; one-hot lets the model learn a separate weight per level. Unseen levels become all zeros.

### 3.5 Why fit on train only (no leakage)
If μ, σ or medians were computed on all data, test information would leak into training. The test set must behave like *future, unseen customers*.

### 3.6 Engineered features
* `NumServices = Σ 1[service = "Yes"]` — customers with many services are more "locked in".
* `AvgMonthlySpend = TotalCharges / tenure` — what the customer *actually* paid per month; comparing it with `MonthlyCharges` reveals recent price increases.
* `ChargePerService = MonthlyCharges / (NumServices + 1)` — price per unit of value (+1 avoids division by zero).

Ratios let a **linear** model represent a non-linear relationship it could not express from the raw columns alone.

---

## Step 4 — Models (`models.py`)

### 4.1 Logistic regression
1. **Linear score:** `z = w·x + b` (any real number).
2. **Sigmoid** squashes it to a probability: `p = 1 / (1 + e^{−z})`. Equivalent statement: the **log-odds are linear**, `log(p/(1−p)) = w·x + b`. A coefficient `w_j` means "one unit (one std-dev) increase of feature j multiplies the odds of churn by `e^{w_j}`".
3. **Loss — binary cross-entropy** (negative log-likelihood of the Bernoulli model):
   `L = −(1/m) Σ [ y log p + (1−y) log(1−p) ]`
   If `y=1` the loss is `−log p`: tiny when `p→1`, huge when `p→0`. Confident wrong answers are punished heavily. This loss is **convex**, so gradient descent reaches the global minimum.
4. **Gradient (beautifully simple):** `∂L/∂w = (1/m) Xᵀ(p − y)`, `∂L/∂b = mean(p − y)`. "Move each weight in proportion to how much that feature correlates with the prediction error."
5. **Update:** `w ← w − η · ∂L/∂w`. Learning rate `η` too big → loss oscillates/diverges; too small → slow.
6. **L2 regularisation:** add `(λ/2m)‖w‖²` → gradient gains `(λ/m) w`. It shrinks weights toward 0, trading a little training fit for lower variance (less overfitting). The bias `b` is not penalised.

*Code detail:* the sigmoid is written piecewise so `exp()` never sees a big positive number (overflow); log uses `clip(p, 1e-12, 1−1e-12)` to avoid `log 0`.

### 4.2 K-Nearest Neighbours
`p(x) = (1/k) Σ_{i ∈ N_k(x)} y_i`, where `N_k(x)` = the k training points with smallest Euclidean distance `‖x − x_i‖`.
* **Intuition:** similar customers behave similarly — "ask the k most similar customers what they did".
* **k = bias–variance dial.** k=1 follows noise (zero training error, high variance → overfits). k=n predicts the base rate for everyone (high bias → underfits). Pick k where *validation* AUC peaks.
* **Fast distance trick:** `‖a−b‖² = ‖a‖² − 2a·b + ‖b‖²` — one matrix product computes all distances.
* Requires standardised features (distance is scale-dependent) and suffers when many columns are irrelevant ("curse of dimensionality": in high dimensions all points look equally far).

### 4.3 Gaussian Naive Bayes
**Bayes' theorem:** `P(y|x) = P(y) P(x|y) / P(x)`. "Naive" = assume features are independent given the class, so
`P(x|y) = Π_j N(x_j; μ_yj, σ²_yj)` with Gaussian density `1/√(2πσ²) · exp(−(x−μ)²/(2σ²))`.
* Training = just computing a mean and variance per class per feature, and the prior `P(y)`.
* Products of many small numbers underflow, so we add **logs**: `log P(y) + Σ_j log N(·)`. The two classes are compared through the log-ratio and converted with a sigmoid: `P(y=1|x) = σ(logP₁ − logP₀)`.
* Weakness: one-hot columns are not Gaussian and correlated features are double-counted, so probabilities are over-confident (the best threshold is ≈ 0.98!). Ranking (AUC) is still decent.
* Variance smoothing adds a small `ε` to every variance so a near-constant feature does not produce division by ≈ 0.

### 4.4 Decision tree (CART, Gini)
* **Gini impurity** of a node with churn fraction `p`: `G = 1 − p² − (1−p)² = 2p(1−p)`. It is 0 for a pure node and 0.5 for a 50/50 mix.
* **Best split:** for feature `f` and threshold `t`, send `x_f ≤ t` left. Gain
  `ΔG = G(parent) − (n_L/n) G(left) − (n_R/n) G(right)`.
  Choose the `(f, t)` with the largest gain; recurse on both children.
* **Efficient search:** sort a feature once, then cumulative sums of labels give the left/right churn counts for **every** threshold at once (O(n log n) per feature).
* Leaf prediction = fraction of churners in the leaf.
* **Depth controls complexity.** Unlimited depth memorises training data (train AUC → 1, validation drops = overfit). Hence `max_depth` and `min_samples_leaf`.
* **Feature importance** = total weighted Gini decrease contributed by splits on that feature, normalised to sum 1.

---

## Step 5 — Evaluation (`metrics.py`)

Confusion matrix: **TP** churners caught, **FP** false alarms, **FN** missed churners, **TN** correctly left alone.

| Metric | Formula | Meaning (churn context) |
|---|---|---|
| Accuracy | (TP+TN)/n | Overall correctness — misleading with 26.5% positives. |
| Precision | TP/(TP+FP) | Of customers we flag, how many really leave → cost of wasted offers. |
| Recall (sensitivity) | TP/(TP+FN) | Of customers who leave, how many we catch → revenue we can save. |
| Specificity | TN/(TN+FP) | Of loyal customers, how many we leave alone. |
| F1 | 2PR/(P+R) | Harmonic mean — high only if both are high (it punishes imbalance more than an average would). |

### ROC & AUC
Sweep the threshold from 1 → 0. Each threshold gives `TPR = recall` and `FPR = FP/(FP+TN)`; plotting them gives the ROC curve. The area, computed by the **trapezoid rule** `Σ (x_{i+1} − x_i)(y_i + y_{i+1})/2`, equals
**AUC = P(score of a random churner > score of a random non-churner)**. 0.5 = coin flip, 1.0 = perfect ranking. AUC does not depend on the threshold or on calibration, which is why we use it to compare models.

### Threshold selection
The model outputs a probability; the *decision* needs a cut-off `t`: predict churn if `p ≥ t`.
* Lower `t` → recall ↑, precision ↓. Higher `t` → the reverse.
* **Best F1:** pick the `t` maximising F1 on **validation**.
* **Recall constraint (sidebar):** "we must catch ≥ X% of churners" → choose the **highest** `t` with validation recall ≥ X, i.e. the fewest false alarms that still satisfy the business requirement.
* The threshold is never tuned on test data.

### Over/under-fitting
Compare train vs validation AUC while one complexity knob moves (λ, k, depth). Large gap = **high variance** (overfit). Both low = **high bias** (underfit). Best setting: validation score is maximal.

---

## Step 6 — Prediction & explanation (`app.py`, `pipeline.explain`)
A new customer goes through the **same fitted** preprocessing (same μ, σ, categories) → model → `p` → compare with the threshold.
**Constraints on input:** numeric fields are bounded by the training range (a linear model extrapolates unreliably outside it), and impossible combos are blocked (internet = No ⇒ add-ons = "No internet service"; phone = No ⇒ multiple lines = "No phone service") because those combinations never occur in the data.
**Explanation:** for each feature, replace the customer's value by the typical training value (median/mode) and record `Δp = p(actual) − p(typical)`. A positive Δ means that feature value pushes this customer's risk up. It is a one-at-a-time *what-if*: an **association**, not proof of cause.

---

## Step 7 — K-means & PCA (unsupervised, no labels)

### K-means
Minimise **inertia** `J = Σ_i ‖x_i − μ_{c(i)}‖²`.
Lloyd's algorithm alternates (a) *assign* each point to its nearest centroid, (b) *update* each centroid to the mean of its points. Each step cannot increase `J`, so it converges (to a local minimum). Because the minimum is local we use **k-means++** (pick next centres with probability ∝ distance², spreading them out) and several restarts, keeping the lowest `J`.
* **Elbow:** `J` always falls as k grows; look for where the gain flattens.
* **Silhouette:** `s = (b − a)/max(a, b)`; `a` = mean distance to own cluster, `b` = mean distance to the nearest other cluster. `s→1` well separated, `0` overlapping, `<0` misassigned.
* Churn is computed **after** clustering purely to describe segments.

### PCA
1. Centre the data. 2. Covariance `C = XᵀX/(n−1)`. 3. Eigen-decompose `C v = λ v`.
Eigenvectors are orthogonal directions; eigenvalue `λ` = variance along that direction. Keeping the top two gives the 2-D picture that preserves the most variance: projection `Z = (X − μ) V₂`. Explained-variance ratio `λ_i / Σλ` tells how much information the 2-D plot retains (the rest is hidden).

---

## Cheat-sheet: the one-sentence version of each step
| Step | One sentence |
|---|---|
| Split | Keep a never-seen test set so the final score is honest. |
| Scale | Put all features on the same ruler. |
| One-hot | Give each category its own weight, no fake order. |
| Logistic reg. | Linear log-odds + cross-entropy + gradient descent. |
| KNN | Vote among the k most similar customers. |
| Naive Bayes | Bayes' rule with independent Gaussian features. |
| Tree | Ask the yes/no question that makes groups purest, repeat. |
| ROC-AUC | Probability a churner is ranked above a loyal customer. |
| Threshold | Turn a probability into an action, tuned to a business constraint. |
| K-means / PCA | Group similar customers / draw them in 2-D. |

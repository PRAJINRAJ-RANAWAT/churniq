"""Step 4: every model written from scratch with NumPy (no scikit-learn).
All classifiers share the same API: fit(X, y) and predict_proba(X) -> P(churn) for each row."""
import numpy as np


# ============================================================ Logistic regression
def sigmoid(z):
    """1/(1+e^-z), written so exp() is never called on a large positive number."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """p(churn|x) = sigmoid(x.w + b), fitted by gradient descent on binary cross-entropy (+ L2)."""

    def __init__(self, learning_rate=0.1, n_iter=1000, l2=0.0):
        self.learning_rate, self.n_iter, self.l2 = learning_rate, n_iter, l2

    def _loss(self, p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        self.w, self.b = np.zeros(n), 0.0
        self.history, self.val_history = [], []
        for _ in range(self.n_iter):
            err = sigmoid(X @ self.w + self.b) - y                   # p - y
            self.w -= self.learning_rate * (X.T @ err / m + self.l2 / m * self.w)
            self.b -= self.learning_rate * err.mean()
            self.history.append(self._loss(sigmoid(X @ self.w + self.b), y))
            if X_val is not None:
                self.val_history.append(self._loss(sigmoid(X_val @ self.w + self.b), y_val))
            if len(self.history) > 1 and abs(self.history[-2] - self.history[-1]) < 1e-9:
                break
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


# ============================================================ K-nearest neighbours
class KNN:
    """Lazy learner: P(churn) = fraction of churners among the k nearest training customers."""

    def __init__(self, k=31):
        self.k = int(k)

    def fit(self, X, y):
        self.X_, self.y_ = X, y.astype(int)
        self._sq = (X ** 2).sum(axis=1)
        return self

    def predict_proba(self, Q, chunk=512):
        out = []
        for s in range(0, len(Q), chunk):                         # chunking keeps memory small
            q = Q[s:s + chunk]
            d2 = (q ** 2).sum(axis=1)[:, None] - 2 * q @ self.X_.T + self._sq[None, :]   # |a-b|^2 expanded
            nn = np.argsort(d2, axis=1, kind="stable")[:, :self.k]
            out.append(self.y_[nn].mean(axis=1))
        return np.concatenate(out)


# ============================================================ Gaussian Naive Bayes
class GaussianNB:
    """P(y|x) ∝ P(y) · Π_j Normal(x_j; mean_yj, var_yj), computed in log space."""

    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.log_prior = np.log([(y == c).mean() for c in (0, 1)])
        self.mu = np.array([X[y == c].mean(axis=0) for c in (0, 1)])
        eps = max(self.var_smoothing * X.var(axis=0).max(), 1e-12)   # keeps variance > 0
        self.var = np.array([X[y == c].var(axis=0) for c in (0, 1)]) + eps
        return self

    def predict_proba(self, X):
        jll = [self.log_prior[c] - 0.5 * np.sum(np.log(2 * np.pi * self.var[c]) + (X - self.mu[c]) ** 2 / self.var[c], axis=1)
               for c in (0, 1)]
        return 1.0 / (1.0 + np.exp(-np.clip(jll[1] - jll[0], -700, 700)))


# ============================================================ Decision tree (CART, Gini)
def gini(n_pos, n):
    p = n_pos / n
    return 1.0 - p ** 2 - (1.0 - p) ** 2


class DecisionTree:
    """Greedy recursive splitting on the feature/threshold that lowers Gini impurity the most.
    Nodes are kept in flat lists (index 0 = root)."""

    def __init__(self, max_depth=4, min_samples_leaf=1):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf

    def fit(self, X, y):
        self.n_features, self.n_total = X.shape[1], len(y)
        self.feature, self.thresh, self.left, self.right, self.proba, self.depth_of, self.gain = [], [], [], [], [], [], []
        self._build(X, y.astype(int), 0)
        imp = np.zeros(self.n_features)
        for f, g in zip(self.feature, self.gain):
            if f >= 0:
                imp[f] += g
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def _build(self, X, y, depth):
        node = len(self.feature)
        for lst, v in ((self.feature, -1), (self.thresh, 0.0), (self.left, -1), (self.right, -1),
                       (self.proba, float(y.mean())), (self.depth_of, depth), (self.gain, 0.0)):
            lst.append(v)
        n = len(y)
        if y.min() == y.max() or n < 2 * self.min_samples_leaf or (self.max_depth is not None and depth >= self.max_depth):
            return node
        f, t, gain = self._best_split(X, y)
        if f < 0:
            return node
        mask = X[:, f] <= t
        self.feature[node], self.thresh[node] = f, t
        self.gain[node] = gain * n / self.n_total
        self.left[node] = self._build(X[mask], y[mask], depth + 1)
        self.right[node] = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """For every feature: sort once, then use cumulative class counts to score all split points at once."""
        n, total_pos = len(y), y.sum()
        parent = gini(total_pos, n)
        best_f, best_t, best_gain = -1, 0.0, 1e-12
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            if xs[0] == xs[-1]:
                continue
            n_l = np.arange(1, n)
            pos_l = np.cumsum(ys)[:-1]
            n_r, pos_r = n - n_l, total_pos - pos_l
            valid = (xs[1:] != xs[:-1]) & (n_l >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
            if not valid.any():
                continue
            weighted = (n_l * gini(pos_l, n_l) + n_r * gini(pos_r, n_r)) / n
            gain = np.where(valid, parent - weighted, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_f, best_gain, best_t = f, float(gain[i]), float((xs[i] + xs[i + 1]) / 2)
        return best_f, best_t, best_gain

    def predict_proba(self, X):
        feature, thresh = np.array(self.feature), np.array(self.thresh)
        left, right, proba = np.array(self.left), np.array(self.right), np.array(self.proba)
        node = np.zeros(len(X), dtype=int)
        active = feature[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            go_left = X[idx, feature[node[idx]]] <= thresh[node[idx]]
            node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
            active = feature[node] >= 0
        return proba[node]


# ============================================================ K-means + PCA (unsupervised)
def _sq_dists(X, C):
    return np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0.0)


class KMeans:
    """Lloyd's algorithm: assign each point to the nearest centroid, move centroids to the mean, repeat.
    k-means++ start and several restarts (keep the lowest inertia). Never sees the churn label."""

    def __init__(self, n_clusters=3, n_init=5, max_iter=300, seed=42):
        self.n_clusters, self.n_init, self.max_iter, self.seed = n_clusters, n_init, max_iter, seed

    def _run(self, X, rng):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):                        # k-means++: far points are likelier centres
            d2 = _sq_dists(X, np.array(C)).min(axis=1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(self.max_iter):
            d2 = _sq_dists(X, C)
            labels = d2.argmin(axis=1)
            newC = np.array([X[labels == k].mean(axis=0) if (labels == k).any() else X[d2.min(axis=1).argmax()]
                             for k in range(self.n_clusters)])
            moved = np.sqrt(((newC - C) ** 2).sum())
            C = newC
            if moved < 1e-6:
                break
        d2 = _sq_dists(X, C)
        return C, d2.argmin(axis=1), float(d2.min(axis=1).sum())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best = min((self._run(X, rng) for _ in range(self.n_init)), key=lambda r: r[2])
        self.centroids_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        return _sq_dists(X, self.centroids_).argmin(axis=1)


def silhouette_score(X, labels):
    """s = (b - a) / max(a, b): a = mean distance to own cluster, b = mean distance to nearest other cluster."""
    D = np.sqrt(_sq_dists(X, X))
    ks, s = np.unique(labels), np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


class PCA:
    """Centre -> covariance matrix -> eigenvectors with the largest eigenvalues (numpy.linalg.eigh) -> project."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        vals, vecs = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
        order = np.argsort(vals)[::-1]
        self.components_ = vecs[:, order[:self.n_components]].T
        self.explained_variance_ratio_ = vals[order[:self.n_components]] / vals.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


MODEL_NAMES = {"logistic_regression": "Logistic Regression", "knn": "K-Nearest Neighbors",
               "decision_tree": "Decision Tree", "gaussian_nb": "Gaussian Naive Bayes"}


def build_model(name, params):
    """Create a model from its name and a parameter dict (used by the app and the sweeps)."""
    return {"logistic_regression": lambda: LogisticRegression(**params), "knn": lambda: KNN(**params),
            "decision_tree": lambda: DecisionTree(**params), "gaussian_nb": lambda: GaussianNB(**params)}[name]()

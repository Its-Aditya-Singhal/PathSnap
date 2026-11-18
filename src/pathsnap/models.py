"""Model zoo: candidate classifiers that score P(candidate arc is the true arc).

Each entry: algorithm family, a factory taking hyperparameters, and the
initial (untuned) hyperparameters. `fit_model` returns (model, history),
where history holds per-epoch / per-iteration train & validation log loss
for models that actually produce one (MLP, LightGBM, XGBoost) and None
otherwise.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


class MLPWithHistory(BaseEstimator, ClassifierMixin):
    """Scaled MLP trained epoch-by-epoch so real train/val loss curves are kept."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024, epochs=15,
                 random_state=SEED):
        self.hidden, self.alpha, self.lr = hidden, alpha, lr
        self.batch_size, self.epochs, self.random_state = batch_size, epochs, random_state

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                  learning_rate_init=self.lr, batch_size=self.batch_size,
                                  random_state=self.random_state)
        rng = np.random.default_rng(self.random_state)
        sub = rng.choice(len(Xs), size=min(len(Xs), 200_000), replace=False)
        self.history_ = {"train_logloss": [], "val_logloss": []}
        for _ in range(self.epochs):
            self.mlp_.partial_fit(Xs, y, classes=[0, 1])
            self.history_["train_logloss"].append(log_loss(y[sub], self.mlp_.predict_proba(Xs[sub])[:, 1]))
            if Xv is not None:
                self.history_["val_logloss"].append(log_loss(y_val, self.mlp_.predict_proba(Xv)[:, 1]))
        self.classes_ = self.mlp_.classes_
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _gnb(var_smoothing=1e-9):
    return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=var_smoothing))


def _knn(n_neighbors=25, weights="distance"):
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1))


def _rf(n_estimators=200, min_samples_leaf=20, max_features="sqrt", max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, max_depth=max_depth,
                                  n_jobs=-1, random_state=SEED)


def _lgbm(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                          num_leaves=num_leaves, min_child_samples=min_child_samples,
                          subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
                          reg_lambda=reg_lambda, random_state=SEED, verbose=-1)


def _xgb(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
         colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=subsample,
                         colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                         reg_lambda=reg_lambda, tree_method="hist", eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


def _mlp(hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024, epochs=15):
    return MLPWithHistory(hidden=hidden, alpha=alpha, lr=lr, batch_size=batch_size, epochs=epochs)


MODELS = {
    "logreg":  dict(family="Linear", factory=_lr, params=dict(C=1.0)),
    "gnb":     dict(family="Probabilistic", factory=_gnb, params=dict(var_smoothing=1e-9)),
    "knn":     dict(family="Instance-based", factory=_knn, params=dict(n_neighbors=25, weights="distance"),
                    train_subsample=200_000),
    "rf":      dict(family="Ensemble (bagging)", factory=_rf,
                    params=dict(n_estimators=200, min_samples_leaf=20, max_features="sqrt")),
    "lightgbm": dict(family="Boosting", factory=_lgbm,
                     params=dict(n_estimators=400, learning_rate=0.05, num_leaves=63)),
    "xgboost": dict(family="Boosting", factory=_xgb,
                    params=dict(n_estimators=400, learning_rate=0.05, max_depth=6)),
    "mlp":     dict(family="Neural network", factory=_mlp,
                    params=dict(hidden=(64, 32), alpha=1e-4, lr=1e-3, epochs=15)),
}


def fit_model(name, X, y, X_val=None, y_val=None, params=None, seed=SEED):
    spec = MODELS[name]
    params = {**spec["params"], **(params or {})}
    model = spec["factory"](**params)
    n_sub = spec.get("train_subsample")
    if n_sub and len(X) > n_sub:
        idx = np.random.default_rng(seed).choice(len(X), n_sub, replace=False)
        X, y = X[idx], y[idx]

    history = None
    if name == "mlp":
        model.fit(X, y, X_val, y_val)
        history = model.history_
    elif name == "lightgbm":
        ev = {}
        from lightgbm import record_evaluation
        sets = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
        model.fit(X, y, eval_set=sets, eval_metric="binary_logloss",
                  callbacks=[record_evaluation(ev)])
        history = {"train_logloss": ev["training"]["binary_logloss"]}
        if X_val is not None:
            history["val_logloss"] = ev["valid_1"]["binary_logloss"]
    elif name == "xgboost":
        sets = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
        model.fit(X, y, eval_set=sets, verbose=False)
        ev = model.evals_result()
        history = {"train_logloss": ev["validation_0"]["logloss"]}
        if X_val is not None:
            history["val_logloss"] = ev["validation_1"]["logloss"]
    else:
        model.fit(X, y)
    return model, history

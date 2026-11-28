"""Baselines, similarity-weighted averages and logistic regression (SPEC section 7)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import auc, log_loss
from features import Snapshot, global_rate, user_prior, weighted_average

# The 2 x 2 x 2 grid of similarity-weighted averages: similarity x set x time decay.
VARIANTS = {
    "J1": "jac_tag", "J2": "jac_ing", "C1": "cos_tag", "C2": "cos_ing",
}


def baseline_constant(snap: Snapshot, label: str) -> np.ndarray:
    """B0: the history's positive rate for every row."""
    return np.full(snap.n, global_rate(snap, label))


def tune_beta(snap: Snapshot, label: str, mask, betas=(1, 2, 5, 10, 20)) -> pd.DataFrame:
    """B1 shrinkage: pick beta by AUC (log loss breaks ties) on the masked rows."""
    y = snap.targets[f"y_{label}"].values[mask]
    rows = []
    for b in betas:
        p = user_prior(snap, label, b)[mask]
        rows.append({"beta": b, "AUC": auc(y, p), "LogLoss": log_loss(y, p)})
    return pd.DataFrame(rows).sort_values(["AUC", "LogLoss"], ascending=[False, True]).reset_index(drop=True)


def tune_weighted_average(snap: Snapshot, sim_col: str, label: str, prior, mask,
                          ks=(5, 10, 20, 50, 100), alphas=(0.5, 1, 2, 5, 10, 20),
                          lams=(0.0,)) -> pd.DataFrame:
    """Grid search k, alpha, lambda for one similarity column; sorted best first."""
    y = snap.targets[f"y_{label}"].values[mask]
    rows = []
    for k, a, lam in itertools.product(ks, alphas, lams):
        p = weighted_average(snap, sim_col, label, prior, k, a, lam)[mask]
        rows.append({"k": k, "alpha": a, "lam": lam, "AUC": auc(y, p), "LogLoss": log_loss(y, p)})
    return pd.DataFrame(rows).sort_values(["AUC", "LogLoss"], ascending=[False, True]).reset_index(drop=True)


def fit_lr(X: pd.DataFrame, y, C: float = 1.0, class_weight=None, sample_weight=None):
    """Standardize, then L2-regularized logistic regression."""
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=5000, class_weight=class_weight))
    return model.fit(X.values, np.asarray(y), logisticregression__sample_weight=sample_weight)


def user_weights(users) -> np.ndarray:
    """1 / sqrt(rows per user), so heavy users do not dominate training."""
    s = pd.Series(np.asarray(users))
    return 1 / np.sqrt(s.map(s.value_counts()).values)


def training_set(snaps: dict, X: dict, keys, label: str, weighted: bool, extra=()):
    """Stack the feature matrices of several snapshots (plus `extra` = [(X, targets)]) into
    one LR training set; returns X, y, sample weights (None when not weighted)."""
    parts = [(X[k], snaps[k].targets) for k in keys] + list(extra)
    Xs = pd.concat([x for x, _ in parts], ignore_index=True)
    t = pd.concat([tt for _, tt in parts], ignore_index=True)
    return Xs, t[f"y_{label}"].values, (user_weights(t.u.values) if weighted else None)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.values)[:, 1]


def lr_coefficients(model, columns) -> pd.DataFrame:
    """Coefficients on standardized features: change in log-odds per one standard deviation."""
    coef = model[-1].coef_.ravel()
    return (pd.DataFrame({"feature": list(columns), "coef": coef})
            .assign(abs=lambda d: d.coef.abs()).sort_values("abs", ascending=False)
            .drop(columns="abs").reset_index(drop=True))

"""Metrics with user-level bootstrap confidence intervals and paired comparisons (SPEC section 3).

Every function resamples users with the same seed, so all models (and all paired differences)
are scored on exactly the same bootstrap samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = 1e-6


def auc(y, p) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties get average ranks)."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


METRICS = {"AUC": auc, "LogLoss": log_loss, "Brier": brier}
HIGHER_IS_BETTER = {"AUC": True, "LogLoss": False, "Brier": False}


def metrics(y, p) -> dict:
    return {m: f(y, p) for m, f in METRICS.items()}


def bootstrap_indices(groups, n_boot: int, seed: int = 0) -> list:
    """Row indices of `n_boot` resamples that draw whole groups (users) with replacement."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    by_group = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    return [np.concatenate([by_group[g] for g in rng.integers(0, len(by_group), len(by_group))])
            for _ in range(n_boot)]


def _ci(values):
    return np.nanpercentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)


def evaluate(preds: dict, y, groups, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Point metrics and 95% CIs for every model."""
    y = np.asarray(y)
    preds = {k: np.asarray(v, dtype=float) for k, v in preds.items()}
    idx = bootstrap_indices(groups, n_boot, seed)
    rows = []
    for k, p in preds.items():
        boot = pd.DataFrame([metrics(y[i], p[i]) for i in idx])
        row = {"model": k}
        for m, v in metrics(y, p).items():
            row[m] = v
            row[f"{m}_lo"], row[f"{m}_hi"] = _ci(boot[m].values) if n_boot else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def compare(preds: dict, base: str, y, groups, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of metric(model) - metric(base) on the same resamples.
    `p_not_better` = share of resamples in which the model is not better than the base
    (a one-sided bootstrap p-value)."""
    y = np.asarray(y)
    preds = {k: np.asarray(v, dtype=float) for k, v in preds.items()}
    idx = bootstrap_indices(groups, n_boot, seed)
    base_point = metrics(y, preds[base])
    base_boot = pd.DataFrame([metrics(y[i], preds[base][i]) for i in idx])
    rows = []
    for k, p in preds.items():
        if k == base:
            continue
        boot = pd.DataFrame([metrics(y[i], p[i]) for i in idx]) - base_boot
        row = {"model": k}
        for m, v in metrics(y, p).items():
            d = boot[m].values
            row[f"Δ{m}"] = v - base_point[m]
            row[f"Δ{m}_lo"], row[f"Δ{m}_hi"] = _ci(d)
            row[f"p_not_better_{m}"] = float(np.mean(d <= 0 if HIGHER_IS_BETTER[m] else d >= 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def mean_ci(values: dict, groups, n_boot: int = 1000, seed: int = 0, base: str | None = None) -> pd.DataFrame:
    """Mean of per-row values (e.g. Hit@10 per ranking list) with 95% CIs; with `base`,
    also the paired difference to the base model."""
    values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    idx = bootstrap_indices(groups, n_boot, seed)
    rows = []
    for k, v in values.items():
        boot = np.array([v[i].mean() for i in idx])
        row = {"model": k, "mean": v.mean()}
        row["lo"], row["hi"] = _ci(boot)
        if base is not None and k != base:
            d = boot - np.array([values[base][i].mean() for i in idx])
            row["Δ"] = v.mean() - values[base].mean()
            row["Δ_lo"], row["Δ_hi"] = _ci(d)
            row["p_not_better"] = float(np.mean(d <= 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def pretty(table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """'0.6123 [0.6010, 0.6240]' strings for an `evaluate` or `compare` table."""
    out = pd.DataFrame(index=table.index)
    for c in table.columns:
        if c == "mean" and {"lo", "hi"} <= set(table.columns):
            out[c] = [f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
                      for v, lo, hi in zip(table["mean"], table["lo"], table["hi"])]
        elif f"{c}_lo" in table.columns:
            out[c] = [f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]" if c.startswith("Δ")
                      else f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
                      for v, lo, hi in zip(table[c], table[f"{c}_lo"], table[f"{c}_hi"])]
        elif c.startswith("p_not_better"):
            out[c] = table[c].map(lambda x: f"{x:.3f}")
    return out

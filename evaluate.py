"""
Evaluation on cold-start rows, with confidence intervals from a user-level bootstrap.

  - rating version (main task):  MSE, RMSE, MAE of predicted ratings      evaluate_rating(), compare_rating()
  - binary version (rating == 5): AUC and log loss of probabilities        evaluate(), compare()
  - ranking: Hit@K and NDCG@K of each 5-star cold row against sampled
    cold recipes the user never reviewed                                   sample_candidates(), ranking_metrics()

A few heavy users contribute a large share of the cold rows, so the bootstrap resamples users
(with all their rows) rather than individual rows.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

N_BOOT = 1000


def _mse(y, pred):
    return np.mean((y - pred) ** 2)


def _user_groups(users):
    return list(pd.Series(np.arange(len(users))).groupby(np.asarray(users)).indices.values())


def _bootstrap(y, scores, users, metric, n_boot=N_BOOT, seed=0):
    """metric of each score array on the same user-level bootstrap samples: array (n_boot, len(scores))."""
    rng = np.random.default_rng(seed)
    groups = _user_groups(users)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(len(groups), size=len(groups))])
        if metric is roc_auc_score and y[idx].min() == y[idx].max():
            continue
        out.append([metric(y[idx], s[idx]) for s in scores])
    return np.array(out)


def _paired(y, score_a, score_b, users, metric, n_boot):
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    diff = np.diff(_bootstrap(y, [b, a], users, metric, n_boot), axis=1)[:, 0]   # metric(a) - metric(b)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return metric(y, a) - metric(y, b), lo, hi, diff


# ---------- rating version ----------
def evaluate_rating(y, pred, users, n_boot=N_BOOT):
    """MSE with 95% user-bootstrap CI, RMSE and MAE. Predictions are clipped to the rating range [1, 5]."""
    y, pred = np.asarray(y, float), np.clip(np.asarray(pred, float), 1, 5)
    lo, hi = np.percentile(_bootstrap(y, [pred], users, _mse, n_boot)[:, 0], [2.5, 97.5])
    mse = _mse(y, pred)
    return {"MSE": mse, "CI low": lo, "CI high": hi, "RMSE": np.sqrt(mse),
            "MAE": np.mean(np.abs(y - pred)), "rows": len(y)}


def compare_rating(y, pred_a, pred_b, users, n_boot=N_BOOT):
    """Paired bootstrap of MSE(a) - MSE(b) (negative = a is better) and share of samples where a is not better."""
    y = np.asarray(y, float)
    d, lo, hi, diff = _paired(y, np.clip(pred_a, 1, 5), np.clip(pred_b, 1, 5), users, _mse, n_boot)
    return {"MSE diff": d, "CI low": lo, "CI high": hi, "p(a >= b)": (diff >= 0).mean()}


# ---------- binary version ----------
def evaluate(y, score, users, n_boot=N_BOOT, is_prob=None):
    """AUC with 95% user-bootstrap CI, and log loss (only for scores that are probabilities;
    by default, scores inside [0, 1])."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    lo, hi = np.percentile(_bootstrap(y, [score], users, roc_auc_score, n_boot)[:, 0], [2.5, 97.5])
    if is_prob is None:
        is_prob = score.min() >= 0 and score.max() <= 1
    return {"AUC": roc_auc_score(y, score), "CI low": lo, "CI high": hi,
            "log loss": log_loss(y, np.clip(score, 1e-6, 1 - 1e-6)) if is_prob else np.nan,
            "rows": len(y)}


def compare(y, score_a, score_b, users, n_boot=N_BOOT):
    """Paired bootstrap of AUC(a) - AUC(b): difference, 95% CI, and share of samples where a <= b."""
    d, lo, hi, diff = _paired(np.asarray(y), score_a, score_b, users, roc_auc_score, n_boot)
    return {"AUC diff": d, "CI low": lo, "CI high": hi, "p(a <= b)": (diff <= 0).mean()}


# ---------- ranking ----------
def sample_candidates(pos_rows, pool, reviewed, submitted, n_neg=99, seed=0):
    """Candidate lists for ranking: each positive row's recipe plus n_neg recipes from `pool` that the user
    never reviewed (`reviewed`: user -> set of recipe ids) and that were already submitted at the row's date."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool)
    pool_submitted = submitted.reindex(pool).values
    out = []
    for g, row in enumerate(pos_rows.itertuples()):
        ok = pool[(pool_submitted < np.datetime64(row.date)) & ~np.isin(pool, list(reviewed.get(row.user_id, ())))]
        negs = rng.choice(ok, size=min(n_neg, len(ok)), replace=False)
        out.append(pd.DataFrame({"group": g, "user_id": row.user_id, "date": row.date,
                                 "recipe_id": np.concatenate([[row.recipe_id], negs]),
                                 "is_pos": np.r_[1, np.zeros(len(negs), int)]}))
    return pd.concat(out, ignore_index=True)


def ranking_metrics(cands, score, k=10):
    """Hit@k and NDCG@k per candidate list (one positive each). Ties are broken at random, so the value is
    the expectation over the positive's possible ranks: a constant score gets Hit@10 = 10 / list length."""
    df = cands[["group", "is_pos"]].assign(s=np.asarray(score, float))
    pos = df[df.is_pos == 1].set_index("group").s
    neg = df[df.is_pos == 0]
    above = (neg.s.values > pos.reindex(neg.group).values)
    tied = (neg.s.values == pos.reindex(neg.group).values)
    g = pd.Series(above).groupby(neg.group.values).sum().reindex(pos.index, fill_value=0).values
    t = pd.Series(tied).groupby(neg.group.values).sum().reindex(pos.index, fill_value=0).values
    hit, ndcg = np.zeros(len(pos)), np.zeros(len(pos))
    for j, (gj, tj) in enumerate(zip(g, t)):
        ranks = np.arange(gj + 1, gj + tj + 2)              # the positive is equally likely at each tied rank
        top = ranks[ranks <= k]
        hit[j] = len(top) / len(ranks)
        ndcg[j] = (1 / np.log2(top + 1)).sum() / len(ranks)
    return pd.DataFrame({"group": pos.index, f"Hit@{k}": hit, f"NDCG@{k}": ndcg})


def mean_ci(values, users, n_boot=N_BOOT, seed=0):
    """Mean of per-row values with a 95% user-bootstrap CI."""
    values = np.asarray(values, float)
    boots = _bootstrap(values, [values], users, lambda y, s: s.mean(), n_boot, seed)[:, 0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return values.mean(), lo, hi

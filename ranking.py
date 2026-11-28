"""Ranking evaluation: can a model pick, for one user, the new recipe they loved?

Each list = one cold review rated 5 stars (the positive) + `n_neg` new recipes the user never
reviewed (negatives). A negative is a recipe with no review in the history that was already
submitted on the positive's date. Models score all 100 recipes; Hit@k / NDCG@k measure where
the positive lands. Ties are counted at their expected position, so a score that is constant
within a list (B0, B1) gets exactly the random value (Hit@10 = 0.10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_candidates(positives: pd.DataFrame, history: pd.DataFrame, recipes: pd.DataFrame,
                      reviewed: dict, n_lists: int = 1000, n_neg: int = 99, seed: int = 0) -> pd.DataFrame:
    """Candidate rows (list_id, user_id, u, recipe_id, date, is_pos) for up to `n_lists` positives.
    `reviewed` maps user_id -> every recipe the user ever reviewed (never used as a negative)."""
    rng = np.random.default_rng(seed)
    pos = positives.sample(n=min(n_lists, len(positives)), random_state=seed).reset_index(drop=True)
    pool = recipes.index[~recipes.index.isin(history.recipe_id)]
    submitted = recipes.submitted.reindex(pool).values
    order = np.argsort(submitted)
    pool_ids, pool_sub = pool.values[order], submitted[order]

    rows = []
    for lid, r in enumerate(pos.itertuples(index=False)):
        n_ok = np.searchsorted(pool_sub, np.datetime64(r.date), side="right")
        seen = reviewed.get(r.user_id, set())
        negs = []
        while len(negs) < n_neg:
            for c in pool_ids[rng.integers(0, n_ok, size=2 * n_neg)]:
                if c != r.recipe_id and c not in seen and c not in negs:
                    negs.append(c)
                    if len(negs) == n_neg:
                        break
        for c, is_pos in [(r.recipe_id, 1)] + [(c, 0) for c in negs]:
            rows.append((lid, r.user_id, r.u, c, r.date, is_pos))
    return pd.DataFrame(rows, columns=["list_id", "user_id", "u", "recipe_id", "date", "is_pos"])


def ranking_metrics(cands: pd.DataFrame, scores, k: int = 10) -> pd.DataFrame:
    """Per-list Hit@k and NDCG@k (expected values under random tie-breaking)."""
    s = np.asarray(scores, dtype=float)
    lid, is_pos = cands.list_id.values, cands.is_pos.values == 1
    n_lists = lid.max() + 1
    pos_score = np.full(n_lists, np.nan)
    pos_score[lid[is_pos]] = s[is_pos]
    neg = ~is_pos
    ps = pos_score[lid[neg]]
    above = np.bincount(lid[neg], weights=(s[neg] > ps), minlength=n_lists)
    tied = np.bincount(lid[neg], weights=(s[neg] == ps), minlength=n_lists)
    # the positive sits uniformly at positions above+1 .. above+tied+1 (1-based)
    first, last = above + 1, above + tied + 1
    hit = np.clip((np.minimum(last, k) - first + 1) / (tied + 1), 0, 1)
    disc = np.r_[0, np.cumsum(1 / np.log2(np.arange(2, k + 2)))]        # disc[j] = sum_{p<=j} 1/log2(p+1)
    ndcg = (disc[np.minimum(last, k).astype(int)] - disc[np.minimum(first - 1, k).astype(int)]) / (tied + 1)
    users = cands.loc[is_pos].sort_values("list_id").user_id.values
    return pd.DataFrame({f"Hit@{k}": hit, f"NDCG@{k}": ndcg, "user_id": users})

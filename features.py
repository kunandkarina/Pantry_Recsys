"""Data loading, recipe sets, snapshots and features for Pantry_Recsys (see SPEC.md).

Snapshot = (history, targets). Every feature of a target row is computed from `history` only,
and every history row is dated before every target row:

    LR training  : history = train before Jan 1 of year Y, targets = train rows in year Y
    validation   : history = train,                     targets = validation
    test         : history = train + validation,        targets = test

This mirrors the cold-start evaluation exactly and never lets a row see its own label or the future.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import MultiLabelBinarizer

DATA = Path(__file__).resolve().parent / "data_clean"
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
                  "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"]
LABELS = ("eq5", "ge4")  # y_eq5 = 1[rating == 5] (main), y_ge4 = 1[rating >= 4] (sensitivity)


# ----------------------------------------------------------------------------- loading
def load_interactions(data_dir: Path = DATA):
    """Return (train, validation, test) with label columns y_eq5 and y_ge4."""
    out = []
    for name in ("train", "validation", "test"):
        df = pd.read_csv(Path(data_dir) / f"interactions_{name}.csv", parse_dates=["date"])
        df["y_eq5"] = (df.rating == 5).astype(np.int8)
        df["y_ge4"] = (df.rating >= 4).astype(np.int8)
        out.append(df)
    return tuple(out)


def load_recipes(data_dir: Path = DATA) -> pd.DataFrame:
    """Recipes indexed by recipe id; tags and ingredients parsed into lists."""
    cols = ["id", "minutes", "contributor_id", "submitted", "tags", "n_steps", "description",
            "ingredients", "n_ingredients", "minutes_missing", "nutrition_extreme", *NUTRITION_COLS]
    r = pd.read_csv(Path(data_dir) / "RAW_recipes_clean.csv", usecols=cols, parse_dates=["submitted"])
    r["description"] = r.description.fillna("")
    r["tags"] = r.tags.map(ast.literal_eval)
    r["ingredients"] = r.ingredients.map(lambda s: [x.strip().lower() for x in ast.literal_eval(s)])
    return r.set_index("id")


# ----------------------------------------------------------------------------- recipe sets
def _binarize(lists):
    mlb = MultiLabelBinarizer(sparse_output=True)
    mat = mlb.fit_transform(lists).tocsr().astype(np.float32)
    return np.asarray(mlb.classes_), mat


class RecipeSets:
    """0/1 recipe x token matrices for tags and ingredients."""

    def __init__(self, recipes: pd.DataFrame):
        self.ids = pd.Index(recipes.index)
        self.tag_vocab, self._tags = _binarize(recipes.tags)
        self.ing_vocab, self.ing = _binarize(recipes.ingredients)
        self._cache = {}

    def pos(self, recipe_ids) -> np.ndarray:
        p = self.ids.get_indexer(recipe_ids)
        if (p < 0).any():
            raise KeyError("recipe id not in RAW_recipes_clean")
        return p

    def tag_doc_freq(self, fit_recipe_ids) -> pd.Series:
        """Share of fit recipes that carry each tag."""
        rows = self.pos(np.unique(fit_recipe_ids))
        return pd.Series(np.asarray(self._tags[rows].mean(axis=0)).ravel(), index=self.tag_vocab)

    def tags(self, max_df: float, fit_recipe_ids) -> sp.csr_matrix:
        """Tag matrix without tags carried by more than `max_df` of the fit recipes."""
        if max_df >= 1:
            return self._tags
        fit = np.unique(fit_recipe_ids)
        key = (max_df, len(fit), int(fit.sum()))
        if key not in self._cache:
            keep = np.flatnonzero(self.tag_doc_freq(fit).values <= max_df)
            self._cache[key] = self._tags[:, keep].tocsr()
        return self._cache[key]


def pair_overlap(mat: sp.csr_matrix, a_pos, b_pos, chunk: int = 500_000):
    """Jaccard and cosine between rows a_pos[k] and b_pos[k] of a 0/1 matrix."""
    sizes = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
    inter = np.empty(len(a_pos), dtype=np.float32)
    for s in range(0, len(a_pos), chunk):
        e = s + chunk
        inter[s:e] = np.asarray(mat[a_pos[s:e]].multiply(mat[b_pos[s:e]]).sum(axis=1)).ravel()
    na, nb = sizes[a_pos], sizes[b_pos]
    union, geo = na + nb - inter, np.sqrt(na * nb)
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    cos = np.divide(inter, geo, out=np.zeros_like(inter), where=geo > 0)
    return jac, cos


# ----------------------------------------------------------------------------- snapshots
def topk_rank(row: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Rank of each pair within its row by similarity (0 = most similar)."""
    order = np.lexsort((-sim, row))
    r_sorted = row[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(r_sorted)])
    rank = np.empty(len(sim), dtype=np.int32)
    rank[order] = np.arange(len(sim)) - np.repeat(starts, counts)
    return rank


@dataclass
class Snapshot:
    name: str
    history: pd.DataFrame   # interactions the features may use
    targets: pd.DataFrame   # rows to predict; column `row` = 0..n-1
    pairs: pd.DataFrame     # (target row, history recipe) for the user's H most recent history recipes
    max_df: float = 1.0
    _ranks: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.targets)

    def rank(self, sim_col: str) -> np.ndarray:
        if sim_col not in self._ranks:
            self._ranks[sim_col] = topk_rank(self.pairs.row.values, self.pairs[sim_col].values)
        return self._ranks[sim_col]

    def compute_similarity(self, rsets: RecipeSets, max_df: float):
        p = self.pairs
        tpos, hpos = rsets.pos(p.recipe_id.values), rsets.pos(p.recipe_id_h.values)
        tags = rsets.tags(max_df, self.history.recipe_id.values)
        p["jac_tag"], p["cos_tag"] = pair_overlap(tags, tpos, hpos)
        if "jac_ing" not in p:
            p["jac_ing"], p["cos_ing"] = pair_overlap(rsets.ing, tpos, hpos)
        self.max_df = max_df
        self._ranks.clear()


def make_snapshot(name, history, targets, rsets: RecipeSets, H: int = 100,
                  max_df: float = 0.5, min_hist: int = 3, cold_only: bool = False) -> Snapshot:
    """Build a snapshot. Targets whose user has fewer than `min_hist` history rows are dropped;
    with `cold_only`, so are targets whose recipe appears in the history (pseudo-cold rows)."""
    if targets.date.min() <= history.date.max():
        raise ValueError("history must end before the first target")
    counts = history.u.value_counts()
    t = targets[targets.u.map(counts).fillna(0).values >= min_hist].copy().reset_index(drop=True)
    t["row"] = np.arange(len(t))
    t["cold_snap"] = (~t.recipe_id.isin(history.recipe_id)).astype(np.int8)
    if cold_only:
        t = t[t.cold_snap == 1].reset_index(drop=True)
        t["row"] = np.arange(len(t))

    h = history.sort_values(["u", "date", "recipe_id"], ascending=[True, False, True])
    h = h.assign(hrank=h.groupby("u").cumcount())
    h = h[h.hrank < H][["u", "recipe_id", "date", "rating", "y_eq5", "y_ge4", "hrank"]]
    p = t[["row", "u", "date", "recipe_id"]].merge(h, on="u", suffixes=("", "_h"))
    p = p.sort_values(["row", "hrank"]).reset_index(drop=True)
    p["dt"] = ((p.date - p.date_h).dt.days / 365.25).astype(np.float32)  # years between the two reviews
    p = p.drop(columns=["u", "date", "date_h"]).astype(
        {"row": np.int32, "recipe_id": np.int32, "recipe_id_h": np.int32, "rating": np.int8,
         "y_eq5": np.int8, "y_ge4": np.int8, "hrank": np.int16})

    snap = Snapshot(name, history, t, p)
    snap.compute_similarity(rsets, max_df)
    return snap


def year_snapshot(train, year: int, rsets: RecipeSets, cold_only: bool = False, **kw) -> Snapshot:
    """LR training snapshot: history = train before `year`, targets = train rows in `year`
    (only pseudo-cold rows, whose recipe has no review before `year`, when `cold_only`)."""
    cut = pd.Timestamp(f"{year}-01-01")
    return make_snapshot(f"train{year}{'-cold' if cold_only else ''}", train[train.date < cut],
                         train[train.date.dt.year == year], rsets, cold_only=cold_only, **kw)


# ----------------------------------------------------------------------------- helpers
def _sum(row, val, n, mask=None):
    if mask is not None:
        row, val = row[mask], (val[mask] if np.ndim(val) else val)
    w = None if np.ndim(val) == 0 else val
    return np.bincount(row, weights=w, minlength=n).astype(float)


def _group_mean(row, val, n, mask=None):
    num, den = _sum(row, val, n, mask), _sum(row, 1.0, n, mask)
    return np.divide(num, den, out=np.zeros(n), where=den > 0)


def _group_max(row, val, n, mask=None):
    out = np.zeros(n)
    if mask is not None:
        row, val = row[mask], val[mask]
    np.maximum.at(out, row, val)
    return out


def global_rate(snap: Snapshot, label: str) -> float:
    return float(snap.history[f"y_{label}"].mean())


def user_prior(snap: Snapshot, label: str, beta: float) -> np.ndarray:
    """p_u: the user's shrunk positive rate over the whole history."""
    g = snap.history.groupby("u")[f"y_{label}"].agg(["sum", "count"])
    s = snap.targets.u.map(g["sum"]).fillna(0).values
    c = snap.targets.u.map(g["count"]).fillna(0).values
    pbar = global_rate(snap, label)
    return (s + beta * pbar) / (c + beta)


def weighted_average(snap: Snapshot, sim_col: str, label: str, prior: np.ndarray,
                     k: int, alpha: float, lam: float) -> np.ndarray:
    """SPEC 5.3: similarity-weighted share of positives among the k most similar history recipes,
    with time decay exp(-lam * years) and shrinkage alpha toward the user's prior."""
    p = snap.pairs
    m = snap.rank(sim_col) < k
    row = p.row.values[m]
    w = p[sim_col].values[m].astype(float)
    if lam:
        w = w * np.exp(-lam * p.dt.values[m])
    y = p[f"y_{label}"].values[m]
    num = np.bincount(row, weights=w * y, minlength=snap.n)
    den = np.bincount(row, weights=w, minlength=snap.n)
    return (num + alpha * prior) / (den + alpha)


# ----------------------------------------------------------------------------- features
FEATURE_GROUPS = {
    "user": ("u_",),
    "sim": ("wa_", "jac_", "has_disliked", "m_"),
    "recipe": ("r_",),
    "author": ("a_", "ua_", "is_own_recipe"),
}


def group_columns(columns, groups) -> list:
    prefixes = tuple(p for g in groups for p in FEATURE_GROUPS[g])
    return [c for c in columns if c.startswith(prefixes)]


def build_features(snap: Snapshot, recipes: pd.DataFrame, label: str, beta: float,
                   wa_params: dict, lam_user: float) -> pd.DataFrame:
    """Feature matrix for every target row of `snap` (columns grouped by FEATURE_GROUPS)."""
    t, p, n = snap.targets, snap.pairs, snap.n
    row = p.row.values
    ycol = f"y_{label}"
    y = p[ycol].values.astype(float)
    hist = snap.history
    pbar = global_rate(snap, label)
    F = {}

    # --- user (whole history)
    g = hist.groupby("u").agg(n=("rating", "size"), pos=(ycol, "sum"), mean=("rating", "mean"),
                              std=("rating", "std"), last=("date", "max")).reindex(t.u.values)
    cnt = g["n"].fillna(0).values
    prior = (g["pos"].fillna(0).values + beta * pbar) / (cnt + beta)
    F["u_log_n"] = np.log1p(cnt)
    F["u_p"] = prior
    F["u_mean_rating"] = g["mean"].fillna(hist.rating.mean()).values
    F["u_std_rating"] = g["std"].fillna(0).values
    days = (t.date.values - g["last"].values) / np.timedelta64(1, "D")
    F["u_log_days_since_last"] = np.log1p(np.nan_to_num(days, nan=3650).clip(0))
    # --- user (time-weighted, from the H most recent history rows)
    w = np.exp(-lam_user * p.dt.values.astype(float))
    F["u_p_decay"] = (_sum(row, w * y, n) + beta * pbar) / (_sum(row, w, n) + beta)
    last_year = p.dt.values < 1
    F["u_p_last_year"] = (_sum(row, y, n, last_year) + beta * pbar) / (_sum(row, 1.0, n, last_year) + beta)

    # --- similarity: weighted averages relative to the user's prior. The raw scores are mostly
    # the user's own rate again; wa - p_u keeps only how this recipe differs from the user's usual.
    for name, prm in wa_params.items():
        F[f"wa_{name}_dev"] = weighted_average(snap, prm["sim"], label, prior,
                                               prm["k"], prm["alpha"], prm["lam"]) - prior
    for s in ("tag", "ing"):
        j = p[f"jac_{s}"].values.astype(float)
        for tag, mask in (("liked", y == 1), ("disliked", y == 0)):
            F[f"jac_{s}_mean_{tag}"] = _group_mean(row, j, n, mask)
            F[f"jac_{s}_max_{tag}"] = _group_max(row, j, n, mask)
        F[f"jac_{s}_mean_recent"] = _group_mean(row, j, n, p.hrank.values < 50)
    F["has_disliked"] = (_sum(row, 1.0, n, y == 0) > 0).astype(float)

    # --- recipe content
    r = recipes.reindex(t.recipe_id.values)
    F["r_log_minutes"] = np.log1p(r.minutes.values)
    for c in NUTRITION_COLS:
        F[f"r_log_{c}"] = np.log1p(r[c].values)
    F["r_n_steps"] = r.n_steps.values.astype(float)
    F["r_n_ingredients"] = r.n_ingredients.values.astype(float)
    F["r_n_tags"] = r.tags.map(len).values.astype(float)
    F["r_has_description"] = (r.description.str.len().values > 0).astype(float)
    F["r_log_desc_len"] = np.log1p(r.description.str.len().values)
    F["r_minutes_missing"] = r.minutes_missing.values.astype(float)
    F["r_nutrition_extreme"] = r.nutrition_extreme.values.astype(float)
    F["r_submitted_year"] = r.submitted.dt.year.values - 2008.0
    age = (t.date.values - r.submitted.values) / np.timedelta64(1, "D")
    F["r_log_age_days"] = np.log1p(np.nan_to_num(age, nan=0).clip(0))

    # --- match: new recipe vs the user's usual recipes
    hrec = recipes[["minutes", "calories"]].reindex(p.recipe_id_h.values)
    F["m_minutes_gap"] = np.abs(F["r_log_minutes"] - _group_mean(row, np.log1p(hrec.minutes.values), n))
    F["m_calories_gap"] = np.abs(np.log1p(r.calories.values)
                                 - _group_mean(row, np.log1p(hrec.calories.values), n))

    # --- author (excluding the target recipe's own reviews, so warm rows match cold rows)
    hc = hist[["u", "recipe_id", ycol]].assign(contributor_id=recipes.contributor_id.reindex(hist.recipe_id).values)
    a = hc.groupby("contributor_id").agg(n=(ycol, "size"), pos=(ycol, "sum"), nrec=("recipe_id", "nunique"))
    rr = hc.groupby("recipe_id").agg(n=(ycol, "size"), pos=(ycol, "sum")).reindex(t.recipe_id.values)
    contrib = r.contributor_id.values
    ac = a.reindex(contrib)
    rn, rp = rr["n"].fillna(0).values, rr["pos"].fillna(0).values
    an = ac["n"].fillna(0).values - rn
    ap = ac["pos"].fillna(0).values - rp
    anrec = ac["nrec"].fillna(0).values - (rn > 0)
    F["a_p"] = (ap + beta * pbar) / (an + beta)
    F["a_log_n_reviews"] = np.log1p(an)
    F["a_log_n_recipes"] = np.log1p(anrec)
    ua = hc.groupby(["u", "contributor_id"]).agg(n=(ycol, "size"), pos=(ycol, "sum"))
    ua = ua.reindex(pd.MultiIndex.from_arrays([t.u.values, contrib]))
    uan = ua["n"].fillna(0).values
    F["ua_p"] = (ua["pos"].fillna(0).values + beta * prior) / (uan + beta)
    F["ua_log_n"] = np.log1p(uan)
    F["is_own_recipe"] = (t.user_id.values == contrib).astype(float)

    X = pd.DataFrame(F)
    if not np.isfinite(X.values).all():
        bad = X.columns[~np.isfinite(X.values).all(axis=0)].tolist()
        raise ValueError(f"non-finite feature values in {bad}")
    return X

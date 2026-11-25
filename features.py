"""
Feature construction for cold-start recipe scoring.

Two targets share these features: the rating itself (1-5, main task) and y = rating == 5 (binary
version). Features that summarize past ratings come in both forms: 5-star share (*_rate, knn{K})
and mean rating (*_mean, knn{K}_rating).

Every history-based feature of a row (user, recipe, date) is computed from a `history` table using
only reviews dated strictly before that row's date, so the same code serves all three cases:
  - pseudo-cold training rows: history = train (or train + validation for the final model)
  - validation rows: history = train
  - test rows: history = train + validation

Feature groups
  A. recipe content (looked up by recipe; no history needed)       recipe_features()
  B. user history: count, mean (raw and smoothed), std, share of 5 stars   history_features()  u_*
  C. user x recipe match over tags / ingredients / techniques        history_features()  m_*
     C relative to the user (k-NN minus the user's average, TF-IDF liked - all)  relative_features()  *_dev, t_cos_diff
  D. author history over their other recipes                         author_features()   a_*
  T. TF-IDF cosine of the recipe text to the user's past recipes     tfidf_features()    t_*
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data_clean"
RAW = ROOT / "dataset" if (ROOT / "dataset").is_dir() else ROOT.parent / "Assignment2" / "dataset_old"

N_TAGS = 300          # most common tags kept as multi-hot columns / match vocabulary
N_INGREDIENTS = 300   # most common ingredients kept for matching
KNN_KS = (5, 10, 20, 50)
SMOOTH_M = 10         # pseudo-count pulling small-sample 5-star shares towards the global share
MIN_PRIOR = 3         # training rows need at least this many earlier ratings by the user
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
                  "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"]


# ---------- loading ----------
def load_data():
    """Cleaned splits and the recipe table (indexed by recipe id, list columns parsed)."""
    splits = {name: pd.read_csv(CLEAN / f"interactions_{name}.csv", parse_dates=["date"])
              for name in ("train", "validation", "test")}
    recipes = pd.read_csv(CLEAN / "RAW_recipes_clean.csv", parse_dates=["submitted"])
    recipes["description"] = recipes.description.fillna("")
    pp = pd.read_csv(RAW / "PP_recipes.csv", usecols=["id", "techniques", "calorie_level"])
    recipes = recipes.merge(pp, on="id").set_index("id")
    for col in ("tags", "ingredients", "techniques"):
        recipes[col] = recipes[col].map(ast.literal_eval)
    return splits, recipes


# ---------- training rows ----------
def pseudo_cold_rows(history):
    """Reviews written on the first day their recipe was reviewed: at that time the recipe had no reviews."""
    first_day = history.groupby("recipe_id").date.transform("min")
    rows = history[history.date == first_day].copy()
    rows["y"] = (rows.rating == 5).astype(int)
    return rows


def training_weights(users):
    """1 / sqrt(rows per user), so heavy users do not dominate training."""
    counts = pd.Series(users).map(pd.Series(users).value_counts())
    return 1 / np.sqrt(counts.values)


# ---------- A. recipe content ----------
def top_vocab(lists, n):
    return pd.Series([x for xs in lists for x in xs]).value_counts().index[:n].tolist()


def multi_hot(lists, vocab):
    """Sparse 0/1 matrix (len(lists) x len(vocab)); items outside the vocabulary are ignored."""
    col = {v: j for j, v in enumerate(vocab)}
    rows, cols = [], []
    for i, xs in enumerate(lists):
        for j in {col[x] for x in xs if x in col}:
            rows.append(i)
            cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lists), len(vocab)))


def recipe_features(recipes, tag_vocab):
    """Group A: one row per recipe."""
    f = pd.DataFrame(index=recipes.index)
    f["log_minutes"] = np.log1p(recipes.minutes)
    for c in ["minutes_missing", "minutes_clipped", "n_steps", "n_ingredients",
              "nutrition_extreme", "calories_zero", "has_description"]:
        f[c] = recipes[c]
    for c in NUTRITION_COLS:
        f[f"log_{c}"] = np.log1p(recipes[c])
    for level in (0, 1, 2):
        f[f"calorie_level_{level}"] = (recipes.calorie_level == level).astype(int)
    f["submitted_year"] = recipes.submitted.dt.year
    f["description_words"] = recipes.description.str.split().str.len()
    tags = pd.DataFrame(multi_hot(recipes.tags.tolist(), tag_vocab).toarray().astype(np.uint8),
                        index=recipes.index, columns=[f"tag_{t}" for t in tag_vocab])
    return pd.concat([f, tags], axis=1)


def content_matrix(recipes, tag_vocab, ingredient_vocab):
    """Binary recipe x (tags | ingredients | techniques) matrix and the column slice of each space."""
    blocks = [multi_hot(recipes.tags.tolist(), tag_vocab),
              multi_hot(recipes.ingredients.tolist(), ingredient_vocab),
              sparse.csr_matrix(np.array(recipes.techniques.tolist(), dtype=np.float32))]
    edges = np.cumsum([0] + [b.shape[1] for b in blocks])
    spaces = {"tags": slice(edges[0], edges[1]), "ingr": slice(edges[1], edges[2]),
              "tech": slice(edges[2], edges[3]), "all": slice(0, edges[3])}
    return sparse.hstack(blocks).tocsr(), spaces


# ---------- B + C. user history and user x recipe match ----------
def _cumsum0(a):
    """Cumulative sums with a leading zero row: _cumsum0(a)[k] = sum of the first k rows."""
    return np.concatenate([np.zeros((1,) + a.shape[1:], dtype=a.dtype), np.cumsum(a, axis=0)])


def _cosine(a, b):
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(1), norm, out=np.full(len(a), np.nan), where=norm > 0)


def _jaccard(a, b):
    """Row-wise Jaccard of the non-zero sets of a and b."""
    a, b = a > 0, b > 0
    inter = (a & b).sum(1)
    union = a.sum(1) + b.sum(1) - inter
    return np.divide(inter, union, out=np.full(len(a), np.nan), where=union > 0)


def history_features(rows, history, recipes, content):
    """Groups B and C for each row, from the user's reviews in `history` strictly before the row's date."""
    M, spaces = content
    pos = pd.Series(np.arange(len(recipes)), index=recipes.index)
    log_minutes = np.log1p(recipes.minutes.values)
    log_calories = np.log1p(recipes.calories.values)

    h = history.sort_values(["user_id", "date"])
    h_date, h_pos = h.date.values, pos[h.recipe_id].values
    h_y, h_rating = (h.rating.values == 5).astype(np.float32), h.rating.values.astype(np.float32)
    p0, mu0 = h_y.mean(), h_rating.mean()
    users, starts, counts = np.unique(h.user_id.values, return_index=True, return_counts=True)
    bounds = dict(zip(users, zip(starts, starts + counts)))

    r_date, r_pos = rows.date.values, pos[rows.recipe_id].values
    out = {}
    def put(name, idx, values):
        out.setdefault(name, np.full(len(rows), np.nan, dtype=np.float32))[idx] = values

    for user, idx in rows.groupby("user_id").indices.items():
        s, e = bounds.get(user, (0, 0))
        k = np.searchsorted(h_date[s:e], r_date[idx], side="left")   # reviews strictly before each row
        y, rating = h_y[s:e], h_rating[s:e]

        # B. user history
        n, n5 = k.astype(np.float32), _cumsum0(y)[k]
        mean = np.divide(_cumsum0(rating)[k], n, out=np.full(len(k), np.nan, dtype=np.float32), where=n > 0)
        sq = np.divide(_cumsum0(rating ** 2)[k], n, out=np.full(len(k), np.nan, dtype=np.float32), where=n > 0)
        put("u_n", idx, n)
        put("u_rate", idx, np.divide(n5, n, out=np.full(len(k), np.nan, dtype=np.float32), where=n > 0))
        put("u_rate_smooth", idx, (n5 + SMOOTH_M * p0) / (n + SMOOTH_M))
        put("u_mean", idx, mean)
        put("u_mean_smooth", idx, (_cumsum0(rating)[k] + SMOOTH_M * mu0) / (n + SMOOTH_M))
        put("u_std", idx, np.sqrt(np.maximum(sq - mean ** 2, 0)))

        # C. numeric preference: distance to the user's average past recipe
        lm, lc = log_minutes[h_pos[s:e]], log_calories[h_pos[s:e]]
        for name, past, target in [("minutes", lm, log_minutes[r_pos[idx]]), ("calories", lc, log_calories[r_pos[idx]])]:
            avg = np.divide(_cumsum0(past)[k], n, out=np.full(len(k), np.nan), where=n > 0)
            put(f"m_{name}_diff", idx, np.abs(target - avg))

        # C. content match: profile similarity and k-nearest past recipes
        H = M[h_pos[s:e]].toarray()            # user's past recipes (in date order)
        T = M[r_pos[idx]].toarray()            # target recipes
        profile, liked = _cumsum0(H)[k], _cumsum0(H * y[:, None])[k]
        future = np.arange(e - s)[None, :] >= k[:, None]
        for space, sl in spaces.items():
            t, hs = T[:, sl], H[:, sl]
            put(f"m_{space}_cos", idx, _cosine(profile[:, sl], t))
            put(f"m_{space}_cos_liked", idx, _cosine(liked[:, sl], t))
            put(f"m_{space}_jaccard", idx, _jaccard(profile[:, sl], t))

            inter = t @ hs.T
            union = t.sum(1)[:, None] + hs.sum(1)[None, :] - inter
            sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            sim[future] = -1                   # never look at reviews on or after the row's date
            has_past = k > 0
            put(f"m_{space}_max_sim", idx, np.where(has_past, sim.max(1, initial=-1), np.nan))
            order = np.argsort(-sim, axis=1, kind="stable")   # ties: earlier review first
            for K in KNN_KS:
                top = order[:, :K]
                w = np.clip(np.take_along_axis(sim, top, 1), 0, None)
                den = w.sum(1)
                for suffix, target in [("", y), ("_rating", rating)]:
                    put(f"m_{space}_knn{K}{suffix}", idx,
                        np.divide((w * target[top]).sum(1), den, out=np.full(len(k), np.nan), where=den > 0))

    return pd.DataFrame(out, index=rows.index)


# ---------- D. author ----------
def author_features(rows, history, recipes):
    """Group D: the recipe author's other recipes, using reviews strictly before the row's date."""
    author = recipes.contributor_id
    h = history.assign(a=author.reindex(history.recipe_id).values, y=(history.rating == 5).astype(int))
    p0, mu0 = h.y.mean(), h.rating.mean()
    daily = h.groupby(["a", "date"]).agg(n=("y", "size"), n5=("y", "sum"), sr=("rating", "sum")).reset_index()
    daily[["a_n", "a_n5", "a_sr"]] = daily.groupby("a")[["n", "n5", "sr"]].cumsum()

    # recipes the author had submitted before the row's date
    sub = recipes.reset_index()[["contributor_id", "submitted"]].rename(columns={"contributor_id": "a", "submitted": "date"})
    sub = sub.sort_values("date")
    sub["a_recipes"] = sub.groupby("a").cumcount() + 1

    left = pd.DataFrame({"date": rows.date.values, "a": author.reindex(rows.recipe_id).values,
                         "_row": np.arange(len(rows))}).sort_values("date")
    left = pd.merge_asof(left, daily[["a", "date", "a_n", "a_n5", "a_sr"]].sort_values("date"),
                         on="date", by="a", allow_exact_matches=False)
    left = pd.merge_asof(left, sub, on="date", by="a", allow_exact_matches=False)
    left = left.sort_values("_row")
    a_n, a_n5, a_sr = (left[c].fillna(0).values for c in ("a_n", "a_n5", "a_sr"))
    return pd.DataFrame({
        "a_n": a_n,
        "a_rate_smooth": (a_n5 + SMOOTH_M * p0) / (a_n + SMOOTH_M),
        "a_mean_smooth": (a_sr + SMOOTH_M * mu0) / (a_n + SMOOTH_M),
        "a_recipes": left.a_recipes.fillna(0).values,
    }, index=rows.index)


# ---------- T. TF-IDF text similarity ----------
def tfidf_matrix(recipes, min_df=5, max_features=50000):
    """L2-normalized TF-IDF of each recipe's name, tags, ingredients, description and steps."""
    text = (recipes.name.fillna("") + " " + recipes.tags.str.join(" ") + " " + recipes.ingredients.str.join(" ")
            + " " + recipes.description + " " + recipes.steps.fillna(""))
    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features, stop_words="english",
                                 sublinear_tf=True, dtype=np.float32)
    return vectorizer.fit_transform(text.values)


def _prefix_norm2(H, block=1024):
    """||sum of the first k rows of H||^2 for k = 0..n, from the Gram matrix computed in row blocks."""
    n = H.shape[0]
    cross = np.zeros(n)                        # cross[j] = H_j . (H_0 + ... + H_{j-1})
    diag = np.asarray(H.multiply(H).sum(1)).ravel()
    for start in range(0, n, block):
        G = (H[start:start + block] @ H[:start + block].T).toarray()
        cross[start:start + block] = np.tril(G, k=start - 1).sum(1)   # columns before each row
    return np.concatenate([[0], np.cumsum(2 * cross + diag)])


def tfidf_features(rows, history, recipes, X):
    """Cosine between each row's recipe and the TF-IDF centroid of the user's earlier recipes:
    all of them (t_cos_all), the 5-star ones (t_cos_liked), and the others (t_cos_other)."""
    pos = pd.Series(np.arange(len(recipes)), index=recipes.index)
    h = history.sort_values(["user_id", "date"])
    h_date, h_pos, h_y = h.date.values, pos[h.recipe_id].values, (h.rating.values == 5)
    users, starts, counts = np.unique(h.user_id.values, return_index=True, return_counts=True)
    bounds = dict(zip(users, zip(starts, starts + counts)))
    r_date, r_pos = rows.date.values, pos[rows.recipe_id].values
    t_norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())

    out = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in ("t_cos_all", "t_cos_liked", "t_cos_other")}
    for user, idx in rows.groupby("user_id").indices.items():
        s, e = bounds.get(user, (0, 0))
        if e == s:
            continue
        k = np.searchsorted(h_date[s:e], r_date[idx], side="left")   # reviews strictly before each row
        H = X[h_pos[s:e]]
        S = (X[r_pos[idx]] @ H.T).toarray()                          # target . each past recipe
        S[np.arange(e - s)[None, :] >= k[:, None]] = 0               # never look at reviews on or after the row's date
        y = h_y[s:e]
        for name, mask in [("t_cos_all", np.ones(e - s, bool)), ("t_cos_liked", y), ("t_cos_other", ~y)]:
            dot = S[:, mask].sum(1)
            norm = np.sqrt(_prefix_norm2(H[np.where(mask)[0]])[np.cumsum(np.concatenate([[0], mask]))[k]])
            denom = norm * t_norm[r_pos[idx]]
            out[name][idx] = np.divide(dot, denom, out=np.full(len(idx), np.nan), where=denom > 0)
    return pd.DataFrame(out, index=rows.index)


def relative_features(f):
    """Group C relative to the user. The raw k-NN features are almost the user's own average (Spearman 0.97
    between m_tags_knn50_rating and u_mean), so they repeat group B. Subtracting the user's average keeps only
    how this kind of recipe differs from what the user usually gives:
      m_<space>_knn<K>_rating_dev = k-NN mean rating   - u_mean
      m_<space>_knn<K>_dev        = k-NN 5-star share  - u_rate
      t_cos_diff                  = t_cos_liked - t_cos_all  (closer to loved recipes than to the usual ones)"""
    out = {}
    for c in f.columns:
        m = re.fullmatch(r"(m_\w+_knn\d+)(_rating)?", c)
        if m:
            out[c + "_dev"] = f[c] - (f.u_mean if m.group(2) else f.u_rate)
    if "t_cos_liked" in f:
        out["t_cos_diff"] = f.t_cos_liked - f.t_cos_all
    return pd.DataFrame(out, index=f.index)


def build_features(rows, history, recipes, content, recipe_feats=None, tfidf=None):
    """All feature groups for `rows`, using `history` strictly before each row's date."""
    parts = [history_features(rows, history, recipes, content),
             author_features(rows, history, recipes),
             pd.DataFrame({"recipe_age": (rows.date.values - recipes.submitted.reindex(rows.recipe_id).values)
                           / np.timedelta64(1, "D") / 365.25}, index=rows.index)]
    if tfidf is not None:
        parts.append(tfidf_features(rows, history, recipes, tfidf))
    if recipe_feats is not None:
        parts.append(recipe_feats.reindex(rows.recipe_id).set_index(rows.index))
    f = pd.concat(parts, axis=1)
    return pd.concat([f, relative_features(f)], axis=1)

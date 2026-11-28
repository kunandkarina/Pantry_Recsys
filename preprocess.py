"""
Build a cleaned, strictly time-split copy of the Food.com dataset into data_clean/.

Why not the official interactions_{train,validation,test}.csv: they are not split by time.
For 82.7% of test users the training file contains reviews dated after their test review, so
user-history features would see the future. The files also leave out 30,674 held-out reviews.

Steps
1. Pool every review of a PP recipe from RAW_interactions.csv (one per user-recipe pair, earliest kept)
   and drop rating == 0 rows (review without a rating, so no label).
2. Global time split: train = before VALID_START, validation = [VALID_START, TEST_START),
   test = from TEST_START.
3. Keep users with >= MIN_USER_RATINGS rated reviews before VALID_START. Only the training period
   decides who is kept, so the filter never looks at the future.
4. Label each validation/test row by how many reviews its recipe has in the data a model is fit on
   (train for validation, train + validation for test): n_fit, and cold = 1 when n_fit == 0.
   The item cold-start task evaluates on cold == 1 rows; warm rows are kept for comparison.
5. Clean recipe features (RAW_recipes, PP subset):
   - minutes: 0 -> missing (imputed with train median, flagged); clip at train p99
   - nutrition: split into 7 columns; clip each at train p99; flag extremes
   - description: fill missing with "" and flag
6. Rebuild PP_users from the cleaned train split.

Clipping thresholds are computed on recipes reviewed in the cleaned train split only.
Indices: i is the PP_recipes index. u is renumbered over the kept users (0..n-1), because the
original u only covers PP users, who were selected using reviews from all years.
Original files are never modified.
"""
import ast
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "dataset") if os.path.isdir(os.path.join(ROOT, "dataset")) else ROOT  # raw Kaggle files
DST = os.path.join(ROOT, "data_clean")
VALID_START, TEST_START = pd.Timestamp("2013-01-01"), pd.Timestamp("2014-01-01")
MIN_USER_RATINGS = 3
CLIP_QUANTILE = 0.99
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
                  "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"]

os.makedirs(DST, exist_ok=True)
log = []

# ---------- 1. pool reviews of PP recipes; drop rating == 0 ----------
pp = pd.read_csv(os.path.join(SRC, "PP_recipes.csv"), usecols=["id", "i", "techniques"])
inter = pd.read_csv(os.path.join(SRC, "RAW_interactions.csv"),
                    usecols=["user_id", "recipe_id", "date", "rating"], parse_dates=["date"])
n0 = len(inter)
inter = inter[inter.recipe_id.isin(pp.id)]
inter = inter.sort_values(["date", "user_id", "recipe_id"]).drop_duplicates(["user_id", "recipe_id"])
log.append(f"RAW_interactions: {n0} -> {len(inter)} reviews of PP recipes (one per user-recipe pair)")
n0 = len(inter)
inter = inter[inter.rating > 0]
log.append(f"drop rating==0: {n0} -> {len(inter)} (-{n0 - len(inter)})")

# ---------- 2-3. global time split; users decided by the training period only ----------
counts = inter[inter.date < VALID_START].user_id.value_counts()
users = np.sort(counts[counts >= MIN_USER_RATINGS].index.values)
log.append(f"users with >= {MIN_USER_RATINGS} rated reviews before {VALID_START.date()}: "
           f"{len(users)} of {inter.user_id.nunique()}")
inter = inter[inter.user_id.isin(users)].copy()
inter["u"] = np.searchsorted(users, inter.user_id.values)
inter["i"] = inter.recipe_id.map(pp.set_index("id").i).astype(int)

splits = {"train": inter[inter.date < VALID_START],
          "validation": inter[(inter.date >= VALID_START) & (inter.date < TEST_START)],
          "test": inter[inter.date >= TEST_START]}

# ---------- 4. cold-start labels ----------
fit_for = {"validation": splits["train"], "test": pd.concat([splits["train"], splits["validation"]])}
cols = ["user_id", "recipe_id", "date", "rating", "u", "i"]
for name in list(splits):
    df = splits[name][cols].copy()
    msg = f"interactions_{name}: {len(df)} rows, {df.user_id.nunique()} users, {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}"
    if name in fit_for:
        n_fit = df.i.map(fit_for[name].i.value_counts()).fillna(0).astype(int)
        df["n_fit"], df["cold"] = n_fit, (n_fit == 0).astype(int)
        cold = df[df.cold == 1]
        msg += f"; cold rows {len(cold)} ({cold.user_id.nunique()} users), rating==5 share in cold rows {(cold.rating == 5).mean():.3f}"
    else:
        msg += f"; rating==5 share {(df.rating == 5).mean():.3f}"
    log.append(msg)
    df.to_csv(os.path.join(DST, f"interactions_{name}.csv"), index=False)
    splits[name] = df

# ---------- 5. recipes ----------
raw = pd.read_csv(os.path.join(SRC, "RAW_recipes.csv"))
raw = raw[raw.id.isin(set(pp.id))].copy()
train_recipes = raw.id.isin(set(splits["train"].recipe_id))

# minutes
raw["minutes_missing"] = (raw.minutes == 0).astype(int)
minutes = raw.minutes.where(raw.minutes > 0)
cap = minutes[train_recipes].quantile(CLIP_QUANTILE)
median = minutes[train_recipes].median()
raw["minutes_clipped"] = (minutes > cap).astype(int)
raw["minutes"] = minutes.fillna(median).clip(upper=cap)
log.append(f"minutes: {raw.minutes_missing.sum()} zero -> imputed {median:.0f}; "
           f"{raw.minutes_clipped.sum()} clipped at {cap:.0f}")

# nutrition
nut = pd.DataFrame(raw.nutrition.map(ast.literal_eval).tolist(),
                   columns=NUTRITION_COLS, index=raw.index)
extreme = np.zeros(len(raw), dtype=bool)
for c in NUTRITION_COLS:
    cap = nut.loc[train_recipes, c].quantile(CLIP_QUANTILE)
    over = nut[c] > cap
    extreme |= over.values
    nut[c] = nut[c].clip(upper=cap)
    log.append(f"nutrition.{c}: {over.sum()} clipped at {cap:.1f}")
raw = pd.concat([raw.drop(columns="nutrition"), nut], axis=1)
raw["nutrition_extreme"] = extreme.astype(int)
raw["calories_zero"] = (raw.calories == 0).astype(int)
log.append(f"nutrition_extreme: {extreme.sum()} recipes; calories_zero: {raw.calories_zero.sum()}")

# description
raw["has_description"] = raw.description.notna().astype(int)
raw["description"] = raw.description.fillna("")
log.append(f"description: {(raw.has_description == 0).sum()} missing -> ''")

raw.to_csv(os.path.join(DST, "RAW_recipes_clean.csv"), index=False)
log.append(f"RAW_recipes_clean: {len(raw)} recipes (PP_recipes subset)")

# ---------- 6. PP_users rebuilt from cleaned train ----------
tech = {i: np.array(ast.literal_eval(t)) for i, t in zip(pp.i, pp.techniques)}
rows = []
for u, g in splits["train"].sort_values("date").groupby("u"):
    items = g.i.tolist()
    rows.append({"u": u,
                 "techniques": np.sum([tech[i] for i in items], axis=0).tolist(),
                 "items": items, "n_items": len(items),
                 "ratings": g.rating.tolist(), "n_ratings": len(items)})
pd.DataFrame(rows).to_csv(os.path.join(DST, "PP_users_clean.csv"), index=False)
log.append(f"PP_users_clean: {len(rows)} users")

with open(os.path.join(DST, "cleaning_log.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
print("\n".join(log))

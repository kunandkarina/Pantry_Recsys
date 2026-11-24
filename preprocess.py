"""
Build a cleaned copy of the Food.com dataset into data_clean/.

Steps
1. Drop rating == 0 rows (review without a rating) from train / val / test.
2. Drop users with fewer than MIN_USER_RATINGS rated train interactions
   (from all three splits, since their history is too thin to profile).
3. Clean recipe features (RAW_recipes):
   - minutes: 0 -> missing (imputed with train median, flagged); clip at train p99
   - nutrition: split into 7 columns; clip each at train p99; flag extremes
   - description: fill missing with "" and flag
4. Rebuild PP_users from the cleaned train split (the original includes rating-0 items).

Clipping thresholds are computed on recipes that appear in the cleaned train split only.
Original files are never modified.
"""
import ast
import os

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(SRC, "data_clean")
MIN_USER_RATINGS = 3
CLIP_QUANTILE = 0.99
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
                  "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"]

os.makedirs(DST, exist_ok=True)
log = []

# ---------- 1. interactions: drop rating == 0 ----------
splits = {}
for name in ["train", "validation", "test"]:
    df = pd.read_csv(os.path.join(SRC, f"interactions_{name}.csv"))
    n0 = len(df)
    df = df[df.rating > 0]
    log.append(f"interactions_{name}: drop rating==0 -> {n0} -> {len(df)} (-{n0 - len(df)})")
    splits[name] = df

# ---------- 2. drop users with thin history ----------
counts = splits["train"].user_id.value_counts()
keep_users = set(counts[counts >= MIN_USER_RATINGS].index)
log.append(f"users with < {MIN_USER_RATINGS} rated train interactions: {len(counts) - len(keep_users)} removed")
for name, df in splits.items():
    n0 = len(df)
    df = df[df.user_id.isin(keep_users)]
    log.append(f"interactions_{name}: drop thin users -> {n0} -> {len(df)} (-{n0 - len(df)})")
    df.to_csv(os.path.join(DST, f"interactions_{name}.csv"), index=False)
    splits[name] = df

# ---------- 3. recipes ----------
pp_ids = set(pd.read_csv(os.path.join(SRC, "PP_recipes.csv"), usecols=["id"]).id)
raw = pd.read_csv(os.path.join(SRC, "RAW_recipes.csv"))
raw = raw[raw.id.isin(pp_ids)].copy()
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

# ---------- 4. PP_users rebuilt from cleaned train ----------
techniques = pd.read_csv(os.path.join(SRC, "PP_recipes.csv"), usecols=["i", "techniques"]).set_index("i")
tech = {i: np.array(ast.literal_eval(t)) for i, t in techniques.techniques.items()}
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

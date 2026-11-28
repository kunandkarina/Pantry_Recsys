"""Settings shared by every notebook, and the hyperparameters passed between them.

01_eda -> 02_baselines (BETA, MAX_DF) -> 03_similarity (WA) -> 04_lr (TRAIN_MODE, C, CLASS_WEIGHT, LAM_USER) -> final
Each development notebook saves what it chose to results/params.json; final.ipynb only reads it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
PARAMS_FILE = RESULTS / "params.json"

LABEL = "eq5"                 # main label 1[rating == 5]; "ge4" = 1[rating >= 4] is the sensitivity label
H = 100                       # most recent history recipes per user used for similarities
N_BOOT = int(os.environ.get("N_BOOT", 1000))   # bootstrap resamples for confidence intervals
# How the LR / GBDT training rows are built (04_lr compares them on validation and picks one).
#   all           : every train row of the snapshot years
#   pseudo_cold   : only rows whose recipe had no review before that year (like a cold row)
#   pseudo_cold_w : pseudo_cold, each row weighted 1 / sqrt(rows of its user)
# years_val: snapshots used when evaluating on validation; years_test: snapshots used, together
# with the validation rows (cold rows only for the pseudo-cold modes), when evaluating on test.
TRAIN_MODES = {
    "all":           {"years_val": [2011, 2012], "years_test": [2012], "cold_only": False, "weighted": False},
    "pseudo_cold":   {"years_val": [2009, 2010, 2011, 2012], "years_test": [2010, 2011, 2012],
                      "cold_only": True, "weighted": False},
    "pseudo_cold_w": {"years_val": [2009, 2010, 2011, 2012], "years_test": [2010, 2011, 2012],
                      "cold_only": True, "weighted": True},
}

N_RANK_LISTS = 1000           # ranking lists (one 5-star cold review + N_NEG new recipes each)
N_NEG = 99

BETA_GRID = (1, 2, 5, 10, 20)
MAX_DF_GRID = (0.2, 0.35, 0.5, 0.75, 1.0)
K_GRID = (5, 10, 20, 50, 100)
ALPHA_GRID = (0.5, 1, 2, 5, 10, 20)
LAMBDA_GRID_T = (0.03, 0.1, 0.3, 1.0, 3.0)
C_GRID = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1, 10)
GBDT_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.1, random_state=0)


def load_params(*required) -> dict:
    """Hyperparameters saved by the earlier notebooks; fails clearly if one is missing."""
    params = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    missing = [k for k in required if k not in params]
    if missing:
        raise RuntimeError(f"{missing} not in {PARAMS_FILE}: run the earlier notebooks first "
                           "(02_baselines -> 03_similarity -> 04_lr)")
    return params


def save_params(**values) -> dict:
    """Merge `values` into results/params.json and return the whole file."""
    params = load_params()
    params.update(values)
    PARAMS_FILE.write_text(json.dumps(params, indent=2))
    return params

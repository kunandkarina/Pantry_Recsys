# Pantry_Recsys

Food.com 食譜推薦的 item cold-start 預測：輸入 `(user_id, recipe_id)`，輸出 user 會喜歡這道**新食譜**的機率（0~1）。

- **任務設定**：依時間切分。val/test 的 user 都在 train 出現過；冷啟動評估只用 `cold == 1` 的列，也就是食譜在模型訓練資料中沒有任何評論。
- **主 label**：`rating == 5`；`rating >= 4` 作為敏感度分析。
- **特徵**：由 ID 查表取得，包括 user 歷史、recipe 內容（tags、食材、營養、時間、步驟），以及兩者的 match feature。

## 資料前處理

處理腳本為 `preprocess.py`，從 Kaggle 原始檔（放在 `dataset/` 或專案根目錄）輸出到 `data_clean/`，原始檔案不會被修改。

### 原則
- **嚴格依時間切分**，所有統計量只用切點之前的資料：user 篩選、clip 門檻、user 歷史都不碰未來。
- **處理 feature，不刪資料列**。唯一的例外是 label 有問題（rating = 0）和歷史太少的 user。
- 所有 clip 門檻只用 **train 中出現的 recipe** 計算，避免 leakage。

### 為什麼不用官方的 `interactions_*.csv`
官方切分不是依時間切：82.7% 的 test user，train 裡有日期**晚於**他 test 那筆的評論，所以用 train 算出的 user 歷史會看到未來。官方檔案也少了 30,674 筆被丟掉的留出評論。因此改從 `RAW_interactions.csv` 重新切。

### 時間切分

| split | 期間 | 列數 | user | 冷啟動列（`cold == 1`） |
|---|---|---|---|---|
| train | 2015-01-01 以前 | 725,446 | 28,671 | — |
| validation | 2015 年 | 5,065 | 1,385 | 489（248 個 user） |
| test | 2016-01-01 以後 | 7,408 | 1,557 | 894（335 個 user） |

- `n_fit`：該食譜在模型訓練資料中的評論數。validation 以 train 計算；test 以 train + validation 計算，因為最終模型會用兩者重新訓練。
- `cold = 1` 表示 `n_fit == 0`。warm 列保留，可以當對照。
- 一個 user 在 validation/test 期間的每則評論都是一列，不是每人一列。

> 注意：2014 年以後站上活動很少，冷啟動 test 只有 894 列。若需要更多樣本，可以把 `VALID_START`、`TEST_START` 提前到 2013 / 2014，新食譜樣本約多 4.5 倍。

### 輸出檔案與修改內容

| 新檔案（`data_clean/`） | 來源 | 修改內容 |
|---|---|---|
| `interactions_train.csv` | `RAW_interactions.csv` | PP 食譜的評論（同一 user-recipe 只留最早一筆）→ 刪 rating = 0 → 只留 2015 年以前、且符合 user 條件：**725,446** 列 |
| `interactions_validation.csv` | `RAW_interactions.csv` | 2015 年的評論：**5,065** 列，多了 `n_fit`、`cold` |
| `interactions_test.csv` | `RAW_interactions.csv` | 2016 年以後的評論：**7,408** 列，多了 `n_fit`、`cold` |
| `RAW_recipes_clean.csv` | `RAW_recipes.csv` | 只保留 PP_recipes 的 178,265 道食譜，並修正 minutes、nutrition、缺值 |
| `PP_users_clean.csv` | 由清理後的 train 重建 | 重新計算 `techniques`、`items`、`ratings`、`n_items`、`n_ratings`；共 28,671 個 user |
| `cleaning_log.txt` | — | 每一步的刪除與修正筆數 |

`PP_recipes.csv` 沒有需要修正的錯誤，直接使用原檔。`i` 仍是 PP_recipes 的索引；`u` **重新編號**（0 到 28,670），因為原本的 `u` 只涵蓋 PP user，而 PP user 是用所有年份的評論篩出來的。

### 1. 刪除 rating = 0
在 Food.com 上，rating = 0 代表「留了評論但沒有評分」，不是 0 分，所以沒有 label。PP 食譜的 935,120 則評論中刪除 46,257 則。

### 2. 刪除歷史太少的 user
只看 **2015 年以前**的評論：有效評分 **少於 3 筆**（`MIN_USER_RATINGS = 3`）的 user 全部移除，保留 28,671 人。只用訓練期決定誰留下，所以篩選本身不會用到未來。

### 3. RAW_recipes 欄位修正

| 欄位 | 問題 | 處理 | 新增欄位 |
|---|---|---|---|
| `minutes` | 834 道為 0；最大值 2,147,483,647 | 0 視為缺值，以 train 中位數 38 填補；超過 p99 = 730 分鐘的 1,748 道 clip 到 730 | `minutes_missing`、`minutes_clipped` |
| `nutrition` | 原為字串 list；calories 最大值 434,360 | 拆成 7 欄：`calories`、`total_fat_pdv`、`sugar_pdv`、`sodium_pdv`、`protein_pdv`、`saturated_fat_pdv`、`carbohydrates_pdv`，各欄在 train p99 clip（每欄約 1,800 道） | `nutrition_extreme`（任一欄被 clip，5,528 道）、`calories_zero`（16 道） |
| `description` | 3,954 道缺值 | 填空字串 | `has_description` |

各營養欄位的 clip 門檻（train p99）：

| calories | total_fat_pdv | sugar_pdv | sodium_pdv | protein_pdv | saturated_fat_pdv | carbohydrates_pdv |
|---|---|---|---|---|---|---|
| 3358.5 | 285.0 | 1092.0 | 197.0 | 177.0 | 381.0 | 150.0 |

### 清理後的檢查結果
- 三個 split 都沒有 rating = 0。
- val/test 的 user 在 train 都至少有 3 筆評分，而且全部在 2015 年以前。
- `rating == 5` 的正例比例：train 75.7%；冷啟動列 validation 80.4%、test 79.6%。

### 使用注意
- CSV 中的空字串會被 pandas 讀成 NaN，讀取 `RAW_recipes_clean.csv` 後要再對 `description` 做 `.fillna("")`，或改用 `has_description` 判斷。
- minutes、nutrition 的 log 轉換與標準化留到 feature engineering 階段再做，這份資料只修正錯誤值。
- `n_steps`（3–37）和 `n_ingredients`（4–20）在 PP 版資料中已經過濾過，不需要再處理。
- user 歷史特徵只能用 train（評估 test 時可用 train + validation），不要用 `RAW_interactions.csv` 直接算。

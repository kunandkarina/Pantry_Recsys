# Pantry_Recsys

Food.com 食譜推薦的 item cold-start 預測：輸入 `(user_id, recipe_id)`，輸出 user 會喜歡這道**新食譜**的機率（0~1）。

- **任務設定**：val/test 的 user 都在 train 出現過，recipe 則 **0%** 出現在 train，屬於 item cold-start。
- **主 label**：`rating == 5`；`rating >= 4` 作為敏感度分析。
- **特徵**：由 ID 查表取得，包括 user 歷史、recipe 內容（tags、食材、營養、時間、步驟），以及兩者的 match feature。

## 資料前處理

處理腳本為 `preprocess.py`，輸出到 `data_clean/`，原始檔案不會被修改。

### 原則
- **處理 feature，不刪資料列**。唯一的例外是 label 有問題（rating = 0）和歷史太少的 user。
- 所有 clip 門檻只用 **train 中出現的 recipe** 計算，避免 leakage。
- 不使用 `RAW_interactions.csv` 計算任何統計量，因為 test 的 12,455 筆互動全部都在裡面。

### 輸出檔案與修改內容

| 新檔案（`data_clean/`） | 來源 | 修改內容 |
|---|---|---|
| `interactions_train.csv` | `interactions_train.csv` | 刪 rating = 0：698,901 → 681,944（−16,957）；刪歷史太少的 user：→ **672,419**（−9,525） |
| `interactions_validation.csv` | `interactions_validation.csv` | 刪 rating = 0：7,023 → 6,667（−356）；刪歷史太少的 user：→ **6,203**（−464） |
| `interactions_test.csv` | `interactions_test.csv` | 刪 rating = 0：12,455 → 11,768（−687）；刪歷史太少的 user：→ **10,136**（−1,632） |
| `RAW_recipes_clean.csv` | `RAW_recipes.csv` | 只保留 PP_recipes 的 178,265 道食譜（涵蓋三個 split），並修正 minutes、nutrition、缺值 |
| `PP_users_clean.csv` | 由清理後的 train 重建 | 原檔的 items/ratings 含 rating = 0 的互動，因此重新計算 `techniques`、`items`、`ratings`、`n_items`、`n_ratings`；共 19,940 個 user |
| `cleaning_log.txt` | — | 每一步的刪除與修正筆數 |

`PP_recipes.csv` 沒有需要修正的錯誤，直接使用原檔。`u`、`i` 索引沒有重新編號，仍可對應 PP 檔。

### 1. 刪除 rating = 0
在 Food.com 上，rating = 0 代表「留了評論但沒有評分」，不是 0 分，所以沒有 label。train、validation、test 三個 split 都刪除。

### 2. 刪除歷史太少的 user
清理後 train 中有效評分 **少於 3 筆**（`MIN_USER_RATINGS = 3`）的 user 共 5,021 人，從三個 split 一起移除，因為他們的歷史太少，無法建立可靠的 profile。

> 注意：評估範圍因此變成「train 中至少有 3 筆評分的 user」，test 減少了 1,632 筆。

### 3. RAW_recipes 欄位修正

| 欄位 | 問題 | 處理 | 新增欄位 |
|---|---|---|---|
| `minutes` | 834 道為 0；最大值 2,147,483,647 | 0 視為缺值，以 train 中位數 35 填補；超過 p99 = 720 分鐘的 1,837 道 clip 到 720 | `minutes_missing`、`minutes_clipped` |
| `nutrition` | 原為字串 list；calories 最大值 434,360 | 拆成 7 欄：`calories`、`total_fat_pdv`、`sugar_pdv`、`sodium_pdv`、`protein_pdv`、`saturated_fat_pdv`、`carbohydrates_pdv`，各欄在 train p99 clip（每欄約 1,800–1,900 道） | `nutrition_extreme`（任一欄被 clip，5,715 道）、`calories_zero`（16 道） |
| `description` | 3,954 道缺值 | 填空字串 | `has_description` |

各營養欄位的 clip 門檻（train p99）：

| calories | total_fat_pdv | sugar_pdv | sodium_pdv | protein_pdv | saturated_fat_pdv | carbohydrates_pdv |
|---|---|---|---|---|---|---|
| 3282.4 | 280.9 | 1058.0 | 196.0 | 175.0 | 368.0 | 148.0 |

### 清理後的檢查結果
- 三個 split 都沒有 rating = 0。
- val/test 的 user 在 train 都至少有 3 筆評分。
- val/test 的 recipe 仍然 100% 未出現在 train，cold-start 設定不受影響。
- `rating == 5` 的正例比例：train 76.0%、validation 63.7%、test 64.3%。

### 使用注意
- CSV 中的空字串會被 pandas 讀成 NaN，讀取 `RAW_recipes_clean.csv` 後要再對 `description` 做 `.fillna("")`，或改用 `has_description` 判斷。
- minutes、nutrition 的 log 轉換與標準化留到 feature engineering 階段再做，這份資料只修正錯誤值。
- `n_steps`（3–37）和 `n_ingredients`（4–20）在 PP 版資料中已經過濾過，不需要再處理。

# Pantry_Recsys

Food.com 食譜推薦的 **item cold-start** 預測：輸入 `(user_id, recipe_id)`，輸出 user 會喜歡這道**新食譜**的機率（0~1），即 P(rating == 5)。

- **任務設定**：依時間切分。val/test 的 user 都在 train 期間至少評分過 3 次；評估主要看 `cold == 1` 的列，也就是該食譜在模型可用的資料中**從未被評論過**。
- **主 label**：`rating == 5`；`rating >= 4` 作為敏感度分析。
- **特徵**：由 ID 查表取得，包括 user 歷史、recipe 內容（tags、食材、營養、時間、步驟、作者），以及兩者的 match feature（Jaccard 相似度為主，Cosine 為對照，並加上時間衰減）。

## 資料前處理

處理腳本為 `preprocess.py`。原始 Kaggle 檔案放在 `dataset/`（沒有的話就讀專案根目錄），輸出到 `data_clean/`，原始檔案不會被修改。各步驟的筆數記錄在 `cleaning_log.txt`。

### 為什麼不用官方的 interactions_{train,validation,test}.csv
官方的切分**不是依時間**：82.7% 的 test user 在 train 中有日期晚於其 test 評論的紀錄，user 歷史特徵會看到未來。官方檔案也漏掉 30,674 筆評論。因此改從 `RAW_interactions.csv` 重新依時間切分。

### 步驟
1. **合併評論**：取 `RAW_interactions.csv` 中屬於 PP 食譜的評論，每個 (user, recipe) 只保留最早一筆：1,132,367 → 935,120。
2. **刪除 rating = 0**：在 Food.com 上 rating = 0 代表「留了評論但沒有評分」，沒有 label：935,120 → 888,863（−46,257）。
3. **全域時間切分**：
   - train：2013-01-01 以前
   - validation：2013-01-01 ～ 2013-12-31
   - test：2014-01-01 以後
   - 選 2013／2014 而不是更晚的年份：Food.com 後期評論很少，原本用 2015／2016 切時 cold 列只有 val 489、test 894 筆。
4. **保留 user**：只保留在 **train 期間**有至少 3 筆有效評分（`MIN_USER_RATINGS = 3`）的 user，共 27,327 / 158,047 人。篩選只看 train 期間，不會偷看未來。
5. **cold-start 標記**：val/test 每一列加上
   - `n_fit`：該食譜在模型 fit 用資料中的評論數（validation 用 train；test 用 train + validation）
   - `cold`：`n_fit == 0` 時為 1
6. **清理食譜欄位**（見下表），門檻只用 train 中出現過的食譜計算。
7. **重建 PP_users**：由清理後的 train 重新計算 `techniques`、`items`、`ratings`、`n_items`、`n_ratings`。

### 索引
- `i`：PP_recipes 的 index，沒有重新編號。
- `u`：在保留的 27,327 個 user 上**重新編號**（0..n−1），因為原本的 `u` 只涵蓋 PP user，而 PP user 是用所有年份的評論挑選的。**新的 `u` 不能對應原始 PP 檔。**

### 輸出檔案

| 檔案 | 內容 |
|---|---|
| `interactions_train.csv` | `user_id, recipe_id, date, rating, u, i` |
| `interactions_validation.csv` | 同上 + `n_fit, cold` |
| `interactions_test.csv` | 同上 + `n_fit, cold` |
| `RAW_recipes_clean.csv` | PP_recipes 的 178,265 道食譜，修正 minutes、nutrition、缺值 |
| `PP_users_clean.csv` | 由清理後 train 重建，27,327 個 user |
| `cleaning_log.txt` | 每一步的刪除與修正筆數 |

### 各 split 統計

| split | 期間 | 列數 | user | 食譜 | rating==5 | rating>=4 |
|---|---|---|---|---|---|---|
| train | 2000-02-25 ～ 2012-12-31 | 683,996 | 27,327 | 166,824 | 75.5% | 94.4% |
| validation（全部） | 2013 | 22,933 | 3,346 | 16,138 | 78.1% | 95.3% |
| validation（cold） | 2013 | **6,410** | 859 | 4,727 | 77.6% | 95.7% |
| test（全部） | 2014-01-01 ～ 2018-12-18 | 22,224 | 3,250 | 15,068 | 80.2% | 95.2% |
| test（cold） | 2014–2018 | **4,099** | 857 | 3,268 | 79.6% | 95.9% |

train 的評分分布：1 分 3,917、2 分 7,443、3 分 26,715、4 分 129,323、5 分 516,598。

### RAW_recipes 欄位修正

| 欄位 | 問題 | 處理 | 新增欄位 |
|---|---|---|---|
| `minutes` | 834 道為 0；最大值 2,147,483,647 | 0 視為缺值，以 train 中位數 40 填補；超過 p99 = 730 分鐘的 1,748 道 clip 到 730 | `minutes_missing`、`minutes_clipped` |
| `nutrition` | 原為字串 list；calories 最大值 434,360 | 拆成 7 欄：`calories`、`total_fat_pdv`、`sugar_pdv`、`sodium_pdv`、`protein_pdv`、`saturated_fat_pdv`、`carbohydrates_pdv`，各欄在 train p99 clip（每欄約 1,745–1,790 道） | `nutrition_extreme`（任一欄被 clip，5,451 道）、`calories_zero`（16 道） |
| `description` | 3,954 道缺值 | 填空字串 | `has_description` |

各營養欄位的 clip 門檻（train p99）：

| calories | total_fat_pdv | sugar_pdv | sodium_pdv | protein_pdv | saturated_fat_pdv | carbohydrates_pdv |
|---|---|---|---|---|---|---|
| 3387.4 | 287.0 | 1110.5 | 198.0 | 178.0 | 385.0 | 151.0 |

`n_steps`（3–37）和 `n_ingredients`（4–20）在 PP 版資料中已經過濾過，不需要再處理。

## 評估

- **評估範圍**：val/test 中 `cold == 1` 的列；`cold == 0` 的列只作為對照。
- **指標**：主要用 **AUC**（正例約 80%，accuracy 沒有鑑別力）；另報 log loss、Brier score。
- **信賴區間**：cold 評估集 val 6,410、test 4,099 筆，所有指標都附 bootstrap 95% CI。
- **`rating >= 4` 的限制**：cold 列中負例只有約 4%（val 275 筆、test 169 筆），這個 label 的結果變異較大，只當作參考。
- **test 只跑一次**：調參只看 validation；最後用 train + validation 重新 fit，再評估 test。

## 防止 leakage

- validation 的特徵只用 train 計算；test 的特徵用 train + validation 計算。
- 新食譜沒有互動紀錄，食譜端只用**內容與作者**特徵，不用任何評論統計。`n_fit`、`cold` 只用來篩選評估列，不當特徵。
- train 每一列的 user 特徵只用該 user **在那天之前**的歷史（依時間累積），不包含該列自己的 label。

## 特徵

### User 端
評分次數、平均評分、收縮後的 P(5)、評分標準差、距上次評分天數、偏好的 tags/食材、平均烹飪時間與熱量、`PP_users_clean.csv` 的 techniques（選用）。

### Recipe 端（只用內容）
log(minutes)、n_steps、n_ingredients、log 後的 7 欄營養、前 K 個 tags/食材 multi-hot、name 與 description 長度、`has_description`、發布年份、作者在 fit 資料中其他食譜的平均評分與數量。

食譜的 techniques 向量需要 `PP_recipes.csv`，這裡不使用；tags、食材與 steps 已涵蓋大部分烹飪方式資訊。

### Match feature（相似度）
在 user 歷史的每道食譜 j 與新食譜 i 之間計算相似度，集合分別為 **tags**（先移除出現在 > 50% 食譜中的常見 tags）與 **ingredients**：

| 相似度 | 公式 | 出處 |
|---|---|---|
| Jaccard（主要） | \|A ∩ B\| / \|A ∪ B\| | recommendation 投影片 p28 |
| Cosine（對照） | \|A ∩ B\| / √(\|A\| · \|B\|) | recommendation 投影片 p29–35 |

**相似度加權平均 → 機率**（recommendation 投影片 p58–60，加上 temporal 投影片 p22–26 的時間衰減）：

```
P(u,i) = ( Σ_j w_uij · y_uj + α · p_u ) / ( Σ_j w_uij + α )
w_uij  = Sim(i,j) · exp(−λ · (t − t_uj) / 365)
```

- j 為 user 歷史中與 i 最相似的前 k 道；`y_uj = 1[r_uj == 5]`；`p_u` 為 user 收縮後的 5 分比例。
- `λ = 0` 為不衰減的版本；`α, β, k, λ` 在 validation 上調。

其他 match feature（給 LR 用）：

| 特徵 | 定義 |
|---|---|
| `{jac,cos}_{tag,ing}_wavg` | 上式的加權平均分數 |
| `jac_{tag,ing}_mean_liked` / `_max_liked` | 與 user 歷史中 rating == 5 食譜的 Jaccard 平均／最大值 |
| `jac_{tag,ing}_mean_disliked` / `_max_disliked` | 與 rating <= 4 食譜的 Jaccard 平均／最大值（無則補 0 並加 flag） |
| `jac_{tag,ing}_mean_recent` | 與 user 最近 N = 50 道評過的食譜的 Jaccard 平均 |
| `p_u_decay` | 時間衰減版 `p_u`（同一個 λ） |

另有：新食譜的時間／熱量與 user 平均的差距、user 對同一作者食譜的歷史 5 分比例、是否為 user 自己發布的食譜。train 列一律只用該列日期之前的歷史。

## 模型

詳細規格見 `SPEC.md`。

| 代號 | 模型 | 用途 |
|---|---|---|
| B0 | 全部預測 train 正例比例 | 下限 |
| B1 | user 收縮後的 P(5) | 強 baseline（train 有 75.5% 是 5 分） |
| J1 / J2 | Jaccard 加權平均（tags / ingredients），λ = 0 | 課堂方法，不需訓練 |
| C1 / C2 | Cosine 加權平均（tags / ingredients），λ = 0 | 對照：比較特徵表示法 |
| J1-T … C2-T | 上面四個加上時間衰減（λ > 0） | 檢查近期評分是否比較準 |
| **LR** | Logistic Regression，全部特徵 | **主模型**，L2 正則化 |
| GBDT（對照） | sklearn HistGradientBoosting，同樣特徵 | 檢查非線性能否再提升 |

主要比較：相似度加權平均的 2 × 2 × 2 表（Jaccard／Cosine × tags／ingredients × 有無時間衰減）。

Ablation（LR）：只用 user 特徵 → 加相似度 → 加食譜內容 → 加作者；並比較 label 用 `== 5`（主）與 `>= 4`、cold 與 warm 的表現。

## 執行

```
python3 preprocess.py          # 產生 data_clean/（原始檔放在 dataset/）
```

再依序執行 notebook：`01_eda` → `02_baselines` → `03_similarity` → `04_lr` → `final`。
02–04 會把選出的超參數存到 `results/params.json`，`final.ipynb` 只讀取它，test 也只在 `final.ipynb` 評估。
程式模組：`features.py`（資料、snapshot、相似度、特徵）、`models.py`（調參、LR、訓練列）、`evaluation.py`（指標、bootstrap、配對 bootstrap）、`ranking.py`（Hit@10／NDCG@10）、`config.py`（共用設定）。

記憶體：03、04、final 最高約 3.6 GB。

## 使用注意
- CSV 中的空字串會被 pandas 讀成 NaN，讀取 `RAW_recipes_clean.csv` 後要對 `description` 做 `.fillna("")`，或改用 `has_description`。
- `tags`、`ingredients`、`steps` 是字串形式的 list，要用 `ast.literal_eval` 轉回 list。
- minutes、nutrition 的 log 轉換與標準化留到 feature engineering 階段。

## 結果

test 的冷啟動列（4,099 筆，2014–2018），label 為 `rating == 5`。CI 為以 user 為單位的 bootstrap 95%（1,000 次）；
ΔAUC 為對 B1 的配對 bootstrap，p 為單尾（模型不比 B1 好的比例）。Hit@10：1 個 5 分新食譜 + 99 個未評過的新食譜，共 1,000 組，隨機為 0.10。
完整表格與圖在 `results/`（`summary.csv`、`test_main.csv`、`test_vs_B1.csv`、`test_ranking_hit10.csv`）。

| 模型 | AUC [95% CI] | ΔAUC vs B1 (p) | log loss | Brier | Hit@10 |
|---|---|---|---|---|---|
| B0 | 0.500 | – | 0.510 | 0.164 | – |
| B1（user P(5)） | 0.721 [0.683, 0.755] | – | 0.457 | 0.147 | 0.100 |
| J1（Jaccard tags） | 0.725 [0.689, 0.758] | +0.005 (0.11) | 0.455 | 0.146 | 0.139 |
| J2（Jaccard ingredients） | 0.727 [0.692, 0.759] | +0.006 (0.03) | 0.453 | 0.145 | 0.118 |
| C1（Cosine tags） | 0.726 [0.688, 0.758] | +0.005 (0.08) | 0.454 | 0.145 | 0.136 |
| C2（Cosine ingredients） | 0.728 [0.693, 0.760] | +0.007 (0.01) | 0.453 | 0.145 | 0.122 |
| **LR（主模型）** | **0.741 [0.708, 0.771]** | **+0.020 (0.002)** | **0.444** | **0.142** | **0.205** |
| GBDT（對照） | 0.741 [0.710, 0.770] | +0.020 (0.002) | 0.443 | 0.143 | 0.220 |

時間衰減版（-T）與不衰減版差距 < 0.001 AUC，表中省略。選出的超參數：`TRAIN_MODE = pseudo_cold`、`C = 0.1`、不加 class_weight（見 `results/params.json`）。

Overall：
- **相似度單獨使用只小幅提升**（+0.005 至 +0.007 AUC），ingredients 在 AUC 上較好，tags 在 Hit@10 上較好。
- **Jaccard 與 Cosine 差不多，時間衰減幾乎沒有影響。**
- **LR 結合相似度、食譜內容與作者特徵後提升才明顯**（+0.020 AUC，p = 0.002），Hit@10 約為隨機的 2 倍；GBDT 與 LR 相當，非線性沒有帶來額外好處。
- **user 自己的歷史是主要訊號**：validation 的 ablation 中，拿掉 user 特徵 AUC 降約 0.02–0.03，只加相似度到 user 特徵上 +0.001。
- 限制：冷啟動列只有幾千筆，CI 約 ±0.03，小差距（如 Jaccard vs Cosine）分不出高下；test 跨 5 年，模型只用 2014 年以前的資料訓練。

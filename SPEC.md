# Spec：Food.com 新食譜喜好預測（CSE 258 Assignment 2）

## 1. 任務

- **輸入**：`(user_id, recipe_id)`，recipe 是**新食譜**（在模型可用的資料中從未被評論過）。
- **輸出**：P(user 很喜歡這道食譜) = P(rating == 5)，介於 0～1。user 實際給幾分都有可能，模型輸出的是「給滿分」的機率。
- **主 label**：`y = 1[rating == 5]`；5 分為 positive，1–4 分為 negative。
- **敏感度分析**：label 改為 `y = 1[rating >= 4]`（cold 列 negative 只有 val 275 筆、test 169 筆，結果只當參考）。
- **為何用 == 5**：資料的差異主要在 4 分與 5 分之間（train 4 分 129,323 筆、5 分 516,598 筆，1–3 分合計 38,075 筆）；cold 列 negative 有 val 1,436 筆、test 836 筆，評估比 `>= 4` 穩定。
- **核心方法**：課堂教的 **Jaccard 相似度**，改用**食譜內容集合**（tags、ingredients）計算，再以相似度加權平均轉成機率。
- **兩個延伸（皆為課堂內容）**：
  1. **Cosine 相似度**當對照組，比較特徵表示法（recommendation 投影片 p29–35）。
  2. **時間衰減**：越近期的評分權重越大（temporal 投影片 p10–13、p22–26）。

## 2. 資料

來源：Food.com（Majumder et al., EMNLP 2019），前處理見 `preprocess.py` 與 `README.md`。

| split | 期間 | 列數 | 評估用 cold 列 | rating==5 | rating>=4 |
|---|---|---|---|---|---|
| train | ～2012-12-31 | 683,996 | — | 75.5% | 94.4% |
| validation | 2013 | 22,933 | **6,410**（859 users） | 77.6%（cold） | 95.7%（cold） |
| test | 2014-01-01～2018-12-18 | 22,224 | **4,099**（857 users） | 79.6%（cold） | 95.9%（cold） |

- 依時間切分（`VALID_START = 2013-01-01`、`TEST_START = 2014-01-01`）；user 在 train 期間至少 3 筆有效評分（27,327 人）；rating = 0 已刪除。
- 資料檔在 `data_clean/`。
- 食譜內容：`RAW_recipes_clean.csv`（tags、ingredients、steps、minutes、營養、作者、發布日）。

## 3. 評估

- **評估列**：val/test 中 `cold == 1`；`cold == 0` 只作對照。
- **指標**：AUC（主要）、log loss、Brier score；全部附 bootstrap 95% CI（1,000 次，以 user 為單位重抽）。
- **配對 bootstrap**：在同一批 bootstrap 樣本上算 `metric(模型) − metric(B1)`，回報差距的 95% CI 與「沒有比較好」的比例（單尾 p 值）。
- **排名指標**：Hit@10、NDCG@10。每個 list = 一筆 5 分的 cold 評論 + 99 道該 user 沒評過、當天已發布、在 history 中沒有評論的新食譜；抽 1,000 個 list。B0、B1 在同一個 list 內分數相同，等於隨機（Hit@10 = 0.10），所以這個指標專門衡量內容相似度。
- **輔助驗證集（選做）**：validation cold 列有 6,410 筆，通常已足夠；需要時可另從 train 最後一年（2012）取「該食譜的第一筆評論」當輔助驗證集（特徵只用 2012 以前的資料）。
- **流程**：在 validation 調參 → 用 train + validation 重算特徵並重新 fit → test **只跑一次**。

## 4. 防止 leakage

1. validation 特徵只用 train；test 特徵用 train + validation。
2. 新食譜只用內容與作者特徵；`n_fit`、`cold` 只用於篩選，不當特徵。
3. **Snapshot 設計**：每組資料都是 (history, targets)，特徵只用 history 計算，且 history 全部早於 targets。

   | 用途 | history | targets |
   |---|---|---|
   | LR 訓練 | train 中 Y 年以前 | train 中 Y 年的列（Y = 2011、2012） |
   | validation | train | validation |
   | test | train + validation | test |

   LR 的訓練列因此和 validation／test 的設定完全相同，不會看到自己的 label 或未來。作者特徵會扣掉目標食譜自己的評論，讓 warm 列和 cold 列一致。

   訓練列有三種模式（`config.TRAIN_MODES`），在 validation 上比較後擇一：
   - `all`：2011、2012 年的所有列。
   - `pseudo_cold`：2009–2012 年中，食譜在該年以前沒有任何評論的列（和 cold 列相同的情況）。
   - `pseudo_cold_w`：同上，每列權重 1/√(該 user 的列數)。
4. 常見 tag 過濾門檻、平滑參數等統計只在 fit 資料上計算。

## 5. 相似度與機率

### 5.1 定義
兩種相似度，都在相同的食譜集合 S 上計算：

| 相似度 | 公式 |
|---|---|
| Jaccard | J(i, j) = \|S_i ∩ S_j\| / \|S_i ∪ S_j\| |
| Cosine（0/1 向量） | C(i, j) = \|S_i ∩ S_j\| / √(\|S_i\| · \|S_j\|) |

S 分別為：
- **tags 集合**：先移除出現在 > 50% 食譜中的 tags（例如 `time-to-make`、`course`、`preparation`），門檻在 validation 上確認。
- **ingredients 集合**：食材字串小寫、去空白後直接比對。

### 5.2 為何不用課堂的 user 集合版本
課堂版以「評過該食譜的 user 集合」計算相似度；新食譜沒有評論，集合為空，相似度恆為 0。改用內容集合是本任務的主要設計。

Cosine 對集合大小的懲罰比 Jaccard 小（分母是幾何平均而不是聯集），食材數差很多的兩道食譜在 cosine 下會比較像，這是兩者比較的重點。

### 5.3 相似度 → 機率（相似度加權平均，recommendation 投影片 p58–60）

```
            Σ_{j∈N_k(u,i)} w_uij · y_uj  +  α · p_u
P(u,i) =  ─────────────────────────────────────────
            Σ_{j∈N_k(u,i)} w_uij          +  α

w_uij = Sim(i,j) · exp(−λ · (t − t_uj) / 365)
```

- `Sim`：Jaccard 或 Cosine
- `t`：這次評論的日期；`t_uj`：user 評食譜 j 的日期；時間差以年計
- `λ`：時間衰減率；**λ = 0 就是沒有時間衰減的版本**，λ 越大越只看近期評分
- `N_k(u,i)`：user u 歷史中與新食譜 i 最相似的前 k 道食譜（依 Sim 排序，不受衰減影響）
- `y_uj = 1[r_uj == 5]`
- `p_u`：user 收縮後的 5 分比例 `(Σ y_uj + β·p̄) / (|I_u| + β)`，`p̄` 為全體 5 分比例
- `α, β, k, λ`：在 validation 上調（AUC 為主、log loss 為輔）；λ 的候選值例如 {0, 0.1, 0.3, 1, 3}
- 沒有相似食譜時自然退回 `p_u`

## 6. 特徵

| 類別 | 特徵 |
|---|---|
| User | `p_u`、評分次數（log）、平均評分、評分標準差、距上次評分天數 |
| 相似度（tags、ingredients 各一組） | 5.3 的 8 個加權分數，改用**相對版** `wa − p_u`（原始分數和 `p_u` 高度相關，相對版只保留這道食譜和 user 平常的差距）、與 5 分食譜的平均/最大 J、與非 5 分（1–4 分）食譜的平均/最大 J（無則補 0 並加 flag）、與最近 N = 50 道的平均 J |
| 時間 | 時間衰減版 `p_u`（同一個 λ）、user 最近一年的 5 分比例 |
| Recipe 內容 | log(minutes)、n_steps、n_ingredients、log 營養 7 欄、`has_description`、description 長度、發布年份 |
| 作者 | 作者在 fit 資料中其他食譜的 5 分比例（平滑）與食譜數；user 對同一作者的歷史 5 分比例；是否為 user 自己的食譜 |
| 其他 match | 新食譜 minutes/calories 與 user 歷史平均的差距 |

## 7. 模型

| 代號 | 模型 | 說明 |
|---|---|---|
| B0 | 常數 `p̄` | 下限，AUC = 0.5 |
| B1 | `p_u` | 強 baseline（train 有 75.5% 是 5 分） |
| J1 | Jaccard 加權平均（tags），λ = 0 | 5.3，不需訓練 |
| J2 | Jaccard 加權平均（ingredients），λ = 0 | 同上 |
| C1 / C2 | Cosine 加權平均（tags / ingredients），λ = 0 | 對照組：比較特徵表示法 |
| J1-T / J2-T / C1-T / C2-T | 上面四個加上時間衰減（λ > 0） | 檢查近期評分是否比較準 |
| **LR** | Logistic Regression，第 6 節全部特徵 | **主模型**；標準化後 L2 正則化；訓練列模式與 C 以 validation cold 列的 **log loss** 選（輸出是機率；不同 C 之間 AUC 只差約 0.001）；不用 `class_weight="balanced"`（會把機率拉向 0.5） |
| GBDT（選做） | sklearn `HistGradientBoostingClassifier`，同樣特徵 | 檢查非線性能否再提升 |

**不適用的模型**：
- latent factor model（MF）與 bias model 的 β_i：新食譜沒有評論，γ_i、β_i 會因正則化退回 0（recommendation 投影片 p102–103）。
- user 集合的 Jaccard：同上，集合為空。
- BPR／one-class：適用於只有隱式回饋的情況；本資料有明確評分與 negative。
- FPMC、RNN 等序列模型：任務不是預測下一個 item，且新食譜沒有 item embedding。

## 8. 實驗

1. **主結果**：所有模型在 val/test cold 列的 AUC、log loss、Brier（含 CI）。
2. **相似度加權平均的 2 × 2 × 2 比較**（主要圖表）：

   | | tags | ingredients |
   |---|---|---|
   | Jaccard，λ = 0 | J1 | J2 |
   | Jaccard，λ > 0 | J1-T | J2-T |
   | Cosine，λ = 0 | C1 | C2 |
   | Cosine，λ > 0 | C1-T | C2-T |

   每格報 AUC（含 CI），並與 B1 比較。
3. **Ablation（LR）**：只用 user 特徵 → + Jaccard → + 內容 → + 作者。
4. **超參數**：k、α、λ、tag 過濾門檻對 AUC 的影響；λ 畫成曲線（λ = 0 → 3）。
5. **對照**：cold vs warm 列的表現差距。
6. **敏感度**：label 改為 `rating >= 4`（negative 很少，只檢查結論方向是否一致）。
7. **解讀**：LR 係數（例如跟 user 給過 5 分的食譜越像 → 越可能給 5 分）。
8. **配對檢定**：每個模型 − B1 的 ΔAUC 與 CI；ablation 以「只用 user 特徵」為基準；訓練列模式以 `all` 為基準。
9. **排名**：8 個相似度版本、LR、GBDT 的 Hit@10／NDCG@10（以 J1 為基準做配對比較）。
10. **診斷**：依 user 歷史長度（3–5、6–20、21+）拆開的 AUC；calibration 圖。

## 9. 報告對應（作業 5 部分）

| 作業部分 | 內容 |
|---|---|
| 1. 資料與探索 | 資料規模、評分分布（75.5% 為 5 分）、時間範圍、新食譜比例、tags/食材分布、為何改用時間切分 |
| 2. 任務與特徵 | 新食譜喜好預測、評估方式與 CI、前處理、探索結果如何支持相似度與時間特徵（例如 user 的 5 分比例是否隨時間變化） |
| 3. 模型 | 第 5、7 節；baseline、調參、不適用模型的理由、overfitting 與資料規模問題 |
| 4. 文獻 | Majumder et al. 2019（資料集）、Linden et al. 2003（item-to-item CF）、Koren 2009（temporal dynamics）、Koren, Bell & Volinsky 2009（MF 與 cold-start）、Kabbur et al. 2013（FISM：user = 評過 item 的平均） |
| 5. 結果 | 第 8 節；哪些有效／無效、係數解讀、結論 |

## 10. 實作步驟

| 檔案 | 內容 |
|---|---|
| `features.py` | 載入資料、tags／ingredients 矩陣與 tag 過濾、snapshot、Jaccard／Cosine、相似度加權平均、全部特徵 |
| `models.py` | B0、β 與加權平均的調參、LR（標準化 + L2）、係數表 |
| `evaluation.py` | AUC／log loss／Brier、以 user 為單位的 bootstrap CI、配對 bootstrap、平均值的 CI |
| `ranking.py` | 排名 list 的抽樣、Hit@10／NDCG@10（同分以期望位置計算） |
| `config.py` | 共用設定（label、H、bootstrap 次數、各參數格點），以及 notebook 之間傳遞超參數的 `results/params.json` |
| `1_data_overview.ipynb` | 探索分析（不選參數） |
| `2_baselines.ipynb` | B0、B1；選 `BETA`、`MAX_DF` |
| `3_similarity.ipynb` | 8 個相似度版本；選每個版本的 k、α、λ（`WA`）；2 × 2 × 2 表（AUC 與 Hit@10）、與 B1 的配對比較、λ、k 曲線 |
| `4_lr.ipynb` | 比較三種訓練列模式並選 `TRAIN_MODE`、`C`、`CLASS_WEIGHT`；ablation、係數、GBDT、cold vs warm、歷史長度、calibration、`>= 4` 敏感度 |
| `5_results.ipynb` | 只讀 `params.json`、不調參；validation 結果、**test 只在這裡跑一次**、報告用的圖表 |

執行順序：`1` → `2` → `3` → `4` → `5`。開發用的 notebook（1–4）不碰 test。結果表與圖存到 `results/`。

固定設定：每個 user 只取最近 H = 100 道歷史食譜計算相似度；tag 過濾門檻 `max_df` 從 {0.2, 0.35, 0.5, 0.75, 1.0} 中在 validation 上選。

# 結果

## 第二輪（本機 RTX 5070 Ti，2026-10-14）

**設定**：daily 4.1h + mixed 34.3h + wordshk 104.2h = **142.6 小時**；字典 9689 個單位（覆蓋 99.76%）；
reader 11.1M 參數，30 epoch（10830 步），feature 遮罩（時間 + 維度），`--head frozen`。

**數據檢查**：驗證 shard 同 train 句子重疊 — daily 0%、mixed 0%、wordshk 0.1%（3 句），可以忽略。

### 驗證 CER（frozen head）

| 步 | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 9500 | 10830 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CER | 94.4% | 34.1% | 25.2% | 19.4% | 18.1% | 17.8% | 16.5% | 16.1% | 15.7% | 15.2% | **14.96%** | 14.97% |

Train 時間：約 1 小時 30 分（每步約 0.24 秒前半，後半同時做評估慢少少）。

### 解碼：greedy vs beam search + 4-gram LM

驗證 shard 打亂分兩半：**dev** 揀 α／β，**test** 報告（`evaluate.py --part`）。

| | dev（1030 句） | **test（1030 句，冇用嚟調參）** |
|---|---|---|
| greedy | 15.95% | **15.77%** |
| beam，冇 LM | — | 15.88% |
| beam + LM（α=0.7，β=5.0） | 13.05% | **12.99%** |

- LM 減少大約 **18%** 嘅錯字（15.8% → 13.0%），主要改正同音字：「珍星→珍惜」「忘緊→忙緊」「成聲成對→成雙成對」「宿四→宿舍」。
- β 要大（4–5）：LM 偏好短句，冇 β 補償會漏字。
- 再加 10.9 萬句粵語文字（raptorkwok 平行語料，已刪走同驗證句子一樣嘅行）入 LM：dev 13.05% → 13.03%，冇分別。
  ⚠️ 呢份語料有 29% wordshk 驗證句子原句，一定要先過濾，否則評估會偏高。

### `samples/`（OmniVoice 合成語音，參考字幕由 SenseVoice 出）

| | yue-1 | yue-2 | yue-3 | en-1 | en-2 | en-3 | cmn | ja |
|---|---|---|---|---|---|---|---|---|
| greedy | 5% | 0% | 13% | 27% | 33% | 42% | 53–72% | 100% |
| + LM | 5% | 0% | 7% | 27% | 42% | 42% | 53–67% | 100% |

粵語好好（剩低錯嘅係「啊→呃／壓」語氣詞）；英文只識粵英夾雜數據入面嘅詞；普通話、日文冇 train 過。
⚠️ 只有 3 句粵語，而且係合成語音 —— 真人表現未知，要用 `scripts/record.py` 錄真人測試集。

速度（短句逐句轉錄，連 LM）：大約 110–200× 即時；未量長音頻批次。

### frozen vs learned（同樣數據、同樣設定，只換分類頭）

| | frozen（EmbeddingGemma 文字空間，固定） | learned（普通 CTC 分類頭） |
|---|---|---|
| 最好驗證 CER | 14.96% | **14.24%** |
| test greedy | 15.77% | **14.90%** |
| test + LM（各自喺 dev 揀 α／β） | 12.99%（α=0.7 β=5.0） | **12.59%**（α=0.7 β=4.0） |
| 去到 15% 要幾多步 | 9500 | 5500 |
| 每步時間 | 約 0.24s | 約 0.4s（要更新 9689×768 字典） |
| **LibriSpeech test-clean 200 句（英文，冇 train 過）WER** | **54.8%** | 57.9% |
| samples en-1 ／ cmn-3 | **27% ／ 53%** | 55% ／ 73% |

**tuned**（用 text embedding 做起點，一齊 train）：最好驗證 14.54%；test greedy 15.16%、+LM 12.70%（α=0.7 β=4.0，dev 揀）；
LibriSpeech WER **59.1%**（三個入面最差）—— 介乎兩者之間，而且將 frozen 嘅泛化優勢 train 走咗。

**learned 大 reader**（d_model 512、8 層、8 heads，reader 約 26M，連字典 33.3M；每步約 0.8s，約 2.5 小時）：
最好驗證 14.09%；test greedy 14.97%、+LM **12.42%**（α=0.7 β=4.0）；LibriSpeech WER 57.9%。
同細 reader（14.90% ／ 12.59%）差唔多 —— **瓶頸唔係 reader 大細**，係凍結 encoder 嘅 feature 同數據。

**結論**：喺同類數據（粵語），**固定用 EmbeddingGemma 文字空間冇幫助**，普通分類頭仲準少少、學得快好多。
frozen 唯一嘅優勢係對冇 train 過嘅語言泛化好少少（英文 WER 54.8% vs 57.9%），差距唔大。

### 同 SenseVoice 比較（同一批 test 句子，由原始音頻轉錄）

`scripts/export_testset.py` 將 `evaluate.py --part test` 嗰 1030 句（102 分鐘）嘅原始音頻對返出嚟（`data/testset/`），
再用 Gaijincall 嘅 SenseVoice（sherpa-onnx int8，CPU 8 線程，`use_itn=False`）轉錄。字典外嘅字都計錯。

| | **合計** | daily | mixed（粵英夾雜） | wordshk |
|---|---|---|---|---|
| learned greedy | 15.1% | 9.5% | 19.0% | 6.4% |
| learned + LM（α=0.7 β=4.5） | 13.0% | 9.5% | 16.4% | **5.4%** |
| frozen greedy ／ + LM | 15.9% ／ 13.2% | | | |
| SenseVoice 2024-07-17 | **9.9%** | 5.4% | **11.1%** | 7.2% |
| SenseVoice 2025-09-09-yue | 10.3% | **4.1%** | 12.1% | 6.7% |

- 整體 **SenseVoice 贏**（9.9% vs 13.0%）。
- 我哋只係喺 wordshk 贏 —— 但我哋 train 過同類數據（同風格、可能同一個 TTS 聲），唔公平。
- 最似真實對話嘅 daily／mixed，SenseVoice 明顯好啲。
- 速度：SenseVoice CPU 約 85× 即時；我哋 GPU 約 1000×（硬件唔同）。
- **結論**：Gaijincall 繼續用 SenseVoice。呢個方法要追上，大概要解凍部分 encoder 或者用多好多真人數據。

### 開放字典（只有 frozen 做得到）

frozen 嘅字典係 text embedding，可以推理時加新字唔使重新 train（`scripts/open_vocab.py`）。
加 6615 個常用英文詞（google-10000-english）：

| | 字典 | 參考字唔喺字典 | LibriSpeech WER |
|---|---|---|---|
| frozen | 9689 | 30.9% | **54.8%** |
| frozen + 英文詞 | 16304 | 11.1% | 55.8% |

**唔得**：新字會被揀（「feels」啱，「whilst」「whats」「containing」錯），但整體冇改善。
reader 只係學識對準 train 過嘅字附近，冇學到通用「聲音 → 文字空間」對應；要開放字典就要好似 eg2 咁，用大字典嘅語言直接 train。
粵語唔受影響（加英文詞之後 yue samples 一字不差）。

### 速度（RTX 5070 Ti，greedy，唔計讀檔）

10 分鐘音頻切 30 秒一段：batch 1 → 1.12s（535×）；**batch 8 → 0.58s（約 1000×，一個鐘約 3.5 秒）**；batch 20 → 0.64s。
（eg2：MacBook Air Neural Engine 一個鐘 13 秒 —— 硬件唔同，唔可以直接比。）

## 第一輪（Colab L4）

98 小時（wordshk 限 60h）、冇遮罩、lr schedule 有 bug（第 6000 步 lr 已經跌到 0）：驗證 CER **18.26%**（greedy）。
驗證句子同第二輪唔同，數字唔可以直接比較。

## 第三輪：加真人數據（MDCC）

MDCC（`ming030890/mdcc`）：香港粵語有聲書真人錄音，逐字口語字幕；train 約 6.5 萬句（約 73 小時）。

### 起點：MDCC test 頭 2000 句（107.6 分鐘），未用 MDCC train

| | greedy | + LM（α=0.7 β=4.5） |
|---|---|---|
| learned | 18.9% | 15.8% |
| learned 大 | 18.0% | **15.4%** |
| frozen | 19.0% | 15.8% |
| SenseVoice 2024-07-17 | **8.0%** | |
| SenseVoice 2025-09-09-yue | **7.7%** | |

真人錄音差距更大（約一倍）。注意：唔知 SenseVoice 自己有冇用 MDCC train 過。

### 加 MDCC 之後（learned，11M reader，daily + mixed + wordshk + MDCC 57.5h ≈ 200h）

字典 10061；15630 步（約 83 分鐘）；最好驗證 12.48%（驗證集多咗 MDCC，同之前唔可以直接比）。

**MDCC test 2000 句**（LM 用 MDCC train 字幕；test 有 11.1% 句子原句喺 train 字幕出現過，所以另計「剔走重複句」）：

| | 全部 2000 句 | 剔走 222 句重複 |
|---|---|---|
| 未加 MDCC（learned + LM） | 15.8% | |
| **加 MDCC greedy** | 7.1% | 7.5% |
| **加 MDCC + LM** | **5.6%** | **6.0%** |
| SenseVoice 2024-07-17 | 8.0% | 7.8% |
| SenseVoice 2025-09-09-yue | 7.7% | 7.6% |

**原本 1030 句 test**（由原始音頻）：

| | 合計 | daily | mixed | wordshk |
|---|---|---|---|---|
| 未加 MDCC（learned + LM） | 13.0% | 9.5% | 16.4% | 5.4% |
| **加 MDCC + LM** | **12.3%** | 7.3% | 15.6% | 5.0% |
| SenseVoice 2024-07-17 | 9.9% | 5.4% | 11.1% | 7.2% |

**結論**
- 57.5 小時真人數據令 MDCC test 由 15.8% 跌到 **5.6%**（剔走重複句 6.0%），**贏 SenseVoice（7.6–7.8%）**。
  ⚠️ 但係同分佈（MDCC 有聲書風格）—— SenseVoice 係 zero-shot，我哋 train 過同類錄音，唔係完全公平。
- 其他數據只係小幅進步（daily 9.5% → 7.3%），整體仲輸 SenseVoice（12.3% vs 9.9%），尤其粵英夾雜。
- 即係話：**凍結 encoder + 細 reader 只要有對口嘅真人數據，可以追到甚至超越 SenseVoice**；瓶頸主要係數據，唔係 reader。

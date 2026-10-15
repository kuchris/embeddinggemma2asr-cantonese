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
再用另一個 project 嘅 SenseVoice（sherpa-onnx int8，CPU 8 線程，`use_itn=False`）轉錄。字典外嘅字都計錯。

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
- 速度：SenseVoice CPU 約 85× 即時；我哋 GPU 約 1000×（⚠️ 硬件唔同，唔可以比）。
- **結論**：實際應用暫時仍然用 SenseVoice。呢個方法要追上，大概要解凍部分 encoder 或者用多好多真人數據。

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

> ⚠️ 以上所有 SenseVoice 數字都係借用另一個 project 嘅 sherpa-onnx int8 CPU 模型同環境量出嚟；準確度可以參考，速度唔可以同我哋（GPU）比較。
> 之後應該喺 gemma2asr 入面獨立準備 GPU 版 SenseVoice 再比。

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

### SpiCE 真人訪談（最難嘅測試）

[SpiCE corpus](https://spice-corpus.readthedocs.io/)（CC BY 4.0）：溫哥華粵英雙語人士訪談，人手校正字幕。
用 VF19A + VM19A 嘅「interview」部分（約 52 分鐘，受訪者聲道），`scripts/spice_eval.py`。
兩邊都刪走轉錄標記同 um／uh 等語氣詞，統一「系／係」「姐係／即係」寫法。英文詞佔 11%／27%。

| | 逐句（按人手切好嘅句） | 長音頻（每 30 秒一段） |
|---|---|---|
| learned_r3 greedy | 49.2% | 89.3% |
| learned_r3 + LM | 47.2% | 84.4% |
| SenseVoice 2024-07-17 | 33.0% | **40.7%** |
| SenseVoice 2025-09-09-yue | **29.7%** | 42.9% |

- 自然對話 + 大量粵英夾雜 + 海外口音：SenseVoice 都有 30% 錯；我哋 47%，冇 train 過呢類語料。
- 長音頻差距最大（84% vs 41%）：受訪者聲道有訪問者漏音同靜音，我哋照樣轉錄（insertion 好多），
  SenseVoice 識得忽略。我哋只 train 過乾淨、有人講嘢嘅短句。
- 長音頻參考字幕只覆蓋約 60% 時間（受訪者講嘢部分），所以長音頻數字兩邊都偏高，只適合相對比較。
- **要用嚟做會議：** 要加真實對話／粵英夾雜數據，同埋處理靜音同背景人聲（加 VAD 或者 train 埋冇人講嘢嘅片段）。

#### 加 VAD（silero，`g2asr/vad.py`）

長音頻先用 VAD 搵出有人講嘢嘅段落（合併到 ≤20 秒），先交畀 reader。silero 改咗做 batch（長音頻切成 30 秒一條流並排行），
30 分鐘由 10.4 秒 → 1.2 秒。

| SpiCE 長音頻（52 分鐘） | 冇 VAD | **有 VAD** |
|---|---|---|
| learned_r3 + LM | 84.4% | **45.7%** |
| SenseVoice 2024-07-17 | 40.7% | **28.6%** |
| SenseVoice 2025-09-09-yue | 42.9% | 28.9% |

速度（VF19A 30 分鐘訪談，同一批 VAD 段）：

| | VAD | 轉錄 | 合計 |
|---|---|---|---|
| **我哋**（GPU，greedy） | 1.2s | 2.0s | **3.2s（約 560× 即時）** |
| SenseVoice（CPU 8 線程） | 1.2s | 11.6s | 12.8s |

- VAD 對兩邊都大幫助（擋走訪問者漏音同靜音），我哋受益最大（84% → 46%）。
- ⚠️ 速度比較唔公平：SenseVoice 用嘅係另一個 project 嗰個 int8 **CPU** 版，我哋用 GPU。GPU 版 SenseVoice 未測，「快幾倍」唔成立。
- 準確度：自然對話仍然輸 SenseVoice（46% vs 29%）—— 要靠對話類 train 數據追。
- `transcribe.py` 長過 20 秒嘅音頻預設用 VAD（`--no-vad` 關閉）。

## 公平比較：SenseVoice 都用 GPU（gemma2asr 獨立環境）

`scripts/baseline_sensevoice.py`：FunASR + `FunAudioLLM/SenseVoiceSmall`（fp32，GPU），`uv sync --extra baseline`。
唔再借用其他 project 嘅環境。速度唔計讀檔，GPU 預熱過，同一批檔案；兩邊都試過按長度排序同唔同 batch，取最好。

我哋推理改咗做 `g2asr/infer.py`：按長度排序 + encoder／reader 成批行 + bf16（準確度唔變：MDCC 7.06% ／ +LM 5.63%）。

| 測試集 | 我哋 learned_r3 CER（greedy ／ +LM） | SenseVoice GPU CER | 我哋速度（greedy） | SenseVoice 速度 |
|---|---|---|---|---|
| 原本 test 1030 句（102 分鐘） | 14.2% ／ 12.3% | **9.7%** | **約 1000×** | 約 280–320× |
| MDCC test 2000 句（108 分鐘） | 7.1% ／ **5.6%** | 7.9% | **約 1000×** | 約 280× |
| MDCC test 剔走重複句 | 7.5% ／ **6.0%** | 7.8% | | |
| SpiCE 逐句 | 49.2% ／ 47.2% | **32.6%** | | |
| SpiCE VAD 段（31 分鐘） | 47.8% ／ 45.7% | **28.0%** | **約 900×** | 約 300–365× |

- **兩邊都用 GPU：我哋 greedy 快大約 3 倍。** 加 LM（純 Python beam search）就慢返落 300× 左右，同 SenseVoice 差唔多。
- 準確度：MDCC 贏；其他（尤其自然對話）輸。
- Windows 上 batch 太大會頂爆 VRAM 變慢好多（我哋 bs 16 最好）；量度有波動，以上取最好嗰次。

## 第四輪：老師標籤（知識蒸餾，Colab A100）

**數據**：`alvanlii/cantonese-radio`（香港電台／podcast，有存取限制）每句都有 Whisper 同 SenseVoice 兩個轉錄。
`scripts/extract_radio.py`：統一寫法（「系→係」，保留「系統／學系」等）、刪語氣詞（誒嗯呃），
**兩個老師 CER < 25%、SNR ≥ 15、1–30 秒**先留，用 SenseVoice 版本做標籤。
85 個 parquet 檔（每隔 12 個揀一個）、3 條流程並行，保留率約 60–75%，共 **519 小時**。
（最初試過 CER < 15%，只留 23%，太少。）

加埋 daily + mixed + wordshk + MDCC ≈ **720 小時**；字典 12368；LM 27 萬句。
learned reader（20.6M 連字典），12 epoch、14076 步、每批 6 萬格（12 萬格會 OOM），A100 約 1 小時 40 分。

驗證 CER（驗證集多咗 3 個 radio shard，唔可以同之前直接比）：
1000 步 26.5% → 3000 步 13.1% → 6000 步 10.3% → 10000 步 9.0% → 最後 **8.6%**。

| | 第三輪（冇 radio） | **第四輪（+519h radio）** | SenseVoice GPU |
|---|---|---|---|
| 原本 test greedy ／ +LM | 13.98% ／ 11.83% | **12.21% ／ 11.57%** | 9.7% |
| MDCC test greedy ／ +LM | 7.1% ／ 5.6% | **6.6%** ／ 5.7% | 7.9% |
| **SpiCE 對話（VAD）greedy ／ +LM** | 47.8% ／ 45.7% | **38.0% ／ 38.5%** | 28.0% |

- **真人對話進步最大**：SpiCE 47.8% → 38.0%（約 10 個百分點），同 SenseVoice 差距由 18 點收窄到 10 點。
- 讀稿類（原本 test、MDCC）只係小幅進步；MDCC 加 LM 冇再進步（5.6% → 5.7%）。
- SpiCE 上 LM 反而差少少（38.0% → 38.5%）：α／β 係喺讀稿類 dev 揀，唔啱對話。
- 學生大概只能追近老師（SenseVoice）；想再好要更準嘅老師（Whisper large-v3／Qwen3-ASR）或者解凍 encoder。

**成本**：A100 每小時 5.3 units，成個流程（抽 feature 約 2 小時 + train + 評估）共 **22.4 units**。
抽 feature 時 GPU 大部分時間等 CPU（解碼／重新取樣音頻）；train 時等硬碟讀 shard（已改 `g2asr/fastdata.py` 預先載入 RAM，下次用）。

**檔案**：本機 `runs/learned_r4/best_infer.pt`（推理用，121MB）、`data/lm_r4.pkl`、`runs/round4/`（log）；
Google Drive `MyDrive/gemma2asr/round4/`（連完整 checkpoint）、`MyDrive/gemma2asr/feats_r4/`（全部 feature，63GB）。

## LLM 修正（Qwen3.5-0.8B，冇 fine-tune）

`scripts/llm_correct.py`：learned_r4 greedy 輸出交畀 Qwen3.5-0.8B（bf16，本機 RTX 5070 Ti，batch 16），
提示叫佢只改明顯錯字、保留口語、唔好加減內容。每個測試集隨機 300 句。

| 測試集 | ASR | + LLM 修正 | 改咗 | 每句 |
|---|---|---|---|---|
| daily + mixed | 14.6% | **17.9%** ❌ | 95 句 | 85ms |
| MDCC test | 7.6% | **9.3%** ❌ | 35 句 | 54ms |
| SpiCE 對話 | 38.1% | **41.5%** ❌ | 74 句 | 95ms |

全部變差：刪字（「有一套石氣」→「一套石氣」）、改書面語（呢個→這個、佢→他）、將英文翻譯做中文（direction→方向）、
重寫句子；真正嘅同音錯字（石氣→石器、fall研→科研）反而改唔到。
**唔 fine-tune 嘅細 LLM 冇用**；要用「ASR 輸出 → 正確答案」配對 fine-tune 先有機會。

Qwen3.5-2B（冇 fine-tune，同一批 300 句）：daily+mixed 14.6% → **15.6%**、MDCC 7.6% → **8.4%**、SpiCE 38.1% → **40.3%**，
每句約 55–96ms。比 0.8B 好少少（識改「青台→青苔」、「hong kong→香港」），但仍然全部變差（翻譯英文、刪字）。

## Gemma 4 E2B 直接轉錄（unsloth GGUF Q4_K_M + mmproj-F16，llama.cpp b11609 CUDA）

`scripts/eval_gemma4.py`：經 `llama-server`（`-np 4`）送錄音，關咗思考模式（`enable_thinking: false`），
提示要求用粵語口語字、英文保留英文。同 `llm_correct.py` 同一批 300 句。
（`llama-mtmd-cli` 喺 Windows 一入 audio 就 crash（0xC0000409），`llama-server` 冇事。）

| 測試集 | **Gemma 4 E2B** | learned_r4（同一批句子） |
|---|---|---|
| daily + mixed | **44.2%** | 14.6% |
| MDCC test | **52.6%** | 7.6% |
| SpiCE 對話 | **71.4%** | 38.1% |
| 速度 | 5–14× 即時（並行 4，每句約 0.3–0.4 秒） | 約 1000× |

- 廣東話好差而且慢：「蔡嘉儀香港電台製作」→「財加一香港電台再介紹」，有時仲會亂作句子；粵英夾雜嘅英文部分反而寫得幾好。
- 可能原因：Gemma 4 冇講明支援粵語；Q4 量化；llama.cpp audio 仲係實驗階段。完整精度版未測。
- Gemma 4 嘅 audio encoder 同 EmbeddingGemma 2 係同一類（`gemma4_audio`）——我哋 720 小時粵語數據 train 嘅細 reader 遠好過佢，
  即係**粵語數據先係關鍵**，唔係 decoder 大細。

Gemma 4 **E4B QAT**（`unsloth/gemma-4-E4B-it-qat-GGUF` UD-Q4_K_XL + mmproj-F16，同一批 300 句，並行 8）：
daily+mixed **42.5%**、MDCC **49.8%**、SpiCE **84.4%**；9–23× 即時。
比 E2B 只好 2–3 點，SpiCE 仲差咗（更易作句子）—— 問題唔係量化或者 model 大細，係 Gemma 4 本身唔識粵語。
並行：4 → 8 句快約一倍（每秒 1.1 → 2.3 句），8 → 12 句只快約 8%。

### Gemma 4 E4B QAT 做修正（`scripts/gemma4_correct.py`，每個測試集 50 句，並行 8）

| 測試集 | learned_r4 草稿 | 淨文字修正 | **聽聲 + 草稿** |
|---|---|---|---|
| daily + mixed | 14.0% | 13.8% | **12.4%** |
| MDCC test | 5.1% | 6.9% | 5.4% |
| SpiCE 對話 | 41.5% | 41.2% | 41.5% |
| 額外時間（每句） | | 約 320ms | 約 390ms |

- 「聽聲 + 草稿」（錄音同我哋轉錄一齊畀佢）比淨文字好：英文詞改得好（digestive→digestion、the addictive→the detective），
  亦改到啲同音字（佳級→階級、明字→名字）；但會刪字、改人名（怪啟和→怪佢）、改字（女兒→女仔）。
- 只係粵英夾雜嗰組有約 1.6 點改善；MDCC、SpiCE 冇改善。樣本只有 50 句，1–2 點差距可能係噪音。
- 每句多 0.3–0.4 秒（ASR 本身約 0.03 秒），**唔值得**；最多適合離線、唔介意慢、粵英夾雜多嘅情況。

### N-best rescoring（`scripts/rescore_nbest.py`，Qwen3.5-0.8B，beam 8，每個測試集 100 句：前 50 揀 λ／μ、後 50 報告）

beam search（4-gram）出 8 個候選，LLM 只喺候選之間揀（唔會自己加字／改人名）。揀咗 λ=0.2、μ=0。

| 測試集（後 50 句） | greedy | 4-gram | + Qwen 重排 | 候選上限（oracle） | 每句額外 |
|---|---|---|---|---|---|
| daily + mixed | 14.5% | 15.5% | 15.0% | 12.5% | 約 43ms |
| MDCC test | 7.6% | 6.5% | **6.1%** | 3.5% | 約 45ms |
| SpiCE 對話 | 39.1% | 37.8% | **37.2%** | 34.5% | 約 47ms |

- 改善 0.4–0.6 點（前 50 句 dev 只由 21.6% → 21.5%），可能係噪音；鍾意揀短啲嘅候選（會刪「problem」「game」）。
- 候選上限仲有 2–3 點空間，但 0.8B 冇 fine-tune 攞唔到。2B 未試。

## 第五輪：encoder 中間層 + 粵拼輔助 CTC（Colab A100）

**動機**：reader 一直用 audio_tower 最後一層（經 embed_audio 嘅 512 維）；中間層可能保留多啲聲調／音素資訊。
粵拼輔助 CTC：reader 第 3 層後面加一個粵拼（音節 + 聲調，ToJyutping 按上下文揀讀音）CTC head，
loss = 0.7 × 字 + 0.3 × 粵拼，推理唔用（零速度成本）。

audio_tower 嘅 hidden_states 有 13 個：[0] 入第 1 層之前、[1]–[11] 係第 1–11 層（1024 維）、[12] = 最後一層經 output_proj（1536 維）。

### 細規模對照（MDCC train 30 小時，同一批數據 train 4 個 reader，30 epoch，`scripts/round5.sh`）

多層 = 第 3／6／7／8／9／10／12 層 + 原本 512 維，每層各自 LayerNorm + 投射，再用 softmax 權重加埋（SUPERB 式）。

| 組合 | 驗證 CER | MDCC test greedy | MDCC test +LM | SpiCE VAD +LM |
|---|---|---|---|---|
| p_final（原本 512 維） | 11.45% | 11.7% | 8.1% | 50.6% |
| p_mix（多層） | 10.41% | **10.6%** | 7.5% | 50.2% |
| p_final_jy（原本 + 粵拼） | 11.46% | 11.5% | 8.1% | 50.2% |
| p_mix_jy（多層 + 粵拼） | **10.26%** | 10.8% | **7.2%** | **49.5%** |

學到嘅層權重（第 3／6／7／8／9／10／12 層／原本 512）：`0.10 / 0.21 / 0.12 / 0.08 / 0.14 / 0.12 / 0.15 / 0.08`
（p_mix 同 p_mix_jy 幾乎一樣）。

- **多層有效**：MDCC test greedy −1.1 點、+LM −0.6 點（相對約 9%）。**第 6 層權重最高，原本用緊嘅 512 維最低。**
- **粵拼單獨用幾乎冇效**（0–0.3 點，噪音範圍）；同多層一齊用 +LM 最好（7.2%）。
- SpiCE 只 train 有聲書，所以絕對值好差（約 50%），只睇相對：改善 0.4–1 點。
- 成本：多層 feature 大好多（30 小時 42GB vs 原本約 2.8GB）；全數據要用學到嘅權重預先加埋先抽得起。

### 全數據 + 粵拼（learned_r5jy：720 小時、12 epoch、粵拼 0.3、只用原本 512 維）

| | learned_r4 | **learned_r5jy** | SenseVoice GPU |
|---|---|---|---|
| 驗證 CER（最好） | 8.60% | **8.16%** | |
| 原本 test greedy ／ +LM | 12.21% ／ 11.57% | **11.67% ／ 11.51%** | 9.7% |
| MDCC test greedy ／ +LM | 6.6% ／ 5.7% | **6.2% ／ 5.2%** | 7.9% |
| SpiCE 對話（VAD）greedy ／ +LM | 38.0% ／ 38.5% | **37.1% ／ 37.2%** | 28.0% |

- 全部測試集都好 0.4–0.9 點（greedy），推理速度一樣（粵拼 head 推理唔用）。
- ⚠️ 唔係純粹粵拼嘅功勞：r5jy 用預先載入 RAM（`fastdata.Preloaded`，全域按長度分 batch，每 epoch 1046 步），
  r4 用逐個 shard 讀（每 epoch 約 1173 步），batch 組成唔同。細規模對照（同一種 batch）粵拼單獨只好 0–0.3 點，
  所以呢度嘅改善可能有一部分嚟自 batch 方式。
- A100 train 約 54 分鐘（r4 約 1 小時 40 分）——預先載入 RAM 快咗差唔多一倍。

### 加試：用邊幾層先夠？（同一批 MDCC 30 小時）

| 組合 | 驗證 CER | MDCC test greedy ／ +LM | SpiCE VAD greedy ／ +LM | 層權重 |
|---|---|---|---|---|
| 原本 512（p_final） | 11.45% | 11.7% ／ 8.1% | 56.0% ／ 50.6% | |
| 只用第 6 層 | 11.08% | 10.9% ／ 7.6% | 59.3% ／ **53.3%** | |
| 第 6 層 + 原本 | 10.78% | 10.6% ／ 7.6% | 56.9% ／ 50.8% | 0.56 ／ 0.44 |
| 第 6／9／12 層 + 原本 | **10.35%** | 10.7% ／ **7.5%** | **54.7% ／ 49.4%** | 0.37 ／ 0.26 ／ 0.23 ／ 0.13 |
| 8 層（p_mix） | 10.41% | 10.6% ／ 7.5% | 55.7% ／ 50.2% | |

- 讀稿類（MDCC）加第 6 層就攞到大部分改善；但**只用第 6 層喺對話（SpiCE）反而差咗 2.7 點**——中間層唔夠穩陣，要同後面幾層一齊用。
- **4 層（6／9／12／原本）同 8 層一樣好**，SpiCE 仲最好；係全數據版嘅最佳候選。
- 問題係大小：4 層 = 1024 + 1024 + 1536 + 512 = 4096 維，係而家 8 倍（720 小時約 500GB），A100 RAM（83GB）預先載入唔到。
  要先壓縮：例如每層 PCA 落 256 維（共 1024 維，fp8 儲存約 63GB，同而家一樣大），要再驗證壓縮後效果唔跌。

**成本**：第五輪（包括 VM 斷線重做）共約 22 units，餘額 152.3。

## 第六輪（準備）：4 層 feature 用 PCA + fp8 壓縮（MDCC 30 小時對照）

每層（第 6／9／12 層 + 原本）各自 PCA 落 256 維（共 1024 維），用 float8_e4m3fn 存——每格大小同原本 512 維 fp16 一樣。
PCA 喺 MDCC 30 小時 200 萬格上面計：保留變異 第 6 層 91.4%、第 9 層 92.2%、第 12 層 98.0%、原本 99.1%。

| | 大小（30 小時） | 驗證 CER | MDCC test greedy | SpiCE VAD greedy |
|---|---|---|---|---|
| p_raw4（原始 4 層，4096 維 fp16） | 21GB | 10.52% | 10.6% | 55.6% |
| **p_pca4（PCA 256×4，fp8）** | **2.6GB** | 10.55% | **10.5%** | **55.2%** |

- **壓縮冇蝕**（大 8 倍細），720 小時預計約 63GB，A100 RAM 預先載入得落。
- 呢次冇 Drive，字典用 MDCC 30 小時自己建、冇 LM，所以數字同第五輪對照組唔可以直接比（只比 raw vs pca）。
- 下一步：`scripts/round6.sh` B 部分——5 個數據集用 PCA + fp8 重抽（radio 要 `hf auth login`），train learned_r6（4 層 + 粵拼）。
  要用 `data/pca_6_9_12_0.pt`（喺 VM 計嘅，已隨 VM 刪除；重跑 round6.sh 會由 MDCC 30 小時重新計，約 20 分鐘）。

**成本**：壓縮測試約 3 units，餘額 149.4。

### 全數據：learned_r6（720 小時，4 層 PCA 256×4 + fp8，粵拼 0.3，12 epoch）

同第四／五輪一樣嘅數據（5 個數據集用 PCA + fp8 重抽，`scripts/round6.sh`）；A100 train 約 58 分鐘。
壓縮測試（vocab_r4、有 LM）：MDCC +LM 原始 7.4% vs 壓縮 7.4%；SpiCE +LM 49.8% vs 48.8%——壓縮冇蝕。

| | learned_r4 | learned_r5jy | **learned_r6** | SenseVoice GPU |
|---|---|---|---|---|
| 驗證 CER（最好） | 8.60% | 8.16% | **7.90%** | |
| 原本 test greedy ／ +LM | 12.21% ／ 11.57% | 11.67% ／ 11.51% | **11.51% ／ 11.13%** | 9.7% |
| MDCC test greedy ／ +LM | 6.6% ／ 5.7% | 6.2% ／ 5.2% | **5.8% ／ 4.9%** | 7.9% |
| SpiCE 對話（VAD）greedy ／ +LM | 38.0% ／ 38.5% | 37.1% ／ 37.2% | **36.3% ／ 36.8%** | 28.0% |

- r5jy → r6（同樣 batch 方式、同樣粵拼，只差 4 層 feature）：全部再好 0.2–0.8 點；MDCC +LM 首次低過 5%。
- 學到嘅層權重（第 6／9／12 層／原本）：`0.31 / 0.26 / 0.22 / 0.21`。
- 推理：Encoder.frames 多攞 3 層 hidden state 再做 PCA（幾個矩陣乘法），速度差唔多。

### 全數據加碼：learned_r6b（1,238 小時 = r6 + 518 小時電台，同樣設定）

多抽 3 條電台流程（parquet offset 2／6／10，同之前 0／4／8 唔重疊）：171 + 175 + 172 小時。
107GB fp8 feature 大過 A100 RAM（83GB），所以用 `train.py --mmap`（分組讀 + 預讀下一組）；
Colab 磁碟每秒約 150–200MB，每步約 0.63 秒（預先載入約 0.24 秒），12 epoch（21,660 步）約 3.8 小時。

| | learned_r6（720h） | **learned_r6b（1,238h）** | SenseVoice GPU |
|---|---|---|---|
| 驗證 CER（最好） | 7.90% | **6.38%** | |
| 原本 test greedy ／ +LM | 11.51% ／ 11.13% | **10.59%** ／ 10.87% | 9.7% |
| MDCC test greedy ／ +LM | 5.8% ／ 4.9% | **5.3% ／ 4.7%** | 7.9% |
| SpiCE 對話（VAD）greedy ／ +LM | 36.3% ／ 36.8% | **34.1%** ／ 35.4% | 28.0% |

- 多 518 小時：驗證 −1.5 點、對話 −2.2 點——**數據仲係最有效**。
- 4-gram LM 開始拖後腿（原本 test、SpiCE 加 LM 反而差）；只有 MDCC（LM 用 MDCC train 字幕整）仲有幫助。建議用 greedy。
- 本機（RTX 5070 Ti）MDCC 2000 句 greedy 5.25%，約 750–850× 即時。
- 已上載 Hugging Face：`kuchris/embeddinggemma2asr-cantonese`。

**成本**：第六輪（壓縮測試 + 重抽 1,238 小時 + r6 + r6b，包括兩次重開 r6b）約 51 units，餘額 98.2。

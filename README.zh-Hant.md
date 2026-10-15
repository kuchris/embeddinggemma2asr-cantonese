# gemma2asr

[English](README.md) | **繁體中文**

以**凍結**的 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 音訊編碼器，加上一個小型 CTC 讀取器，
實現快速的粵語語音辨識。靈感來自只支援英文的 [Embedding Gemma Transcribe](https://eg2.haku.sh/)。

**最新模型 learned_r6**（2,080 萬參數，720 小時訓練資料）：在 RTX 5070 Ti 上約為**即時速度的 850 倍**，
在同一張 GPU 上比 SenseVoice-Small 快約 3 倍。

| 測試集（貪婪解碼／加 4-gram 語言模型） | learned_r6 | SenseVoice-Small |
|---|---|---|
| 朗讀語音及粵英夾雜 | 11.5%／11.1% | **9.7%** |
| MDCC（有聲書） | 5.8%／**4.9%** | 7.9% |
| SpiCE（自然對話） | 36.3%／36.8% | **28.0%** |

數值為字錯誤率（CER），越低越好。MDCC 與本模型的訓練資料屬同類，但對 SenseVoice 而言是零樣本測試。

- 專案報告：[REPORT.md](REPORT.md)（粵語）
- 各次實驗的詳細數據：[RESULTS.md](RESULTS.md)（粵語）

## 原理

```
16 kHz 音訊 → audio_tower（凍結）→ 取第 6／9／12 層及 embed_audio 輸出，每層以 PCA 壓縮至 256 維
           → 加權求和（權重可學習）→ 讀取器（6 層 Transformer 加 depthwise 卷積）
           → CTC 輸出中文字及英文詞
```

只訓練讀取器。編碼器只需執行一次，其特徵預先存成分片（`extract.py --layers --pca --fp8`）。
訓練時另加粵拼 CTC 輔助輸出（`train.py --jy-weight 0.3`），推論時不使用。

## 使用方法

```bash
uv sync                    # 推論
uv sync --extra train      # 訓練（另需 ToJyutping）
```

轉錄。模型路徑可以是 `scripts/export_hf.py` 匯出的資料夾（`model.safetensors` 及 `config.json`），
也可以是訓練產生的 `.pt` 檢查點。

```bash
uv run python scripts/transcribe.py hf/embeddinggemma2asr-cantonese samples/*.wav
uv run --with sounddevice python scripts/live.py --ckpt hf/embeddinggemma2asr-cantonese
```

```python
from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.text import join_units

enc = Encoder()
model, units = load_reader("hf/embeddinggemma2asr-cantonese")
print(join_units(transcribe_batch(enc, model, units, [wav_16k])[0]))   # 單聲道 16 kHz，每段不超過 30 秒
```

訓練（Colab A100；完整流程見 `scripts/round6.sh`）：

```bash
python scripts/extract.py --dataset mdcc --out data/feats/mdcc_l4 --max-hours 30 --layers 6,9,12,0
python scripts/fit_pca.py data/feats/mdcc_l4 --layers 6,9,12,0 --k 256 --out data/pca_6_9_12_0.pt
python scripts/extract.py --dataset mdcc --out data/feats6/mdcc --layers 6,9,12,0 --pca data/pca_6_9_12_0.pt --fp8
python scripts/train.py data/feats6/* --vocab data/vocab_r4.pt --out runs/learned_r6 --head learned --jy-weight 0.3
python scripts/export_hf.py runs/learned_r6/best.pt --out hf/embeddinggemma2asr-cantonese
```

若訓練資料超出記憶體容量，可使用 `train.py --mmap`。此模式會分組從磁碟讀取分片，並預先讀取下一組；
但在 Colab 的磁碟上，速度仍比全部載入記憶體慢 2 至 3 倍。

## 資料

| 名稱 | 來源 | 時數 | 授權（資料集頁面所載） |
|---|---|---|---|
| `wordshk` | AlienKevin/wordshk_cantonese_speech | 104 | CC BY 4.0 |
| `mixed` | AlienKevin/mixed_cantonese_and_english_speech | 34 | 未註明 |
| `daily` | ziyou-li/cantonese_daily | 4 | CC BY-NC-ND 4.0 |
| `mdcc` | ming030890/mdcc | 57 | HF 頁面未註明 |
| `radio_*` | alvanlii/cantonese-radio（需申請存取，並執行 `hf auth login`） | 519（另加 500） | 未註明；內容為電台節目 |

電台資料只保留資料集內 Whisper 與 SenseVoice 兩份轉錄相符（CER 低於 25%）的語句，並以 SenseVoice 的轉錄作為標籤。
測試集：MDCC test 及 [SpiCE](https://spice-corpus.readthedocs.io/)（CC BY 4.0）。

由於部分訓練資料限非商業用途或未註明授權，模型權重建議僅作研究及非商業用途。

所有模型快取均存放於 `models/`（可用 `G2ASR_MODELS` 更改）。

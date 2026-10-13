# gemma2asr

用凍結嘅 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio encoder，加一個細 reader 做語音辨識（主力粵語，順便中文／英文）。
靈感嚟自 [Embedding Gemma Transcribe](https://eg2.haku.sh/)（佢只做英文，Mac 專用）。

## 原理

```
16kHz 音頻 → audio_tower + embed_audio（凍結）→ 每 40ms 一格 512 維 feature
           → reader（約 11M 參數，要 train）→ 每格 768 維向量
           → 同字典每個字嘅 text embedding（凍結）比 cosine → CTC
```

中文逐個字做單位，英文逐個詞。只有 reader 會 train，encoder 只係預先行一次，將 feature 存落 shard。

## Spike 0 結果（本機 RTX 5070 Ti）

`scripts/spike0.py`：12 段錄音（粵／普／英／日各 3 段）

| 測試 | 冇 prompt | `Document` prompt |
|---|---|---|
| 成段錄音 → 揀返啱嗰句（top-1） | 9/12 | 11/12 |
| 1 秒窗口 → 最近 5 個字有講緊嘅字 | 90% | 97%（隨機 45%） |

英文最清楚，1 秒窗口幾乎逐個詞對得上；粵語最弱，會偏向「唔、冇、哋」呢啲常見字。

## 用法

Colab：打開 `notebooks/train_colab.ipynb`，由頭行到尾。

本機（uv）：

```bash
uv sync
uv run python scripts/spike0.py
uv run python scripts/extract.py --dataset daily --out data/feats/daily
uv run python scripts/build_vocab.py data/feats/daily --out data/vocab.pt
uv run python scripts/train.py data/feats/daily --vocab data/vocab.pt --out runs/frozen
uv run python scripts/transcribe.py runs/frozen/best.pt samples/*.wav
```

`train.py --head`：`frozen`（原本方法）、`tuned`（字典 embedding 一齊 train）、`learned`（普通 CTC 分類頭，做對照組）。

## 數據

| 名 | 來源 | 備註 |
|---|---|---|
| `wordshk` | AlienKevin/wordshk_cantonese_speech | 約 11.5 萬句，words.hk 例句同釋義 |
| `mixed` | AlienKevin/mixed_cantonese_and_english_speech | 約 1.4 萬句粵英夾雜 |
| `daily` | ziyou-li/cantonese_daily | 約 4 千句日常粵語 |

直接讀 Hugging Face 自動轉好嘅 parquet，唔使 `datasets`／torchcodec。
`samples/` 係 Gaijincall 嘅測試錄音，參考字幕由 SenseVoice 出（未人手校對）。

所有模型 cache 放 `models/`（Colab 用 `G2ASR_MODELS` 指去 Drive）。

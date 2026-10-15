# gemma2asr

用凍結嘅 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio encoder，加一個細 CTC reader 做廣東話語音辨識。
靈感嚟自 [Embedding Gemma Transcribe](https://eg2.haku.sh/)（佢只做英文）。

**最新版本 learned_r6**（2,080 萬參數，720 小時數據）：RTX 5070 Ti 上約 **850 倍即時**（SenseVoice-Small 約 3 倍慢）。

| 測試集（greedy ／ +LM） | learned_r6 | SenseVoice-Small |
|---|---|---|
| 讀稿 + 粵英夾雜 | 11.5% ／ 11.1% | **9.7%** |
| MDCC 有聲書 | 5.8% ／ **4.9%** | 7.9% |
| SpiCE 自然對話 | 36.3% ／ 36.8% | **28.0%** |

- 總結：[REPORT.md](REPORT.md)
- 每次實驗嘅詳細數字：[RESULTS.md](RESULTS.md)

## 原理

```
16kHz 音頻 → audio_tower（凍結）→ 攞第 6／9／12 層 + embed_audio 輸出，每層 PCA 256 維
           → 權重加埋 → reader（6 層 Transformer + depthwise conv）→ CTC（中文逐字、英文逐詞）
```

只有 reader 會 train。encoder 預先行一次，將 feature 存落 shard（`extract.py --layers --pca --fp8`）。
train 時加粵拼 CTC 輔助 head（`train.py --jy-weight 0.3`），推理唔用。

## 用法

```bash
uv sync                    # 推理
uv sync --extra train      # train（多咗 ToJyutping）
```

轉錄（model 資料夾 = `scripts/export_hf.py` 出嘅 `model.safetensors` + `config.json`，或者 train 出嚟嘅 `.pt`）：

```bash
uv run python scripts/transcribe.py hf/gemma2asr-yue samples/*.wav
uv run --with sounddevice python scripts/live.py --ckpt hf/gemma2asr-yue
```

```python
from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.text import join_units

enc = Encoder()
model, units = load_reader("hf/gemma2asr-yue")
print(join_units(transcribe_batch(enc, model, units, [wav_16k])[0]))
```

Train（Colab A100，見 `scripts/round6.sh`）：

```bash
python scripts/extract.py --dataset mdcc --out data/feats/mdcc_l4 --max-hours 30 --layers 6,9,12,0
python scripts/fit_pca.py data/feats/mdcc_l4 --layers 6,9,12,0 --k 256 --out data/pca_6_9_12_0.pt
python scripts/extract.py --dataset mdcc --out data/feats6/mdcc --layers 6,9,12,0 --pca data/pca_6_9_12_0.pt --fp8
python scripts/train.py data/feats6/* --vocab data/vocab_r4.pt --out runs/learned_r6 --head learned --jy-weight 0.3
python scripts/export_hf.py runs/learned_r6/best.pt --out hf/gemma2asr-yue
```

數據大過 RAM：`train.py --mmap`（分組讀 + 預讀；但 Colab 磁碟慢，會慢 2–3 倍）。

## 數據

| 名 | 來源 | 小時 | 授權（數據集頁面） |
|---|---|---|---|
| `wordshk` | AlienKevin/wordshk_cantonese_speech | 104 | CC BY 4.0 |
| `mixed` | AlienKevin/mixed_cantonese_and_english_speech | 34 | 冇寫 |
| `daily` | ziyou-li/cantonese_daily | 4 | CC BY-NC-ND 4.0 |
| `mdcc` | ming030890/mdcc | 57 | HF 頁面冇寫 |
| `radio_*` | alvanlii/cantonese-radio（gated，要 `hf auth login`） | 519（+500） | 冇寫；電台節目 |

電台數據用數據集入面 Whisper 同 SenseVoice 兩個轉錄互相驗證（CER < 25%），用 SenseVoice 版本做標籤。
測試：MDCC test、[SpiCE](https://spice-corpus.readthedocs.io/)（CC BY 4.0）。

因為部分數據係非商業或者冇註明授權，model 權重只建議作研究／非商業用途。

所有模型 cache 放 `models/`（`G2ASR_MODELS` 可以改）。

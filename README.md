# embeddinggemma2asr-cantonese

**English** | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md)

Fast Cantonese speech recognition from the **frozen** audio encoder of
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), plus a small CTC reader.
Inspired by [Embedding Gemma Transcribe](https://eg2.haku.sh/), which does the same for English.

**Latest model, learned_r6b** (20.8M parameters, 1,238 hours of training data): roughly **750–850× real time** on an
RTX 5070 Ti, about 3× faster than SenseVoice-Small on the same GPU.
Weights: [kuchris/embeddinggemma2asr-cantonese](https://huggingface.co/kuchris/embeddinggemma2asr-cantonese).

| Test set (greedy / with 4-gram LM) | learned_r6b | SenseVoice-Small |
|---|---|---|
| Read speech + Cantonese–English code-switching | 10.8% / **9.6%** | 9.7% |
| MDCC (audiobooks) | 5.3% / **4.5%** | 7.9% |
| SpiCE (spontaneous conversation) | 34.1% / 33.3% | **28.0%** |

Character error rate; lower is better. MDCC is in-domain for our model but zero-shot for SenseVoice.
The 4-gram LM (α=0.3, β=2.5, re-tuned for r6b) helps on every test set by about 1 point; greedy is faster.
α/β were picked on one half of these test sets; on the other half alone the gain is the same (RESULTS.md).

- Project report: [REPORT.md](REPORT.md) (Cantonese)
- Detailed numbers for every experiment: [RESULTS.md](RESULTS.md) (Cantonese)

## How it works

```
16 kHz audio → audio_tower (frozen) → layers 6 / 9 / 12 + embed_audio output, each PCA-compressed to 256 dims
             → learned weighted sum → reader (6-layer Transformer + depthwise conv)
             → CTC over Chinese characters and English words
```

Only the reader is trained. The encoder runs once and its features are stored as shards
(`extract.py --layers --pca --fp8`). Training adds an auxiliary Jyutping CTC head (`train.py --jy-weight 0.3`),
which is not used at inference.

## Usage

```bash
uv sync                    # inference
uv sync --extra train      # training (adds ToJyutping)
```

Transcribe. The model path can be a folder from `scripts/export_hf.py` (`model.safetensors` + `config.json`) or a
training `.pt` checkpoint.

```bash
uv run python scripts/transcribe.py hf/embeddinggemma2asr-cantonese samples/*.wav
uv run --with sounddevice python scripts/live.py --ckpt hf/embeddinggemma2asr-cantonese
uv run --with sounddevice --with soundcard python scripts/live.py --ckpt hf/embeddinggemma2asr-cantonese --loopback  # what the computer is playing (Windows)
```

```python
from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.text import join_units

enc = Encoder()
model, units = load_reader("hf/embeddinggemma2asr-cantonese")
print(join_units(transcribe_batch(enc, model, units, [wav_16k])[0]))   # mono 16 kHz, ≤ 30 s
```

Train (Colab A100; see `scripts/round6.sh` for the full pipeline):

```bash
python scripts/extract.py --dataset mdcc --out data/feats/mdcc_l4 --max-hours 30 --layers 6,9,12,0
python scripts/fit_pca.py data/feats/mdcc_l4 --layers 6,9,12,0 --k 256 --out data/pca_6_9_12_0.pt
python scripts/extract.py --dataset mdcc --out data/feats6/mdcc --layers 6,9,12,0 --pca data/pca_6_9_12_0.pt --fp8
python scripts/train.py data/feats6/* --vocab data/vocab_r4.pt --out runs/learned_r6b --head learned --jy-weight 0.3 --mmap
python scripts/export_hf.py runs/learned_r6b/best.pt --out hf/embeddinggemma2asr-cantonese
```

If the training data does not fit in RAM, use `train.py --mmap`. It reads shards from disk in groups and prefetches
the next group, but on Colab's disk it is still 2–3× slower than loading everything into RAM.

## Data

| Name | Source | Hours | License (as stated on the dataset page) |
|---|---|---|---|
| `wordshk` | AlienKevin/wordshk_cantonese_speech | 104 | CC BY 4.0 |
| `mixed` | AlienKevin/mixed_cantonese_and_english_speech | 34 | not stated |
| `daily` | ziyou-li/cantonese_daily | 4 | CC BY-NC-ND 4.0 |
| `mdcc` | ming030890/mdcc | 57 | not stated on the HF page |
| `radio_*` | alvanlii/cantonese-radio (gated; needs `hf auth login`) | 1,037 | not stated; radio broadcasts |

For the radio data, an utterance is kept only when the dataset's Whisper and SenseVoice transcripts agree
(CER < 25%); the SenseVoice transcript is used as the label.
Test sets: MDCC test and [SpiCE](https://spice-corpus.readthedocs.io/) (CC BY 4.0).

Because some training data is non-commercial or has no stated license, the model weights are recommended for
research and non-commercial use only.

## License

Code: [MIT](LICENSE). Model weights: research and non-commercial use only (see above).

All model caches are stored under `models/` (override with `G2ASR_MODELS`).

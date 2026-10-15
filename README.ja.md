# embeddinggemma2asr-cantonese

[English](README.md) | [繁體中文](README.zh-Hant.md) | **日本語**

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) の音声エンコーダを**凍結**したまま使い、
小さな CTC リーダーを追加した高速な広東語音声認識です。
英語で同じことを行う [Embedding Gemma Transcribe](https://eg2.haku.sh/) に着想を得ています。

**最新モデル learned_r6b**（2,080 万パラメータ、学習データ 1,238 時間）：RTX 5070 Ti で**実時間の約 750〜850 倍**の速度で、
同じ GPU 上の SenseVoice-Small より約 3 倍高速です。
重み：[kuchris/embeddinggemma2asr-cantonese](https://huggingface.co/kuchris/embeddinggemma2asr-cantonese)

| テストセット（貪欲デコード／4-gram 言語モデル併用） | learned_r6b | SenseVoice-Small |
|---|---|---|
| 朗読音声＋広東語・英語の混在 | 10.8%／**9.6%** | 9.7% |
| MDCC（オーディオブック） | 5.3%／**4.5%** | 7.9% |
| SpiCE（自然な会話） | 34.1%／33.3% | **28.0%** |

数値は文字誤り率（CER）で、低いほど良い結果です。MDCC は本モデルにとって学習データと同種ですが、SenseVoice にとってはゼロショットです。
4-gram 言語モデル（α=0.3、β=2.5、r6b 向けに再調整）はすべてのテストセットで誤りを約 1 ポイント減らします。貪欲デコードの方が高速です。
α／β はこれらのテストセットの半分で選び、残り半分だけで見ても改善幅は同じです（RESULTS.md 参照）。

- プロジェクト報告：[REPORT.md](REPORT.md)（広東語）
- 全実験の詳細な数値：[RESULTS.md](RESULTS.md)（広東語）

## 仕組み

```
16 kHz 音声 → audio_tower（凍結）→ 第 6／9／12 層と embed_audio の出力を取り出し、各層を PCA で 256 次元に圧縮
            → 学習可能な重みで加算 → リーダー（6 層 Transformer＋depthwise 畳み込み）
            → 中国語の文字と英単語を単位とする CTC
```

学習するのはリーダーのみです。エンコーダは一度だけ実行し、その特徴量をシャードとして保存します
（`extract.py --layers --pca --fp8`）。学習時には補助的な粵拼（Jyutping）CTC ヘッドを追加しますが
（`train.py --jy-weight 0.3`）、推論時には使用しません。

## 使い方

```bash
uv sync                    # 推論
uv sync --extra train      # 学習（ToJyutping を追加）
```

文字起こし。モデルのパスには、`scripts/export_hf.py` で出力したフォルダ（`model.safetensors` と `config.json`）と、
学習で得られた `.pt` チェックポイントのどちらも指定できます。

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
print(join_units(transcribe_batch(enc, model, units, [wav_16k])[0]))   # モノラル 16 kHz、1 区間 30 秒以内
```

学習（Colab A100。全体の手順は `scripts/round6.sh` を参照）：

```bash
python scripts/extract.py --dataset mdcc --out data/feats/mdcc_l4 --max-hours 30 --layers 6,9,12,0
python scripts/fit_pca.py data/feats/mdcc_l4 --layers 6,9,12,0 --k 256 --out data/pca_6_9_12_0.pt
python scripts/extract.py --dataset mdcc --out data/feats6/mdcc --layers 6,9,12,0 --pca data/pca_6_9_12_0.pt --fp8
python scripts/train.py data/feats6/* --vocab data/vocab_r4.pt --out runs/learned_r6b --head learned --jy-weight 0.3 --mmap
python scripts/export_hf.py runs/learned_r6b/best.pt --out hf/embeddinggemma2asr-cantonese
```

学習データがメモリに収まらない場合は `train.py --mmap` を使います。シャードをグループごとにディスクから読み込み、
次のグループを先読みしますが、Colab のディスクでは全データをメモリに載せる場合より 2〜3 倍遅くなります。

## データ

| 名前 | 出典 | 時間 | ライセンス（データセットのページの記載） |
|---|---|---|---|
| `wordshk` | AlienKevin/wordshk_cantonese_speech | 104 | CC BY 4.0 |
| `mixed` | AlienKevin/mixed_cantonese_and_english_speech | 34 | 記載なし |
| `daily` | ziyou-li/cantonese_daily | 4 | CC BY-NC-ND 4.0 |
| `mdcc` | ming030890/mdcc | 57 | HF のページに記載なし |
| `radio_*` | alvanlii/cantonese-radio（アクセス申請が必要。`hf auth login` を実行） | 1,037 | 記載なし。ラジオ番組 |

ラジオデータは、データセット内の Whisper と SenseVoice の 2 つの書き起こしが一致する（CER 25% 未満）発話のみを残し、
SenseVoice の書き起こしをラベルとして使用しています。
テストセット：MDCC test と [SpiCE](https://spice-corpus.readthedocs.io/)（CC BY 4.0）。

学習データの一部が非商用限定またはライセンス未記載であるため、モデルの重みは研究および非商用目的での利用のみを推奨します。

## ライセンス

コード：[MIT](LICENSE)。モデルの重み：研究および非商用目的のみ（上記参照）。

モデルのキャッシュはすべて `models/` に保存されます（`G2ASR_MODELS` で変更可能）。

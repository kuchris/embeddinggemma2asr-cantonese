"""數據集設定、音頻解碼，同埋 feature shard 讀寫。

Shard 格式（torch.save）：{"feats": [Tensor(T, 512) fp16], "units": [list[str]], "text": [str]}
"""

import io
import random
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfFileSystem

# 名 → (Hugging Face 數據集, split, 字幕欄)
DATASETS = {
    # words.hk 粵語讀音（大部分係短句同釋義，約 11.5 萬句）
    "wordshk": ("AlienKevin/wordshk_cantonese_speech", "train", "text"),
    # 粵英夾雜（約 1.4 萬句）
    "mixed": ("AlienKevin/mixed_cantonese_and_english_speech", "train", "sentence"),
    # 日常粵語（約 4 千句）
    "daily": ("ziyou-li/cantonese_daily", "train", "sentence"),
}


def iter_rows(repo: str, split: str, columns: list[str], config: str = "default"):
    """直接讀 Hugging Face 自動轉好嘅 parquet（refs/convert/parquet），唔經 datasets 嘅音頻解碼（佢要 torchcodec）。"""
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}@~parquet/{config}/{split}/*.parquet"))
    if not files:
        raise SystemExit(f"{repo} 冇 parquet 版本")
    for path in files:
        with fs.open(path) as f:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=columns):
                yield from batch.to_pylist()


def decode_audio(audio: dict, sr: int = 16000) -> np.ndarray:
    """datasets 嘅 Audio(decode=False) 欄 → 16kHz mono float32。"""
    if audio.get("bytes"):
        wav, file_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        wav, file_sr = sf.read(audio["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav


def shard_paths(dirs: list[str]) -> list[Path]:
    paths = []
    for d in dirs:
        paths += sorted(Path(d).glob("shard_*.pt"))
    if not paths:
        raise SystemExit(f"喺 {dirs} 搵唔到 shard_*.pt")
    return paths


def batches(paths: list[Path], unit_to_id: dict[str, int], max_frames: int, shuffle: bool):
    """逐個 shard 讀入（慳 RAM），按長度分 batch。每個 batch：(feats, lengths, targets, target_lengths, texts)。"""
    paths = list(paths)
    if shuffle:
        random.shuffle(paths)
    for path in paths:
        shard = torch.load(path)
        items = [(f, [unit_to_id[u] for u in us if u in unit_to_id], t)
                 for f, us, t in zip(shard["feats"], shard["units"], shard["text"])]
        items = [x for x in items if x[1] and len(x[0]) >= 2 * len(x[1])]  # CTC 要夠格數
        items.sort(key=lambda x: len(x[0]))
        groups, cur = [], []
        for item in items:
            if cur and (len(cur) + 1) * len(item[0]) > max_frames:
                groups.append(cur)
                cur = []
            cur.append(item)
        if cur:
            groups.append(cur)
        if shuffle:
            random.shuffle(groups)
        for group in groups:
            lengths = torch.tensor([len(f) for f, _, _ in group])
            feats = torch.zeros(len(group), lengths.max(), group[0][0].shape[1], dtype=torch.float16)
            for i, (f, _, _) in enumerate(group):
                feats[i, :len(f)] = f
            targets = torch.tensor([u for _, us, _ in group for u in us])
            target_lengths = torch.tensor([len(us) for _, us, _ in group])
            yield feats, lengths, targets, target_lengths, [t for _, _, t in group]

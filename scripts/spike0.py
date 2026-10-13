"""Spike 0：EmbeddingGemma 2 嘅 audio 同文字，係咪真係落喺同一個空間？

A. 成段：12 段錄音（粵／普／英／日）對 12 句參考字幕，睇吓啱嗰句係咪排第一。
B. 逐段：用 1 秒窗口滑過每段錄音，同所有參考字幕入面嘅字比，睇吓最近嗰幾個字係咪真係講緊嘅字。
   如果 B 完全唔掂，reader 要由零學起，呢個方法就冇乜優勢。

用法：uv run python scripts/spike0.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import soundfile as sf
import torch

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.text import units

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PROMPTS = {"冇 prompt": "", "Document": "title: none | text: "}
WINDOW, HOP, TOPK = 1.0, 0.25, 5

refs = json.loads((SAMPLES / "refs.json").read_text(encoding="utf-8"))
names = list(refs)
wavs = []
for n in names:
    wav, sr = sf.read(SAMPLES / f"{n}.wav", dtype="float32")
    assert sr == SAMPLE_RATE, sr
    wavs.append(wav if wav.ndim == 1 else wav.mean(1))

enc = Encoder()
print(f"dtype={enc.dtype}")

# ---------- A. 成段檢索 ----------
audio = enc.embed_audio(wavs)
for label, prompt in PROMPTS.items():
    text = enc.embed_texts([refs[n]["text"] for n in names], prompt=prompt)
    sim = audio @ text.T
    hits = (sim.argmax(1) == torch.arange(len(names))).sum().item()
    diag = sim.diag().mean().item()
    off = (sim.sum() - sim.diag().sum()).item() / (len(names) ** 2 - len(names))
    print(f"\n[A｜{label}] 成段 top-1：{hits}/{len(names)}　啱嗰對平均相似度 {diag:.3f}，錯嗰對 {off:.3f}")
    for i, n in enumerate(names):
        j = sim[i].argmax().item()
        print(f"  {n:6s} → {names[j]:6s} {'✓' if i == j else '✗'}  {sim[i, i]:.3f}")

# ---------- B. 1 秒窗口對單字 ----------
vocab = sorted({u for n in names for u in units(refs[n]["text"], to_traditional=False)})
print(f"\n[B] 候選字 {len(vocab)} 個（所有參考字幕嘅字／詞），窗口 {WINDOW}s，步長 {HOP}s")
for label, prompt in PROMPTS.items():
    vemb = enc.embed_texts(vocab, prompt=prompt)
    hit_total = win_total = 0
    chance_total = 0.0
    lines = []
    for n, wav in zip(names, wavs):
        truth = set(units(refs[n]["text"], to_traditional=False))
        step, size = int(HOP * SAMPLE_RATE), int(WINDOW * SAMPLE_RATE)
        chunks = [wav[s:s + size] for s in range(0, max(1, len(wav) - size + 1), step)]
        top = (enc.embed_audio(chunks) @ vemb.T).topk(TOPK, dim=1).indices
        hits = sum(any(vocab[k] in truth for k in row.tolist()) for row in top)
        hit_total += hits
        win_total += len(chunks)
        # 隨機揀 TOPK 個字，至少一個喺呢句入面嘅機率
        p_miss = 1.0
        for k in range(TOPK):
            p_miss *= (len(vocab) - len(truth) - k) / (len(vocab) - k)
        chance_total += (1 - p_miss) * len(chunks)
        sample = " | ".join("".join(vocab[k] for k in row[:3].tolist()) for row in top[::4])
        lines.append(f"  {n:6s} {hits:2d}/{len(chunks):2d}  每秒 top-3：{sample}")
    print(f"\n[B｜{label}] top-{TOPK} 命中 {hit_total}/{win_total} = {hit_total / win_total:.0%}"
          f"（隨機大約 {chance_total / win_total:.0%}）")
    print("\n".join(lines))

"""即時聽寫：對住咪講嘢，每講完一句（停頓約半秒）就轉錄一次。Ctrl+C 離開。

用法：uv run --with sounddevice python scripts/live.py [--lm] [--ckpt runs/learned_r6b/best.pt]
      --file x.wav 可以用錄音檔模擬（測試用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import queue
import time
from collections import deque

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.lm import NGramLM
from g2asr.text import join_units
from g2asr.vad import MAX_SEC

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 512  # 32ms，silero 每次睇呢個長度
PRE_ROLL = 10  # 開始講嘢前保留 10 格（約 0.3 秒），費事切走第一個字

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r6b" / "best.pt"))
parser.add_argument("--lm", action="store_true", help="用 beam search + LM（準啲，慢少少）")
parser.add_argument("--lm-path", default=str(ROOT / "data" / "lm_r4.pkl"))
parser.add_argument("--silence", type=int, default=500, help="停頓幾多毫秒當一句完")
parser.add_argument("--file", help="用錄音檔模擬咪（測試用）")
args = parser.parse_args()

print("載入模型…")
enc = Encoder()
model, vocab = load_reader(args.ckpt)
lm = NGramLM.load(args.lm_path) if args.lm else None
transcribe_batch(enc, model, vocab, [np.zeros(SAMPLE_RATE, np.float32)], lm)  # 預熱
vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SAMPLE_RATE,
                  min_silence_duration_ms=args.silence, speech_pad_ms=100)

blocks: queue.Queue = queue.Queue()


def emit(audio: np.ndarray):
    t0 = time.time()
    text = join_units(transcribe_batch(enc, model, vocab, [audio], lm)[0])
    if text:
        print(f"[{len(audio) / SAMPLE_RATE:4.1f}s 句｜{(time.time() - t0) * 1000:3.0f}ms] {text}")


def run(source):
    recent = deque(maxlen=PRE_ROLL)
    speech: list[np.ndarray] | None = None
    for block in source:
        event = vad(torch.from_numpy(block))
        if speech is None:
            recent.append(block)
            if event and "start" in event:
                speech = list(recent)
        else:
            speech.append(block)
            if (event and "end" in event) or len(speech) * BLOCK >= MAX_SEC * SAMPLE_RATE:
                emit(np.concatenate(speech))
                speech = None
                recent.clear()
    if speech:
        emit(np.concatenate(speech))


if args.file:
    import librosa
    wav, _ = librosa.load(args.file, sr=SAMPLE_RATE, mono=True)
    run(wav[: len(wav) // BLOCK * BLOCK].astype(np.float32).reshape(-1, BLOCK))
else:
    import sounddevice as sd

    def from_mic():
        while True:
            yield blocks.get()

    print(f"咪：{sd.query_devices(kind='input')['name']}")
    print(f"開始講嘢啦（{'beam search + LM' if lm else 'greedy'}）… Ctrl+C 離開\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                        callback=lambda data, *_: blocks.put(data[:, 0].copy())):
        try:
            run(from_mic())
        except KeyboardInterrupt:
            print("\n拜拜")

"""即時聽寫：對住咪講嘢，講緊嘅時候每 0.5 秒用 greedy 更新同一行，講完一句（停頓約半秒）先出最終版本。Ctrl+C 離開。

用法：uv run --with sounddevice python scripts/live.py [--lm] [--ckpt runs/learned_r6b/best.pt]
      --file x.wav 可以用錄音檔模擬（測試用）
      --loopback 轉錄電腦播緊嘅聲（預設喇叭，Windows WASAPI），要用 uv run --with soundcard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import queue
import shutil
import threading
import time
import unicodedata
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
LOOPBACK_MAX_SEC = 10  # 片段成日冇停頓、有背景音樂，VAD 唔會切；最多 10 秒就出一次字

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r6b" / "best.pt"))
parser.add_argument("--lm", action="store_true", help="用 beam search + LM（準啲，慢少少）")
parser.add_argument("--lm-path", default=str(ROOT / "data" / "lm_r4.pkl"))
parser.add_argument("--silence", type=int, default=500, help="停頓幾多毫秒當一句完")
parser.add_argument("--file", help="用錄音檔模擬咪（測試用）")
parser.add_argument("--loopback", action="store_true", help="唔用咪，轉錄電腦喇叭播緊嘅聲")
parser.add_argument("--partial", type=float, default=0.5, help="講緊嘢嘅時候每幾多秒更新一次（0 = 一句完先出字）")
args = parser.parse_args()
max_sec = LOOPBACK_MAX_SEC if args.loopback else MAX_SEC
partial_every = 0 if args.file else round(args.partial * SAMPLE_RATE / BLOCK)  # 幾多格更新一次

print("載入模型…")
enc = Encoder()
model, vocab = load_reader(args.ckpt)
lm = NGramLM.load(args.lm_path) if args.lm else None
transcribe_batch(enc, model, vocab, [np.zeros(SAMPLE_RATE, np.float32)], lm)  # 預熱
vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SAMPLE_RATE,
                  min_silence_duration_ms=args.silence, speech_pad_ms=100)

blocks: queue.Queue = queue.Queue()


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def show_partial(audio: np.ndarray):
    """講緊嘢：greedy 轉錄到而家為止嘅部分，覆蓋同一行。"""
    text = join_units(transcribe_batch(enc, model, vocab, [audio])[0])
    room = shutil.get_terminal_size().columns - 6
    while width(text) > room:  # 太長就淨係顯示最尾，換咗行就覆蓋唔到
        text = text[1:]
    print(f"\r\033[K  … {text}", end="", flush=True)


def emit(audio: np.ndarray):
    t0 = time.time()
    text = join_units(transcribe_batch(enc, model, vocab, [audio], lm)[0])
    print("\r\033[K", end="")  # 清走即時更新嗰行
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
            if (event and "end" in event) or len(speech) * BLOCK >= max_sec * SAMPLE_RATE:
                emit(np.concatenate(speech))
                speech = None
                recent.clear()
            elif partial_every and len(speech) % partial_every == 0 and blocks.qsize() < partial_every:
                show_partial(np.concatenate(speech))  # 追唔切（隊列積壓）就跳過今次
    if speech:
        emit(np.concatenate(speech))


if args.file:
    import librosa
    wav, _ = librosa.load(args.file, sr=SAMPLE_RATE, mono=True)
    run(wav[: len(wav) // BLOCK * BLOCK].astype(np.float32).reshape(-1, BLOCK))
elif args.loopback:
    import warnings

    import soundcard as sc

    # 冇聲播嘅時候 Windows 唔送數據，soundcard 會用靜音補返兼出警告；冇害
    warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
    speaker = sc.default_speaker()
    loop = sc.get_microphone(str(speaker.name), include_loopback=True)

    def record():
        # 另一條 thread 錄音，轉錄嗰陣都唔會漏聲；多聲道平均做單聲道
        with loop.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
            while True:
                blocks.put(rec.record(numframes=BLOCK).mean(axis=1).astype(np.float32))

    def from_speaker():
        while True:
            yield blocks.get()

    threading.Thread(target=record, daemon=True).start()
    print(f"錄電腦聲：{speaker.name}")
    print(f"開始播啦（{'beam search + LM' if lm else 'greedy'}）… Ctrl+C 離開\n")
    try:
        run(from_speaker())
    except KeyboardInterrupt:
        print("\n拜拜")
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

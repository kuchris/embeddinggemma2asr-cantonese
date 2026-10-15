"""即時聽寫：對住咪講嘢，講緊嘅時候每 0.5 秒用 greedy 更新同一行，講完一句（停頓約半秒）先出最終版本。Ctrl+C 離開。

用法：uv run --with sounddevice python scripts/live.py [--lm] [--ckpt runs/learned_r6b/best.pt]
      --file x.wav 可以用錄音檔模擬（測試用）
      --loopback 轉錄電腦播緊嘅聲（預設喇叭，Windows WASAPI），要用 uv run --with soundcard
      --plain 淨係出字（冇時間、唔分行），--vad 0.3 調低 VAD 門檻
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
from rapidfuzz.distance import Levenshtein
from silero_vad import VADIterator, load_silero_vad

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.lm import NGramLM
from g2asr.text import LATIN, join_units
from g2asr.vad import MAX_SEC

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 512  # 32ms，silero 每次睇呢個長度
PRE_ROLL = 10  # 開始講嘢前保留 10 格（約 0.3 秒），費事切走第一個字
LOOPBACK_MAX_SEC = 10  # 片段成日冇停頓、有背景音樂，VAD 唔會切；最多 10 秒就出一次字

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r6b" / "best.pt"))
parser.add_argument("--lm", action="store_true", help="用 beam search + LM（準啲，慢少少）")
parser.add_argument("--lm-path", default=str(ROOT / "data" / "lm_r4.pkl"))
parser.add_argument("--silence", type=int, help="停頓幾多毫秒當一句完（預設：咪 500、--loopback 300）")
parser.add_argument("--file", help="用錄音檔模擬咪（測試用）")
parser.add_argument("--loopback", action="store_true", help="唔用咪，轉錄電腦喇叭播緊嘅聲")
parser.add_argument("--partial", type=float, help="講緊嘢嘅時候每幾多秒更新一次，0 = 一句完先出字（預設：咪 0.5、--loopback 0.25）")
parser.add_argument("--vad", type=float, help="VAD 門檻，越低越易當係講嘢（預設：咪 0.5、--loopback 0.3）")
parser.add_argument("--plain", action="store_true", help="淨係出字：唔顯示時間，一句接一句連續寫落去")
args = parser.parse_args()
max_sec = LOOPBACK_MAX_SEC if args.loopback else MAX_SEC
vad_threshold = args.vad if args.vad is not None else 0.3 if args.loopback else 0.5  # 片段成日有背景音樂
# 片段講得快、少停頓：loopback 更新密啲、停短啲就當一句完
silence_ms = args.silence if args.silence is not None else 300 if args.loopback else 500
partial_sec = args.partial if args.partial is not None else 0.25 if args.loopback else 0.5
partial_every = 0 if args.file else round(partial_sec * SAMPLE_RATE / BLOCK)  # 幾多格更新一次

print("載入模型…")
enc = Encoder()
model, vocab = load_reader(args.ckpt)
lm = NGramLM.load(args.lm_path) if args.lm else None
transcribe_batch(enc, model, vocab, [np.zeros(SAMPLE_RATE, np.float32)], lm)  # 預熱
vad = VADIterator(load_silero_vad(), threshold=vad_threshold, sampling_rate=SAMPLE_RATE,
                  min_silence_duration_ms=silence_ms, speech_pad_ms=100)

blocks: queue.Queue = queue.Queue()


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def tail(text: str, room: int) -> str:
    """太長就淨係留最尾：換咗行就覆蓋唔到。"""
    while width(text) > room:
        text = text[1:]
    return text


def fit(text: str, room: int) -> str:
    """由頭開始留，留到夠位為止。"""
    out = ""
    for c in text:
        if width(out + c) > room:
            break
        out += c
    return out


# --plain：定咗嘅字只寫一次（之後唔再郁，交畀終端機自己換行）；淨係行尾未定嘅字會覆蓋，
# 用 ESC 7／ESC 8（記住／返去游標位置）+ ESC[J（清到畫面尾）。
col = 0  # 估計游標喺第幾格：未定嘅字唔可以換行，換行令畫面捲動之後，記住嘅位置會錯
last = ""  # 最後寫咗嘅字（英文詞之間要加空格）


def joined(us: list[str]) -> str:
    """接住之前寫咗嘅嘢：英文詞之間加空格（同 join_units 一樣）。"""
    text = join_units(us)
    return " " + text if text and LATIN.match(last) and LATIN.match(text[0]) else text


def write_plain(fixed: list[str], moving: list[str], end: str = ""):
    global col, last
    cols = shutil.get_terminal_size().columns
    text = joined(fixed) + end
    for c in text:
        w = width(c)
        col = w if col + w > cols else col + w
    last = text[-1] if text else last
    more = fit(joined(moving), cols - col - 1)
    print(f"\0338\033[J{text}\0337" + (f"\033[2m{more}\033[0m" if more else ""), end="", flush=True)


# local agreement：連續兩次即時轉錄都一樣嘅開頭先當「定咗」，之後唔再改（最終版本都保留）
prev: list[str] = []
stable: list[str] = []
printed = 0  # --plain：今句已經寫咗幾多個定咗嘅單位


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def show_partial(audio: np.ndarray):
    """講緊嘢：greedy 轉錄到而家為止嘅部分，覆蓋同一行；未定嘅字用暗色。"""
    global prev, stable, printed
    hyp = transcribe_batch(enc, model, vocab, [audio])[0]
    agreed = common_prefix(prev, hyp)
    if len(agreed) > len(stable) and agreed[:len(stable)] == stable:
        stable = agreed
    prev = hyp
    moving = hyp[len(common_prefix(stable, hyp)):]  # 呢幾個字仲會變
    if args.plain:
        write_plain(stable[printed:], moving)
        printed = len(stable)
        return
    fixed, text = join_units(stable), join_units(stable + moving)
    shown = tail(text, shutil.get_terminal_size().columns - 5)
    n = len(text) - len(fixed)
    if 0 < n <= len(shown):
        shown = f"{shown[:-n]}\033[2m{shown[-n:]}\033[0m"
    print(f"\r\033[K  … {shown}", end="", flush=True)


def keep_stable(hyp: list[str]) -> list[str]:
    """最終版本（beam + LM）：定咗嘅字照用，只換佢哋後面嗰截。"""
    if not stable:
        return hyp
    # stable 尾喺 hyp 入面對應嘅位置：hyp 邊個開頭同 stable 最似（打和就揀長度最接近嗰個）
    end = min(range(len(hyp) + 1), key=lambda j: (Levenshtein.distance(stable, hyp[:j]), abs(j - len(stable))))
    return stable + hyp[end:]


def emit(audio: np.ndarray):
    global prev, stable, printed
    t0 = time.time()
    units = keep_stable(transcribe_batch(enc, model, vocab, [audio], lm)[0])
    if args.plain:
        write_plain(units[printed:], [], " " if units else "")  # 句同句之間隔一格
        prev, stable, printed = [], [], 0
        return
    prev, stable = [], []
    text = join_units(units)
    print("\r\033[K", end="")  # 清走即時更新嗰行
    if text:
        print(f"[{len(audio) / SAMPLE_RATE:4.1f}s 句｜{(time.time() - t0) * 1000:3.0f}ms] {text}")


def run(source):
    recent = deque(maxlen=PRE_ROLL)
    if args.plain:
        print("\0337", end="", flush=True)  # 記住起點，write_plain 由呢度寫
    speech: list[np.ndarray] | None = None
    for block in source:
        event = vad(torch.from_numpy(block))
        if speech is None:
            recent.append(block)
            if event and "start" in event:
                speech = list(recent)
        else:
            speech.append(block)
            if event and "end" in event:
                emit(np.concatenate(speech))
                speech = None
                recent.clear()
            elif len(speech) * BLOCK >= max_sec * SAMPLE_RATE:
                # 夠長就先出字；VAD 仲當係講緊嘢（唔會再有 start），所以即刻開始錄下一段，唔係會漏字
                emit(np.concatenate(speech))
                speech = []
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

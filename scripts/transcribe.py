"""用 train 好嘅 reader 轉錄音頻檔。

用法：python scripts/transcribe.py runs/learned_r3/best.pt audio.wav [--lm data/lm_r3.pkl --alpha 0.3 --beta 2.5]
      長過 20 秒嘅音頻會先用 VAD 切出有人講嘢嘅段落（--no-vad 就照舊每 30 秒硬切）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # 寫去檔案都即時睇到 log

import argparse
import json
import time

import librosa

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.lm import NGramLM
from g2asr.text import edit_distance, join_units, units
from g2asr.vad import MAX_SEC, speech_segments

CHUNK_SEC = 30  # 長音頻切 30 秒一段

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("audio", nargs="+")
parser.add_argument("--lm", help="n-gram LM（build_lm.py 整）；唔畀就用 greedy")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=2.5)
parser.add_argument("--no-vad", action="store_true", help="長音頻唔用 VAD，每 30 秒硬切")
args = parser.parse_args()

model, vocab = load_reader(args.ckpt)
enc = Encoder()
lm = NGramLM.load(args.lm) if args.lm else None

# samples/refs.json 有參考字幕就順便計 CER
refs = {}
for path in args.audio:
    ref_file = Path(path).parent / "refs.json"
    if ref_file.exists():
        refs.update(json.loads(ref_file.read_text(encoding="utf-8")))

for path in args.audio:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    t0 = time.time()
    if len(wav) > MAX_SEC * SAMPLE_RATE and not args.no_vad:
        chunks = [wav[a:b] for a, b in speech_segments(wav)]
    else:
        chunks = [wav[i:i + CHUNK_SEC * SAMPLE_RATE] for i in range(0, len(wav), CHUNK_SEC * SAMPLE_RATE)]
    hyp = [u for us in transcribe_batch(enc, model, vocab, chunks, lm, args.alpha, args.beta) for u in us]
    elapsed = time.time() - t0
    line = f"{Path(path).name}  ({len(wav) / SAMPLE_RATE / elapsed:.0f}× 即時)  {join_units(hyp)}"
    ref = refs.get(Path(path).stem)
    if ref:
        ref_units = units(ref["text"])
        line += f"\n    參考：{join_units(ref_units)}　CER {edit_distance(hyp, ref_units) / len(ref_units):.0%}"
    print(line)

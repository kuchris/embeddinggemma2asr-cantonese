"""用 train 好嘅 reader 轉錄音頻檔。

用法：python scripts/transcribe.py runs/frozen/best.pt samples/*.wav
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import librosa
import torch

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units, units

CHUNK_SEC = 30  # 長音頻切 30 秒一段

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("audio", nargs="+")
args = parser.parse_args()

ck = torch.load(args.ckpt, weights_only=False)
model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
model.load_state_dict(ck["model"])
enc = Encoder()

# samples/refs.json 有參考字幕就順便計 CER
refs = {}
for path in args.audio:
    ref_file = Path(path).parent / "refs.json"
    if ref_file.exists():
        refs.update(json.loads(ref_file.read_text(encoding="utf-8")))

for path in args.audio:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    t0 = time.time()
    chunks = [wav[i:i + CHUNK_SEC * SAMPLE_RATE] for i in range(0, len(wav), CHUNK_SEC * SAMPLE_RATE)]
    ids = []
    with torch.no_grad():
        for feats in enc.frames(chunks):
            lengths = torch.tensor([len(feats)]).cuda()
            ids += Reader.greedy(model(feats[None].cuda().float(), lengths), lengths)[0]
    hyp = [ck["units"][i] for i in ids]
    elapsed = time.time() - t0
    line = f"{Path(path).name}  ({len(wav) / SAMPLE_RATE / elapsed:.0f}× 即時)  {join_units(hyp)}"
    ref = refs.get(Path(path).stem)
    if ref:
        ref_units = units(ref["text"])
        line += f"\n    參考：{join_units(ref_units)}　CER {edit_distance(hyp, ref_units) / len(ref_units):.0%}"
    print(line)

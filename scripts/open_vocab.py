"""開放字典實驗：frozen head 嘅字典係 text embedding，所以可以喺推理時加入 train 時未見過嘅字，唔使重新 train。

將額外詞表（每行一個）計 text embedding，加落 checkpoint 嘅字典，再轉錄。

用法：python scripts/open_vocab.py runs/frozen/best.pt data/wordlists/en10k.txt samples/en-*.wav [--out runs/frozen_en.pt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json

import librosa
import torch

from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units, units

DOC_PROMPT = "title: none | text: "

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("wordlist")
parser.add_argument("audio", nargs="+")
parser.add_argument("--out", help="存一個加大咗字典嘅 checkpoint")
args = parser.parse_args()

ck = torch.load(args.ckpt, weights_only=False)
if ck["config"]["head"] != "frozen":
    raise SystemExit("只有 frozen head 先可以咁加字（learned／tuned 嘅字典係 train 出嚟嘅）")

enc = Encoder()
old = set(ck["units"])
new = [w for line in open(args.wordlist, encoding="utf-8") for w in units(line) if w not in old]
new = sorted(set(new))
print(f"原本字典 {len(old)}，加 {len(new)} 個新單位")
new_emb = enc.embed_texts(new, prompt=DOC_PROMPT)

refs = {}
for path in args.audio:
    ref_file = Path(path).parent / "refs.json"
    if ref_file.exists():
        refs.update(json.loads(ref_file.read_text(encoding="utf-8")))


def load(units_, emb):
    model = Reader(emb, **ck["config"]).cuda().eval()
    state = {k: v for k, v in ck["model"].items() if k != "vocab"}
    model.load_state_dict(state, strict=False)
    return model


variants = {"原本字典": (ck["units"], ck["vocab_emb"]),
            "加大字典": (ck["units"] + new, torch.cat([ck["vocab_emb"], new_emb]))}
feats_cache = {p: enc.frames([librosa.load(p, sr=SAMPLE_RATE, mono=True)[0]])[0] for p in args.audio}
for name, (us, emb) in variants.items():
    model = load(us, emb)
    errors = total = 0
    print(f"\n== {name}（{len(us)}）")
    for path, feats in feats_cache.items():
        lengths = torch.tensor([len(feats)]).cuda()
        with torch.no_grad():
            ids = Reader.greedy(model(feats[None].cuda().float(), lengths), lengths)[0]
        hyp = [us[i] for i in ids]
        line = f"  {Path(path).name}  {join_units(hyp)}"
        ref = refs.get(Path(path).stem)
        if ref:
            r = units(ref["text"])
            e = edit_distance(hyp, r)
            errors, total = errors + e, total + len(r)
            line += f"  （CER {e / len(r):.0%}）"
        print(line)
    if total:
        print(f"  合計 CER {errors / total:.1%}")

if args.out:
    us, emb = variants["加大字典"]
    torch.save({**ck, "units": us, "vocab_emb": emb,
                "model": {**ck["model"], "vocab": torch.nn.functional.normalize(emb.float(), dim=-1)}}, args.out)
    print(f"已存 {args.out}")

"""喺外部數據集（冇用嚟 train）比較幾個 checkpoint，例如 LibriSpeech 英文。

英文單位係詞，所以算出嚟嘅「CER」其實係 WER。

用法：python scripts/eval_external.py --repo openslr/librispeech_asr --config clean --split test --text-col text \
        --n 200 runs/frozen/best.pt runs/frozen_en.pt runs/learned/best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json

import soundfile as sf
import torch

from g2asr.data import decode_audio, iter_rows
from g2asr.decode import beam_search
from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.lm import NGramLM
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units, units

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--repo", required=True)
parser.add_argument("--config", default="default")
parser.add_argument("--split", default="test")
parser.add_argument("--text-col", default="text")
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--show", type=int, default=4)
parser.add_argument("--lm", help="加 beam search + LM（α=0.7 β=4.5）")
parser.add_argument("--export", help="將用到嘅音頻存做 wav + refs.json（畀其他 ASR 比較）")
args = parser.parse_args()

enc = Encoder()
wavs, refs = [], []
for row in iter_rows(args.repo, args.split, ["audio", args.text_col], args.config):
    wav = decode_audio(row["audio"], SAMPLE_RATE)
    if len(wav) / SAMPLE_RATE > 20:
        continue
    wavs.append(wav)
    refs.append(units(row[args.text_col]))
    if len(wavs) >= args.n:
        break
feats = [f for i in range(0, len(wavs), 16) for f in enc.frames(wavs[i:i + 16])]
if args.export:
    exp = Path(args.export)
    exp.mkdir(parents=True, exist_ok=True)
    refs_out = {}
    for i, (wav, ref) in enumerate(zip(wavs, refs)):
        sf.write(exp / f"{i:05d}.wav", wav, SAMPLE_RATE)
        refs_out[f"{i:05d}"] = {"text": join_units(ref)}
    (exp / "refs.json").write_text(json.dumps(refs_out, ensure_ascii=False, indent=1), encoding="utf-8")
lm = NGramLM.load(args.lm) if args.lm else None
total = sum(map(len, refs))
print(f"{args.repo}/{args.config}/{args.split}：{len(wavs)} 句，{total} 個單位，"
      f"{sum(len(w) for w in wavs) / SAMPLE_RATE / 60:.1f} 分鐘")

for path in args.ckpts:
    ck = torch.load(path, weights_only=False)
    model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
    model.load_state_dict(ck["model"])
    vocab = set(ck["units"])
    errors, errors_lm, hyps = 0, 0, []
    with torch.no_grad():
        for f, ref in zip(feats, refs):
            lengths = torch.tensor([len(f)]).cuda()
            lp = model(f[None].cuda().float(), lengths)
            hyp = [ck["units"][i] for i in Reader.greedy(lp, lengths)[0]]
            errors += edit_distance(hyp, ref)
            hyps.append(hyp)
            if lm:
                ids = beam_search(lp[0].float().cpu().numpy(), ck["units"], lm, alpha=0.7, beta=4.5)
                errors_lm += edit_distance([ck["units"][i] for i in ids], ref)
    oov = sum(u not in vocab for r in refs for u in r) / total
    print(f"\n{path}（head={ck['config']['head']}，字典 {len(vocab)}，參考字唔喺字典 {oov:.1%}）：錯誤率 {errors / total:.1%}"
          + (f"，+LM {errors_lm / total:.1%}" if lm else ""))
    for hyp, ref in list(zip(hyps, refs))[:args.show]:
        print(f"  參考：{join_units(ref)}\n  輸出：{join_units(hyp)}")

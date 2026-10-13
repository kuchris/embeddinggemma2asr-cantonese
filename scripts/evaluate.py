"""喺驗證 shard 比較 greedy 同 beam search + LM 嘅 CER，順便揀 alpha／beta。

用法：python scripts/evaluate.py runs/frozen/best.pt data/feats/daily data/feats/mixed data/feats/wordshk
      --lm data/lm.pkl [--max-utts 1500] [--alphas 0.3 0.5 0.8] [--betas 0.5 1.5]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import time

import torch

from g2asr.data import shard_paths
from g2asr.decode import beam_search
from g2asr.lm import NGramLM
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--lm")
parser.add_argument("--max-utts", type=int, default=1500)
parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.8])
parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.5])
parser.add_argument("--beam", type=int, default=8)
args = parser.parse_args()

ck = torch.load(args.ckpt, weights_only=False)
model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
model.load_state_dict(ck["model"])
units = ck["units"]
unit_to_id = {u: i for i, u in enumerate(units)}

# 每個數據集嘅驗證 shard 平均抽句子
items = []
per_set = args.max_utts // len(args.feat_dirs)
random.seed(0)
for d in args.feat_dirs:
    shard = torch.load(shard_paths([d])[-1])
    pairs = [(f, [unit_to_id[u] for u in us if u in unit_to_id]) for f, us in zip(shard["feats"], shard["units"])]
    pairs = [p for p in pairs if p[1]]
    items += random.sample(pairs, min(per_set, len(pairs)))

log_probs = []
with torch.no_grad():
    for f, _ in items:
        lp = model(f[None].cuda().float(), torch.tensor([len(f)]).cuda())[0]
        log_probs.append(lp.float().cpu().numpy())
refs = [ref for _, ref in items]
total = sum(map(len, refs))


def cer(hyps):
    return sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / total


greedy = [Reader.greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0] for lp in log_probs]
print(f"{len(items)} 句，{total} 個字")
print(f"greedy                CER {cer(greedy):.2%}")

t0 = time.time()
no_lm = [beam_search(lp, units, None, beam=args.beam) for lp in log_probs]
print(f"beam（冇 LM）           CER {cer(no_lm):.2%}  {time.time() - t0:.0f}s")

if args.lm:
    lm = NGramLM.load(args.lm)
    best = None
    for alpha in args.alphas:
        for beta in args.betas:
            t0 = time.time()
            hyps = [beam_search(lp, units, lm, alpha, beta, beam=args.beam) for lp in log_probs]
            c = cer(hyps)
            print(f"beam + LM α={alpha} β={beta}  CER {c:.2%}  {time.time() - t0:.0f}s")
            if best is None or c < best[0]:
                best = (c, alpha, beta, hyps)
    c, alpha, beta, hyps = best
    print(f"\n最好：α={alpha} β={beta}，CER {c:.2%}（greedy {cer(greedy):.2%}）\n例子（LM 改咗嘅）：")
    shown = 0
    for g, h, r in zip(greedy, hyps, refs):
        if g != h and shown < 8:
            print(f"  參考：{join_units([units[i] for i in r])}\n  greedy：{join_units([units[i] for i in g])}"
                  f"\n  +LM：{join_units([units[i] for i in h])}")
            shown += 1

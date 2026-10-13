"""由 train shard 嘅字幕整 n-gram 語言模型（唔用驗證 shard，費事作弊）。

可以再加純文字檔（每行一句）擴大語料。

用法：python scripts/build_lm.py data/feats/daily data/feats/mixed data/feats/wordshk --out data/lm.pkl
      [--order 4] [--text extra.txt ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # 寫去檔案都即時睇到 log

import argparse

import torch

from g2asr.data import shard_paths
from g2asr.lm import NGramLM
from g2asr.text import units

parser = argparse.ArgumentParser()
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--order", type=int, default=4)
parser.add_argument("--text", nargs="*", default=[], help="額外純文字語料，每行一句")
args = parser.parse_args()

sentences = []
for d in args.feat_dirs:
    for path in shard_paths([d])[:-1]:  # 最後一個 shard 係驗證集
        sentences += torch.load(path)["units"]
for path in args.text:
    sentences += [us for line in open(path, encoding="utf-8") if (us := units(line))]

lm = NGramLM(args.order).fit(sentences)
lm.save(args.out)
print(f"已存 {args.out}：{len(sentences)} 句，{sum(map(len, sentences))} 個單位，{args.order}-gram")

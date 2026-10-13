"""第 2 步：由 shard 入面嘅字幕整字典，再用 EmbeddingGemma 2 計每個字嘅 text embedding。

用法：python scripts/build_vocab.py data/feats/wordshk data/feats/mixed --out data/vocab.pt [--min-count 2]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import Counter

import torch

from g2asr.data import shard_paths
from g2asr.encoder import Encoder

DOC_PROMPT = "title: none | text: "  # Spike 0：用 Document prompt 嘅字同音頻對得最齊

parser = argparse.ArgumentParser()
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--min-count", type=int, default=2)
args = parser.parse_args()

counts = Counter()
for path in shard_paths(args.feat_dirs):
    for us in torch.load(path)["units"]:
        counts.update(us)
vocab = sorted(u for u, c in counts.items() if c >= args.min_count)
covered = sum(counts[u] for u in vocab) / sum(counts.values())
print(f"字典 {len(vocab)} 個單位（共 {len(counts)} 個，覆蓋 {covered:.2%} 出現次數）")

enc = Encoder(with_audio=False)
emb = enc.embed_texts(vocab, prompt=DOC_PROMPT)
torch.save({"units": vocab, "emb": emb, "prompt": DOC_PROMPT}, args.out)
print(f"已存 {args.out}：{tuple(emb.shape)}")

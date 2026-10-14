"""喺多層 feature shard（extract.py --layers ...）上面為每層計 PCA，畀 extract.py --pca 壓縮用。

每層：mean (D,) 同頭 k 個主成分 comps (D, k)，壓縮 = (x − mean) @ comps。
4 層全存（6／9／12／原本 = 4096 維）太大：4 層都壓縮做 256 維（共 1024 維）再用 fp8 存，
每格 1024 bytes，同原本 512 維 fp16 一樣大（720 小時約 63GB，A100 RAM 預先載入得落）。

用法：python scripts/fit_pca.py data/feats/mdcc_layers --layers 6,9,12,0 --k 256 --out data/pca_6_9_12_0.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import random

import torch

from g2asr.data import shard_paths
from g2asr.encoder import layer_dim

parser = argparse.ArgumentParser()
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--layers", required=True, help="要壓縮嘅層（逗號分隔）")
parser.add_argument("--k", type=int, default=256)
parser.add_argument("--max-frames", type=int, default=2_000_000, help="最多用幾多格計（隨機抽）")
parser.add_argument("--out", required=True)
args = parser.parse_args()
want = [int(x) for x in args.layers.split(",")]
random.seed(0)

paths = shard_paths(args.feat_dirs)
random.shuffle(paths)
per_shard = max(1, args.max_frames // len(paths))
cols, samples = None, {l: [] for l in want}
for path in paths:
    shard = torch.load(path)
    have = shard.get("layers", [0])
    dims = shard.get("dims") or [layer_dim(l) for l in have]
    starts = dict(zip(have, [sum(dims[:i]) for i in range(len(dims))]))
    feats = torch.cat(shard["feats"])
    idx = torch.randperm(len(feats))[:per_shard]
    for l in want:
        samples[l].append(feats[idx, starts[l]:starts[l] + dict(zip(have, dims))[l]].float())
    del shard, feats

pca = {}
for l in want:
    x = torch.cat(samples[l]).cuda()
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / (len(x) - 1)
    evals, evecs = torch.linalg.eigh(cov)  # 由細到大
    evals, evecs = evals.flip(0), evecs.flip(1)
    kept = (evals[:args.k].sum() / evals.sum()).item()
    pca[l] = (mean.cpu(), evecs[:, :args.k].contiguous().cpu())
    print(f"第 {l} 層：{x.shape[1]} → {args.k} 維，保留 {kept:.1%} 變異（用 {len(x)} 格）")
torch.save({"pca": pca, "k": args.k}, args.out)
print(f"已存：{args.out}")

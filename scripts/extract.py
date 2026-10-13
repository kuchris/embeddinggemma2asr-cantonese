"""第 1 步：將數據集嘅音頻過一次凍結 encoder，存逐格 feature 做 shard（之後 train 唔使再行 encoder）。

大約每小時音頻 92MB（25 格／秒 × 512 維 × fp16）。

用法：python scripts/extract.py --dataset wordshk --out data/feats/wordshk [--max-hours 20]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from g2asr.data import DATASETS, decode_audio, iter_rows
from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.text import units

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=DATASETS)
parser.add_argument("--out", required=True)
parser.add_argument("--max-hours", type=float, default=0, help="0 = 全部")
parser.add_argument("--max-sec", type=float, default=20, help="長過呢個嘅句子跳過")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--shard-size", type=int, default=4000)
parser.add_argument("--workers", type=int, default=8, help="解碼音頻嘅線程數")
args = parser.parse_args()

repo, split, text_col = DATASETS[args.dataset]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
progress = out / "progress.json"  # 斷咗可以續做：記住已經睇過幾多行
state = json.loads(progress.read_text()) if progress.exists() else {"shards": 0, "rows": 0, "hours": 0.0}
done, skip = state["shards"], state["rows"]

ds = iter_rows(repo, split, ["audio", text_col])
enc = Encoder()
shard = {"feats": [], "units": [], "text": []}
pending, hours, n_seen, n_skipped = [], 0.0, skip, 0
total_hours = state["hours"]


def flush_batch():
    feats = enc.frames([w for w, _, _ in pending])
    for f, (_, us, text) in zip(feats, pending):
        shard["feats"].append(f)
        shard["units"].append(us)
        shard["text"].append(text)
    pending.clear()


def flush_shard():
    global done, shard
    if shard["feats"]:
        torch.save(shard, out / f"shard_{done:04d}.pt")
        done += 1
        progress.write_text(json.dumps({"shards": done, "rows": n_seen, "hours": total_hours + hours}))
    shard = {"feats": [], "units": [], "text": []}


def prepare(row):
    """喺線程入面做：解碼 + 重新取樣 + 切單位（soundfile／soxr 會放開 GIL，可以真係並行）。"""
    text = row[text_col] or ""
    try:
        return decode_audio(row["audio"], SAMPLE_RATE), units(text), text, None
    except Exception as e:  # 壞檔就跳過
        return None, None, text, e


def prepared(rows):
    """按原本次序 yield（行號, 結果），背後用線程池預先解碼，等 GPU 唔使等 CPU。"""
    with ThreadPoolExecutor(args.workers) as pool:
        queue = deque()
        for i, row in enumerate(rows, 1):
            if i <= skip:
                continue
            queue.append((i, pool.submit(prepare, row)))
            if len(queue) >= args.workers * 8:
                i0, fut = queue.popleft()
                yield i0, fut.result()
        while queue:
            i0, fut = queue.popleft()
            yield i0, fut.result()


bar = tqdm(prepared(ds), desc=args.dataset, initial=skip)
for n_seen, (wav, us, text, err) in bar:
    if err is not None:
        n_skipped += 1
        bar.write(f"跳過（解碼失敗）：{err}")
        continue
    sec = len(wav) / SAMPLE_RATE
    if not us or sec < 0.3 or sec > args.max_sec:
        n_skipped += 1
        continue
    pending.append((wav, us, text))
    hours += sec / 3600
    if len(pending) == args.batch:
        flush_batch()
    if len(shard["feats"]) >= args.shard_size:
        flush_shard()
    bar.set_postfix(hours=f"{hours:.1f}", shards=done, skipped=n_skipped)
    if args.max_hours and total_hours + hours >= args.max_hours:
        break

if pending:
    flush_batch()
flush_shard()
print(f"完成：{done} 個 shard，累計 {total_hours + hours:.1f} 小時，今次跳過 {n_skipped} 句")

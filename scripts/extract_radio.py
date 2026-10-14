"""老師標籤（知識蒸餾）：alvanlii/cantonese-radio 每句都有 Whisper 同 SenseVoice 兩個轉錄。
只留兩個老師講法差唔多（CER < --agree）而且音質好嘅句子，用 SenseVoice 版本做標籤，抽 feature 做 shard。

數據集有存取限制：要先喺 Hugging Face 同意條款，再 `hf auth login`。

用法：python scripts/extract_radio.py --out data/feats/radio --max-hours 500 [--stride 12]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from rapidfuzz.distance import Levenshtein

from g2asr.data import RAW_DIR, decode_audio
from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.text import units

REPO = "alvanlii/cantonese-radio"
FILLERS = re.compile(r"[誒嗯呃]")
# SenseVoice 成日用「系」代「係」；呢啲詞入面嘅「系」先係真係「系」
REAL_XI = re.compile(r"系(統|列|數|譜|眼)|(體|派|院|學|科|水|銀河|太陽|星|直|旁|母|父)系")

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--max-hours", type=float, default=500)
parser.add_argument("--stride", type=int, default=12, help="每隔幾多個 parquet 檔揀一個（覆蓋多啲唔同節目）")
parser.add_argument("--offset", type=int, default=0, help="由第幾個檔開始（幾條流程並行時，各自用唔同 offset）")
parser.add_argument("--agree", type=float, default=0.15, help="兩個老師之間 CER 上限")
parser.add_argument("--min-snr", type=float, default=15)
parser.add_argument("--max-sec", type=float, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--workers", type=int, default=12)
args = parser.parse_args()


def clean_label(text: str) -> str:
    text = FILLERS.sub("", text or "")
    protected = [m.span() for m in REAL_XI.finditer(text)]
    return "".join("係" if c == "系" and not any(a <= i < b for a, b in protected) else c for i, c in enumerate(text))


def keep(row) -> list[str] | None:
    try:
        snr, dur = float(row["snr"]), float(row["speech_duration"])
    except (TypeError, ValueError):
        return None
    if snr < args.min_snr or not 1.0 <= dur <= args.max_sec:
        return None
    sv, wh = units(clean_label(row["transcript_sensevoice"])), units(clean_label(row["transcript_whisper"]))
    if len(sv) < 3 or Levenshtein.distance(sv, wh) / len(sv) > args.agree:
        return None
    return sv


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
progress = out / "progress.json"
state = json.loads(progress.read_text()) if progress.exists() else {"shards": 0, "files": [], "hours": 0.0, "seen": 0, "kept": 0}

fs = HfFileSystem()
files = sorted(fs.glob(f"datasets/{REPO}/data/*.parquet"))[args.offset::args.stride]
todo = [f for f in files if Path(f).name not in state["files"]]
print(f"揀咗 {len(files)} 個 parquet 檔，未做 {len(todo)} 個；已有 {state['hours']:.1f} 小時")


def download(path: str) -> Path:
    name = "data/" + Path(path).name
    return Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=RAW_DIR / REPO))


enc = Encoder()
shard = {"feats": [], "units": [], "text": []}


def flush_shard():
    global shard
    if shard["feats"]:
        torch.save(shard, out / f"shard_{state['shards']:04d}.pt")
        state["shards"] += 1
    shard = {"feats": [], "units": [], "text": []}


def decode_row(row):
    try:
        return decode_audio(row["audio"], SAMPLE_RATE)
    except Exception:
        return None


pool = ThreadPoolExecutor(args.workers)
prefetch = {}
lock = threading.Lock()
cols = ["audio", "transcript_sensevoice", "transcript_whisper", "snr", "speech_duration"]
for n, path in enumerate(todo):
    if path not in prefetch:
        prefetch[path] = pool.submit(download, path)
    if n + 1 < len(todo):  # 下載下一個檔同處理呢個檔同時做
        prefetch[todo[n + 1]] = pool.submit(download, todo[n + 1])
    local = prefetch.pop(path).result()
    pf = pq.ParquetFile(local)
    for rg in range(pf.num_row_groups):
        rows = pf.read_row_group(rg, columns=cols).to_pylist()
        state["seen"] += len(rows)
        picked = [(r, us) for r in rows if (us := keep(r))]
        wavs = list(pool.map(lambda x: decode_row(x[0]), picked))
        good = [(w, us, r["transcript_sensevoice"]) for w, (r, us) in zip(wavs, picked)
                if w is not None and len(w) <= args.max_sec * SAMPLE_RATE]
        for i in range(0, len(good), args.batch):
            chunk = good[i:i + args.batch]
            for f, (w, us, text) in zip(enc.frames([w for w, _, _ in chunk]), chunk):
                shard["feats"].append(f)
                shard["units"].append(us)
                shard["text"].append(text)
                state["hours"] += len(w) / SAMPLE_RATE / 3600
                state["kept"] += 1
    flush_shard()  # 每個 parquet 檔一個 shard（約 1000 句），斷咗由下一個檔續做
    state["files"].append(Path(path).name)
    progress.write_text(json.dumps(state))
    local.unlink()  # 慳硬碟
    print(f"[{n + 1}/{len(todo)}] {Path(path).name}：累計 {state['hours']:.1f} 小時，"
          f"保留 {state['kept']}/{state['seen']}（{state['kept'] / max(state['seen'], 1):.0%}）")
    if state["hours"] >= args.max_hours:
        break
print(f"完成：{state['shards']} 個 shard，{state['hours']:.1f} 小時")

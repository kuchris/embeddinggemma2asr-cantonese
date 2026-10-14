"""將 evaluate.py --part test 嗰批句子嘅原始音頻攞返嚟（shard 只存 feature），存做 wav + refs.json，
等其他 ASR（例如 SenseVoice）可以喺完全一樣嘅句子比較。順便計我哋 model 喺呢批句子嘅 CER。

句子揀法同 evaluate.py 一模一樣（random.seed(0)，每個數據集驗證 shard 打亂後取後一半嘅頭 N 句）。
音頻係由數據集 parquet 尾段按字幕次序對返出嚟。

用法：python scripts/export_testset.py runs/frozen/best.pt runs/learned/best.pt --out data/testset --lm data/lm.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import random

import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfFileSystem

from g2asr.data import DATASETS, decode_audio, shard_paths
from g2asr.decode import beam_search
from g2asr.encoder import SAMPLE_RATE
from g2asr.lm import NGramLM
from g2asr.reader import Reader
from g2asr.text import edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--lm")
parser.add_argument("--feat-root", default="data/feats")
parser.add_argument("--sets", nargs="+", default=["daily", "mixed", "wordshk"])
parser.add_argument("--max-utts", type=int, default=1500)
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
vocab_units = torch.load(args.ckpts[0], weights_only=False)["units"]
unit_to_id = {u: i for i, u in enumerate(vocab_units)}

# ---------- 1. 重現 evaluate.py --part test 揀句 ----------
random.seed(0)
per_set = args.max_utts // len(args.sets)
chosen = {}  # 數據集 → [(驗證 shard 入面嘅 index, feature, 單位)]
for name in args.sets:
    shard = torch.load(shard_paths([f"{args.feat_root}/{name}"])[-1])
    pairs = [(i, f, us) for i, (f, us) in enumerate(zip(shard["feats"], shard["units"]))
             if [unit_to_id[u] for u in us if u in unit_to_id]]
    random.shuffle(pairs)
    chosen[name] = (pairs[len(pairs) // 2:][:per_set], shard["text"])

# ---------- 2. 由 parquet 尾段對返音頻 ----------
fs = HfFileSystem()
items = []  # (檔名, feature, 單位, 原文)
for name, (picked, val_texts) in chosen.items():
    repo, split, text_col = DATASETS[name]
    files = sorted(fs.glob(f"datasets/{repo}@~parquet/default/{split}/*.parquet"))
    rows = []
    for path in files[::-1]:  # 由尾開始讀，夠蓋過驗證 shard 就停
        with fs.open(path) as f:
            rows = pq.read_table(f, columns=["audio", text_col]).to_pylist() + rows
        if len(rows) >= len(val_texts) * 1.2 + 200:
            break
    # 驗證 shard 係數據集最尾嗰段（跳過咗少量壞句）；由尾向前對字幕
    want = {i for i, _, _ in picked}
    audio_of = {}
    j = len(val_texts) - 1
    for row in reversed(rows):
        if j < 0:
            break
        if (row[text_col] or "") == val_texts[j]:
            if j in want:
                audio_of[j] = row["audio"]
            j -= 1
    for i, f, us in picked:
        if i in audio_of:
            fname = f"{name}-{i:05d}"
            sf.write(out / f"{fname}.wav", decode_audio(audio_of[i], SAMPLE_RATE), SAMPLE_RATE)
            items.append((fname, f, us, val_texts[i]))
    print(f"{name}：揀咗 {len(picked)} 句，對到音頻 {sum(i in audio_of for i, _, _ in picked)} 句")

refs = {fname: {"text": text} for fname, _, _, text in items}
(out / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
total = sum(len(us) for _, _, us, _ in items)
print(f"已存 {len(items)} 句（{total} 個字）去 {out}")

# ---------- 3. 我哋 model 喺呢批句子嘅 CER ----------
lm = NGramLM.load(args.lm) if args.lm else None
for path in args.ckpts:
    ck = torch.load(path, weights_only=False)
    model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
    model.load_state_dict(ck["model"])
    errs_g = errs_lm = 0
    with torch.no_grad():
        for _, f, us, _ in items:
            lengths = torch.tensor([len(f)]).cuda()
            lp = model(f[None].cuda().float(), lengths)
            hyp = [ck["units"][i] for i in Reader.greedy(lp, lengths)[0]]
            errs_g += edit_distance(hyp, us)
            if lm:
                ids = beam_search(lp[0].float().cpu().numpy(), ck["units"], lm, alpha=0.7, beta=4.5)
                errs_lm += edit_distance([ck["units"][i] for i in ids], us)
    line = f"{path}：greedy CER {errs_g / total:.2%}"
    if lm:
        line += f"，+LM（α=0.7 β=4.5）{errs_lm / total:.2%}"
    print(line)

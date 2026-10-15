"""重新為某個 checkpoint 揀 LM 嘅 α／β：三個測試集各自打亂分兩半，前一半（合埋）揀參數，後一半報告。

log 機率會 cache 落 --cache（npz），之後試其他 LM 唔使再行 GPU。

用法：python scripts/lm_sweep.py runs/learned_r6b/best.pt --lm data/lm_r4.pkl [--n 400]
      [--alphas 0 0.1 0.2 0.3 0.5 0.7] [--betas 0 1 2 3 4.5] [--cache data/lp_r6b.npz]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from g2asr import env  # noqa: F401  # 必須最先 import

import argparse
import json
import random
import time

import numpy as np
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

from g2asr.decode import beam_search
from g2asr.lm import NGramLM
from g2asr.reader import Reader
from g2asr.spice import clean, norm
from g2asr.text import units

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("--lm", nargs="*", default=[])
parser.add_argument("--n", type=int, default=400, help="每個測試集抽幾多句（一半揀參數、一半報告）")
parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7])
parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.5])
parser.add_argument("--beam", type=int, default=8)
parser.add_argument("--cache")
args = parser.parse_args()
random.seed(0)


def load_set(folder: Path, ref_fn, keep=None):
    refs = json.loads((folder / "refs.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in refs if (keep is None or keep(k)) and (folder / f"{k}.wav").exists())
    keys = random.sample(keys, min(args.n, len(keys)))
    return [(folder / f"{k}.wav", ref_fn(refs[k]["text"])) for k in keys]


sets = {
    "原本 test": load_set(ROOT / "data" / "testset", units),
    "MDCC test": load_set(ROOT / "data" / "mdcc_test2000", units),
    "SpiCE 逐句": [x for t in ("VF19A", "VM19A")
                 for x in load_set(ROOT / "data" / "spice" / "prepared" / t / "seg", clean)],
}
sets["SpiCE 逐句"] = random.sample(sets["SpiCE 逐句"], min(args.n, len(sets["SpiCE 逐句"])))
to_units = {name: (lambda us: norm(us)) if name.startswith("SpiCE") else (lambda us: us) for name in sets}

ck_units = torch.load(args.ckpt, weights_only=False, map_location="cpu")["units"]
if args.cache and Path(args.cache).exists():
    z = np.load(args.cache, allow_pickle=True)
    log_probs = z["log_probs"].item()
else:
    from g2asr.encoder import Encoder
    from g2asr.infer import load_reader
    from torch.nn.utils.rnn import pad_sequence
    enc = Encoder()
    model, _ = load_reader(args.ckpt)
    log_probs = {}
    with torch.inference_mode():
        for name, items in sets.items():
            wavs = [sf.read(p, dtype="float32")[0] for p, _ in items]
            out = []
            for i in range(0, len(wavs), 16):
                feats = enc.frames(wavs[i:i + 16], model.feat_layers, model.pca())
                lengths = torch.tensor([len(f) for f in feats]).cuda()
                x = pad_sequence(feats, batch_first=True).cuda().float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = model(x, lengths).float().cpu().numpy()
                out += [lp[j, :n].astype(np.float16) for j, n in enumerate(lengths.tolist())]
            log_probs[name] = out
    del enc, model
    if args.cache:
        np.savez(args.cache, log_probs=np.array(log_probs, dtype=object))

halves = {name: (range(len(items) // 2), range(len(items) // 2, len(items))) for name, items in sets.items()}


def errors(name, idx, hyps):
    refs = [sets[name][i][1] for i in idx]
    return (sum(Levenshtein.distance(to_units[name]([ck_units[j] for j in hyps[i]]), r) for i, r in zip(idx, refs)),
            sum(map(len, refs)))


def report(label, hyps_by_set):
    dev = [errors(n, halves[n][0], hyps_by_set[n]) for n in sets]
    test = [errors(n, halves[n][1], hyps_by_set[n]) for n in sets]
    dev_cer = sum(e for e, _ in dev) / sum(t for _, t in dev)
    cols = "｜".join(f"{n} {e / t:.2%}" for n, (e, t) in zip(sets, test))
    print(f"{label:<28} dev {dev_cer:.2%}｜test：{cols}")
    return dev_cer


lps = {n: [lp.astype(np.float32) for lp in log_probs[n]] for n in sets}
print(f"{args.ckpt}｜每集 {[len(v) for v in sets.values()]} 句")
report("greedy", {n: [Reader.greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0] for lp in lps[n]]
                  for n in sets})
report("beam（冇 LM）", {n: [beam_search(lp, ck_units, None, beam=args.beam) for lp in lps[n]] for n in sets})

for lm_path in args.lm:
    lm = NGramLM.load(lm_path)
    print(f"\n== {lm_path}")
    best = None
    for alpha in args.alphas:
        for beta in args.betas:
            t0 = time.time()
            hyps = {n: [beam_search(lp, ck_units, lm, alpha, beta, beam=args.beam) for lp in lps[n]] for n in sets}
            d = report(f"α={alpha} β={beta} ({time.time() - t0:.0f}s)", hyps)
            if best is None or d < best[0]:
                best = (d, alpha, beta)
    print(f"最好（dev）：α={best[1]} β={best[2]}")

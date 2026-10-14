"""N-best rescoring：beam search（4-gram）出幾個候選，再用細 LLM 為每個候選評分，揀最順嗰句。

LLM 只可以喺候選之間揀，唔會自己加字、刪字、改人名。
最終分數 = ASR 分數（CTC + 4-gram）+ λ × LLM log 機率 + μ × 字數

每個測試集抽 --n 句，前一半用嚟揀 λ、μ（三個測試集一齊揀），後一半報告結果。

用法：python scripts/rescore_nbest.py [--llm Qwen/Qwen3.5-0.8B] [--n 100] [--beam 8]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from g2asr import env  # noqa: F401  # 必須最先 import

import argparse
import itertools
import json
import random
import time

import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForCausalLM, AutoTokenizer

from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch, transcribe_nbest
from g2asr.lm import NGramLM
from g2asr.spice import clean, norm
from g2asr.text import join_units, units

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "以下係一句廣東話口語：\n"
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
MUS = [0.0, 0.5, 1.0, 2.0, 3.0]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r4" / "best_infer.pt"))
parser.add_argument("--lm-path", default=str(ROOT / "data" / "lm_r4.pkl"))
parser.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--n", type=int, default=100, help="每個測試集抽幾多句（一半揀參數、一半報告）")
parser.add_argument("--beam", type=int, default=8)
args = parser.parse_args()
random.seed(0)


def load_set(folder: Path, ref_fn, keep=None):
    refs = json.loads((folder / "refs.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in refs if (keep is None or keep(k)) and (folder / f"{k}.wav").exists())
    keys = random.sample(keys, min(args.n, len(keys)))
    return [(sf.read(folder / f"{k}.wav", dtype="float32")[0], ref_fn(refs[k]["text"])) for k in keys]


sets = {
    "daily+mixed（原本 test）": load_set(ROOT / "data" / "testset", units, keep=lambda k: not k.startswith("wordshk")),
    "MDCC test": load_set(ROOT / "data" / "mdcc_test2000", units),
    "SpiCE 對話": [x for t in ("VF19A", "VM19A")
                   for x in load_set(ROOT / "data" / "spice" / "prepared" / t / "seg", clean)][:args.n],
}

# ---------- 1. ASR：greedy 同 beam 候選 ----------
enc = Encoder()
reader, vocab = load_reader(args.ckpt)
lm = NGramLM.load(args.lm_path)
greedy, nbest, beam_ms = {}, {}, {}
for name, items in sets.items():
    wavs = [w for w, _ in items]
    greedy[name] = [join_units(h) for h in transcribe_batch(enc, reader, vocab, wavs)]
    t0 = time.time()
    nbest[name] = [[(join_units(u), s) for u, s in cands]
                   for cands in transcribe_nbest(enc, reader, vocab, wavs, lm, 0.7, 5.0, args.beam)]
    beam_ms[name] = (time.time() - t0) / len(items) * 1000
del enc, reader
torch.cuda.empty_cache()

# ---------- 2. LLM 為每個候選評分 ----------
tok = AutoTokenizer.from_pretrained(args.llm)
llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).cuda().eval()
prefix_ids = tok(PREFIX, add_special_tokens=False)["input_ids"]
end_ids = tok("\n", add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def llm_logp(texts: list[str]) -> list[float]:
    """每句（喺 PREFIX 之後、包埋句尾換行）嘅 log 機率總和。"""
    seqs = [prefix_ids + tok(t, add_special_tokens=False)["input_ids"] + end_ids for t in texts]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    logits = llm(input_ids=ids, attention_mask=mask).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
    keep = mask[:, 1:].clone()
    keep[:, :len(prefix_ids) - 1] = 0  # 唔計 PREFIX 本身
    return (lp * keep).sum(1).tolist()


llm_scores, llm_ms = {}, {}
for name in sets:
    t0 = time.time()
    llm_scores[name] = [llm_logp([t for t, _ in cands]) for cands in nbest[name]]
    llm_ms[name] = (time.time() - t0) / len(sets[name]) * 1000

# ---------- 3. 揀 λ、μ（前一半），報告（後一半） ----------
to_units = {name: (lambda t: norm(units(t))) if name.startswith("SpiCE") else units for name in sets}


def pick(name, i, lam, mu):
    cands = nbest[name][i]
    best = max(range(len(cands)), key=lambda j: cands[j][1] + lam * llm_scores[name][i][j]
               + mu * len(units(cands[j][0])))
    return cands[best][0]


def errors(name, idx, hyp_fn):
    refs = [sets[name][i][1] for i in idx]
    return sum(Levenshtein.distance(to_units[name](hyp_fn(i)), r) for i, r in zip(idx, refs)), sum(map(len, refs))


halves = {name: (range(len(items) // 2), range(len(items) // 2, len(items))) for name, items in sets.items()}


def dev_cer(lam, mu):
    e = t = 0
    for name in sets:
        de, dt = errors(name, halves[name][0], lambda i: pick(name, i, lam, mu))
        e, t = e + de, t + dt
    return e / t


lam, mu = min(itertools.product(LAMBDAS, MUS), key=lambda p: dev_cer(*p))
print(f"\n揀咗 λ={lam}、μ={mu}（前一半 dev CER {dev_cer(lam, mu):.1%}；λ=0 即係淨用 4-gram：{dev_cer(0, 0):.1%}）")
print(f"LLM：{args.llm}｜beam {args.beam}｜以下係後一半（冇用嚟揀參數）")

for name in sets:
    idx = halves[name][1]

    def cer(fn):
        e, t = errors(name, idx, fn)
        return f"{e / t:.1%}"

    oracle = lambda i: min((c for c, _ in nbest[name][i]),
                           key=lambda c: Levenshtein.distance(to_units[name](c), sets[name][i][1]))
    print(f"\n== {name}：{len(idx)} 句｜greedy {cer(lambda i: greedy[name][i])}"
          f"｜4-gram {cer(lambda i: nbest[name][i][0][0])}"
          f"｜+LLM 重排 {cer(lambda i: pick(name, i, lam, mu))}"
          f"｜候選上限 {cer(oracle)}"
          f"｜每句 beam {beam_ms[name]:.0f}ms + LLM {llm_ms[name]:.0f}ms")
    shown = 0
    for i in idx:
        a, b = nbest[name][i][0][0], pick(name, i, lam, mu)
        if a != b and shown < 5:
            print(f"  參考：{join_units(sets[name][i][1])}\n  4-gram：{a}\n  +LLM ：{b}")
            shown += 1

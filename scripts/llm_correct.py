"""用細 LLM（預設 Qwen3.5-0.8B，冇 fine-tune）修正 ASR 輸出，比較修正前後 CER。

1. gemma2asr（greedy）轉錄測試集 → 假設句子
2. LLM 逐批改正（只改明顯錯字、保留口語）
3. 計 CER：ASR 原本 vs LLM 修正後

測試集：data/testset、data/mdcc_test2000、data/spice/prepared/*/seg（要先用其他 script 準備好）

用法：python scripts/llm_correct.py [--llm Qwen/Qwen3.5-0.8B] [--n 300]
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

import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForCausalLM, AutoTokenizer

from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.spice import clean, norm
from g2asr.text import join_units, units

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("你係廣東話語音辨識嘅校對員。下面係語音辨識出嚟嘅一句說話，可能有同音字或者近音字錯誤。\n"
          "規則：只改明顯錯嘅字；保留廣東話口語寫法（唔好改做書面語）；唔好加減內容；唔使加標點；"
          "如果冇錯就原句照抄。只輸出改正後嘅句子。\n\n句子：{text}")

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r4" / "best_infer.pt"))
parser.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--n", type=int, default=300, help="每個測試集抽幾多句")
parser.add_argument("--batch", type=int, default=16)
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

# ---------- 1. ASR ----------
enc = Encoder()
reader, vocab = load_reader(args.ckpt)
hyps = {name: transcribe_batch(enc, reader, vocab, [w for w, _ in items]) for name, items in sets.items()}
del enc, reader
torch.cuda.empty_cache()

# ---------- 2. LLM 修正 ----------
tok = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).cuda().eval()


@torch.inference_mode()
def correct(texts: list[str]) -> tuple[list[str], float]:
    out, t0 = [], time.time()
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for t in chunk]
        enc_in = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        gen = llm.generate(**enc_in, max_new_tokens=max(len(t) for t in chunk) * 2 + 16, do_sample=False)
        out += [tok.decode(g[enc_in["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
                for g in gen]
    return out, time.time() - t0


# ---------- 3. 比較 ----------
for name, items in sets.items():
    refs = [r for _, r in items]
    is_spice = name.startswith("SpiCE")
    to_units = (lambda t: norm(units(t))) if is_spice else units
    asr = [to_units(join_units(h)) for h in hyps[name]]
    fixed, elapsed = correct([join_units(h) for h in hyps[name]])
    fixed_units = [to_units(t) for t in fixed]
    total = sum(map(len, refs))
    e_asr = sum(Levenshtein.distance(h, r) for h, r in zip(asr, refs)) / total
    e_llm = sum(Levenshtein.distance(h, r) for h, r in zip(fixed_units, refs)) / total
    changed = sum(a != f for a, f in zip(asr, fixed_units))
    print(f"\n== {name}：{len(items)} 句｜ASR {e_asr:.1%} → LLM 修正 {e_llm:.1%}｜改咗 {changed} 句｜"
          f"每句 {elapsed / len(items) * 1000:.0f}ms（batch {args.batch}）")
    shown = 0
    for a, f, r in zip(asr, fixed_units, refs):
        if a != f and shown < 6:
            print(f"  參考：{join_units(r)}\n  ASR ：{join_units(a)}\n  LLM ：{join_units(f)}")
            shown += 1

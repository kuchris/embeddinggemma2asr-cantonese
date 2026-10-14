"""用 Gemma 4（llama-server）改正 gemma2asr 嘅轉錄，兩種方式：
  text ：淨係睇草稿改錯字
  audio：聽錄音 + 睇草稿，對住錄音改錯字（同音字靠聽）

同 llm_correct.py／eval_gemma4.py 同一批句子（seed 0，每個測試集 300 句）。

先開 server：tools/llama.cpp/llama-server.exe -m <gguf> --mmproj <mmproj> -ngl 99 --port 8099 -c 16384 -np 8
用法：python scripts/gemma4_correct.py [--n 300] [--parallel 8]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from g2asr import env  # noqa: F401  # 必須最先 import

import argparse
import base64
import io
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

from g2asr.encoder import Encoder
from g2asr.infer import load_reader, transcribe_batch
from g2asr.spice import clean, norm
from g2asr.text import join_units, units

ROOT = Path(__file__).resolve().parent.parent
RULES = ("規則：只改明顯錯嘅字（多數係同音字或者近音字）；保留廣東話口語寫法，唔好改做書面語；"
         "英文字保留英文，唔好翻譯；唔好加減內容；唔使加標點；冇錯就原句照抄。只輸出改正後嘅句子。")
TEXT_PROMPT = "下面係廣東話語音辨識嘅草稿，可能有錯字。\n" + RULES + "\n\n草稿：{draft}"
AUDIO_PROMPT = "上面係一段廣東話錄音。下面係語音辨識嘅草稿，可能有錯字。請對住錄音改正草稿。\n" + RULES + "\n\n草稿：{draft}"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "runs" / "learned_r4" / "best_infer.pt"))
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--port", type=int, default=8099)
parser.add_argument("--parallel", type=int, default=8)
args = parser.parse_args()
random.seed(0)


def load_set(folder: Path, ref_fn, keep=None):
    refs = json.loads((folder / "refs.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in refs if (keep is None or keep(k)) and (folder / f"{k}.wav").exists())
    keys = random.sample(keys, min(args.n, len(keys)))
    return [(folder / f"{k}.wav", ref_fn(refs[k]["text"])) for k in keys]


sets = {
    "daily+mixed（原本 test）": load_set(ROOT / "data" / "testset", units, keep=lambda k: not k.startswith("wordshk")),
    "MDCC test": load_set(ROOT / "data" / "mdcc_test2000", units),
    "SpiCE 對話": [x for t in ("VF19A", "VM19A")
                   for x in load_set(ROOT / "data" / "spice" / "prepared" / t / "seg", clean)][:args.n],
}

enc = Encoder()
reader, vocab = load_reader(args.ckpt)
drafts = {name: [join_units(h) for h in transcribe_batch(enc, reader, vocab,
                                                         [sf.read(p, dtype="float32")[0] for p, _ in items])]
          for name, items in sets.items()}
del enc, reader
torch.cuda.empty_cache()


def ask(path: Path, draft: str, with_audio: bool) -> str:
    content = []
    if with_audio:
        wav, sr = sf.read(path, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, wav[: 30 * sr], sr, format="WAV", subtype="PCM_16")
        content.append({"type": "input_audio",
                        "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}})
    content.append({"type": "text", "text": (AUDIO_PROMPT if with_audio else TEXT_PROMPT).format(draft=draft)})
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{args.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    return (msg.get("content") or "").strip().split("\n")[0]


for name, items in sets.items():
    to_units = (lambda t: norm(units(t))) if name.startswith("SpiCE") else units
    refs = [r for _, r in items]
    total = sum(map(len, refs))
    cer = lambda hyps: sum(Levenshtein.distance(to_units(h), r) for h, r in zip(hyps, refs)) / total
    line = f"\n== {name}：{len(items)} 句｜ASR 草稿 {cer(drafts[name]):.1%}"
    outs = {}
    for mode, with_audio in (("淨文字", False), ("聽聲+草稿", True)):
        t0 = time.time()
        with ThreadPoolExecutor(args.parallel) as pool:
            outs[mode] = list(pool.map(lambda x: ask(x[0][0], x[1], with_audio), zip(items, drafts[name])))
        line += f"｜{mode} {cer(outs[mode]):.1%}（每句 {(time.time() - t0) / len(items) * 1000:.0f}ms）"
    print(line)
    shown = 0
    for d, a, r in zip(drafts[name], outs["聽聲+草稿"], refs):
        if to_units(d) != to_units(a) and shown < 6:
            print(f"  參考：{join_units(r)}\n  草稿：{d}\n  修正：{a}")
            shown += 1

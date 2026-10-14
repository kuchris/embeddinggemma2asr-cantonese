"""Gemma 4（E2B／E4B）經 llama-server 做廣東話轉錄，喺同一批測試集計 CER 同速度。

先開 server（tools/llama.cpp，GGUF + mmproj）：
  tools/llama.cpp/llama-server.exe -m gemma-4-E2B-it-Q4_K_M.gguf --mmproj mmproj-F16.gguf -ngl 99 --port 8099 -c 4096 -np 4

用法：python scripts/eval_gemma4.py [--n 300] [--port 8099] [--parallel 4]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import base64
import io
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from rapidfuzz.distance import Levenshtein

from g2asr.spice import clean, norm
from g2asr.text import join_units, units

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("Transcribe the following speech segment in Cantonese into Cantonese text, using colloquial written "
          "Cantonese characters (e.g. 係, 喺, 嘅, 唔, 咗). Keep English words in English. "
          "Only output the transcription, with no newlines.")

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--port", type=int, default=8099)
parser.add_argument("--parallel", type=int, default=4)
args = parser.parse_args()
random.seed(0)


def load_set(folder: Path, ref_fn, keep=None):
    refs = json.loads((folder / "refs.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in refs if (keep is None or keep(k)) and (folder / f"{k}.wav").exists())
    keys = random.sample(keys, min(args.n, len(keys)))
    return [(folder / f"{k}.wav", ref_fn(refs[k]["text"])) for k in keys]


# 同 llm_correct.py 一樣嘅抽法（同一個 seed、同一個次序），所以係同一批句子
sets = {
    "daily+mixed（原本 test）": load_set(ROOT / "data" / "testset", units, keep=lambda k: not k.startswith("wordshk")),
    "MDCC test": load_set(ROOT / "data" / "mdcc_test2000", units),
    "SpiCE 對話": [x for t in ("VF19A", "VM19A")
                   for x in load_set(ROOT / "data" / "spice" / "prepared" / t / "seg", clean)][:args.n],
}


def transcribe(path: Path) -> str:
    wav, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav[: 30 * sr], sr, format="WAV", subtype="PCM_16")  # Gemma 4 每段最多 30 秒
    body = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}},
        {"type": "text", "text": PROMPT}]}],
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{args.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    return (msg.get("content") or "").strip()


for name, items in sets.items():
    is_spice = name.startswith("SpiCE")
    to_units = (lambda t: norm(units(t))) if is_spice else units
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        outs = list(pool.map(transcribe, [p for p, _ in items]))
    elapsed = time.time() - t0
    refs = [r for _, r in items]
    hyps = [to_units(o) for o in outs]
    total = sum(map(len, refs))
    cer = sum(Levenshtein.distance(h, r) for h, r in zip(hyps, refs)) / total
    audio_sec = sum(sf.info(p).duration for p, _ in items)
    print(f"\n== {name}：{len(items)} 句｜Gemma 4 CER {cer:.1%}｜{audio_sec / 60:.1f} 分鐘用 {elapsed:.0f}s"
          f"（{audio_sec / elapsed:.0f}× 即時，並行 {args.parallel}）")
    for h, r in list(zip(hyps, refs))[:5]:
        print(f"  參考：{join_units(r)}\n  輸出：{join_units(h)}")

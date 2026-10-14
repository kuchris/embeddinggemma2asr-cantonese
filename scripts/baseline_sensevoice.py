"""SenseVoice-Small（FunASR，GPU）喺同一批測試集嘅準確度同速度，同 gemma2asr 公平比較。

需要：uv sync --extra baseline（模型 FunAudioLLM/SenseVoiceSmall 會落載去 models/hf）
測試集要先由其他 script 準備好：
  data/testset/               ← export_testset.py
  data/mdcc_test2000/         ← eval_external.py --export
  data/spice/prepared/*/seg、vad  ← spice_eval.py --prepare

用法：uv run --extra baseline python scripts/baseline_sensevoice.py [--batch 16]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from g2asr import env  # noqa: F401  # 必須最先 import

import argparse
import json
import re
import time
from collections import defaultdict

import soundfile as sf
import torch
from funasr import AutoModel
from huggingface_hub import snapshot_download
from rapidfuzz.distance import Levenshtein

from g2asr.spice import clean, norm
from g2asr.text import units

DATA = Path(__file__).resolve().parent.parent / "data"
TAG = re.compile(r"<\|[^|]*\|>")

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

path = snapshot_download("FunAudioLLM/SenseVoiceSmall",
                         allow_patterns=["model.pt", "config.yaml", "am.mvn", "*.bpe.model", "configuration.json"])
model = AutoModel(model=path, device="cuda:0", disable_update=True)


def transcribe(files: list[Path]) -> tuple[list[str], float]:
    wavs = [sf.read(f, dtype="float32")[0] for f in files]
    model.generate(input=wavs[:2], language="auto", use_itn=False, batch_size=2)  # 預熱
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(input=wavs, language="auto", use_itn=False, batch_size=args.batch)
    torch.cuda.synchronize()
    return [TAG.sub("", r["text"]) for r in out], time.time() - t0


def score(name: str, folder: Path, ref_fn, hyp_fn=units, group=lambda k: "合計", keep=None):
    refs = json.loads((folder / "refs.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in refs if keep is None or k in keep)
    hyps, elapsed = transcribe([folder / f"{k}.wav" for k in keys])
    err, tot = defaultdict(int), defaultdict(int)
    for k, h in zip(keys, hyps):
        r = ref_fn(refs[k]["text"])
        for g in {group(k), "合計"}:
            err[g] += Levenshtein.distance(hyp_fn(h), r)
            tot[g] += len(r)
    audio_min = sum(sf.info(folder / f"{k}.wav").duration for k in keys) / 60
    parts = "  ".join(f"{g} {err[g] / tot[g]:.1%}" for g in sorted(tot) if g != "合計")
    print(f"{name}：CER {err['合計'] / tot['合計']:.2%}  {parts}  （{audio_min:.1f} 分鐘用 {elapsed:.1f}s，"
          f"{audio_min * 60 / elapsed:.0f}× 即時）")


score("原本 test 1030 句", DATA / "testset", units, group=lambda k: k.split("-")[0])
new_ids = set(json.load(open(DATA / "mdcc_test2000" / "new_ids.json")))
score("MDCC test 2000 句", DATA / "mdcc_test2000", units)
score("MDCC test（剔走重複句）", DATA / "mdcc_test2000", units, keep=new_ids)

# SpiCE：逐句同 VAD 段（長音頻參考字幕係成段串埋）
for mode in ("seg", "vad"):
    err = tot = 0
    minutes = elapsed_all = 0.0
    for talker in ("VF19A", "VM19A"):
        folder = DATA / "spice" / "prepared" / talker
        seg_refs = json.loads((folder / "seg" / "refs.json").read_text(encoding="utf-8"))
        files = sorted((folder / mode).glob("*.wav"))
        hyps, elapsed = transcribe(files)
        if mode == "seg":
            refs = [clean(seg_refs[f.stem]["text"]) for f in files]
            err += sum(Levenshtein.distance(norm(units(h)), r) for h, r in zip(hyps, refs))
            tot += sum(map(len, refs))
        else:
            ref = [u for k in sorted(seg_refs) for u in clean(seg_refs[k]["text"])]
            err += Levenshtein.distance(norm([u for h in hyps for u in units(h)]), ref)
            tot += len(ref)
        minutes += sum(sf.info(f).duration for f in files) / 60
        elapsed_all += elapsed
    print(f"SpiCE {mode}：CER {err / tot:.2%}（{minutes:.1f} 分鐘用 {elapsed_all:.1f}s，{minutes * 60 / elapsed_all:.0f}× 即時）")

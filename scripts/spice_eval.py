"""SpiCE 真人訪談（人手校正字幕）長音頻測試。

兩種模式，都只用「interview」部分、受訪者聲道：
  逐句：按 utterance tier 切開逐句轉錄（乾淨，量純辨識能力）
  長音頻：成段每 30 秒切一段連續轉錄（似真實會議用法；訪問者聲音漏入、靜音都會影響）

--prepare 會將兩種模式嘅音頻同參考字幕存去 data/spice/prepared/（畀 SenseVoice 等其他 ASR 用）。

用法：python scripts/spice_eval.py runs/learned_r3/best.pt --lm data/lm_r3.pkl [--prepare]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

from g2asr.decode import beam_search
from g2asr.encoder import SAMPLE_RATE, Encoder
from g2asr.lm import NGramLM
from g2asr.reader import Reader
from g2asr.spice import clean, norm, tier
from g2asr.text import join_units
from g2asr.vad import speech_segments

SPICE = Path(__file__).resolve().parent.parent / "data" / "spice"
CHUNK_SEC = 30

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="*")
parser.add_argument("--lm")
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--vad-threshold", type=float, default=0.5)
args = parser.parse_args()


talkers = {}
for tg in sorted(SPICE.glob("*_Cantonese_*.TextGrid")):
    text = tg.read_text(encoding="utf-16")
    start, end = [(a, b) for a, b, t in tier(text, "task") if t == "interview"][0]
    utts = [(a, b, t) for a, b, t in tier(text, "utterance") if t and a >= start and b <= end]
    audio, sr = sf.read(tg.with_suffix(".wav"), dtype="float32")
    # 受訪者聲道：utterance 時段入面能量較大嗰個
    mask = np.zeros(len(audio), bool)
    for a, b, _ in utts:
        mask[int(a * sr):int(b * sr)] = True
    ch = int(np.argmax([np.mean(audio[mask, c] ** 2) for c in range(audio.shape[1])]))
    wav = librosa.resample(audio[:, ch], orig_sr=sr, target_sr=SAMPLE_RATE)
    talkers[tg.stem.split("_")[0]] = (wav, start, end, utts)
    print(f"{tg.stem}：interview {end - start:.0f}s，{len(utts)} 句，受訪者聲道 {ch}")

if args.prepare:
    for name, (wav, start, end, utts) in talkers.items():
        out = SPICE / "prepared" / name
        (out / "seg").mkdir(parents=True, exist_ok=True)
        (out / "long").mkdir(parents=True, exist_ok=True)
        refs = {}
        for i, (a, b, t) in enumerate(utts):
            sf.write(out / "seg" / f"{i:04d}.wav", wav[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], SAMPLE_RATE)
            refs[f"{i:04d}"] = {"text": t}
        (out / "seg" / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
        seg = wav[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        for i in range(0, len(seg), CHUNK_SEC * SAMPLE_RATE):
            sf.write(out / "long" / f"{i // (CHUNK_SEC * SAMPLE_RATE):04d}.wav", seg[i:i + CHUNK_SEC * SAMPLE_RATE], SAMPLE_RATE)
        (out / "long" / "ref.txt").write_text(" ".join(t for _, _, t in utts), encoding="utf-8")
        (out / "vad").mkdir(parents=True, exist_ok=True)
        for old in (out / "vad").glob("*.wav"):
            old.unlink()
        for i, (a, b) in enumerate(speech_segments(seg, args.vad_threshold)):
            sf.write(out / "vad" / f"{i:04d}.wav", seg[a:b], SAMPLE_RATE)
    print(f"已存去 {SPICE / 'prepared'}")

if not args.ckpts:
    sys.exit()

enc = Encoder()
lm = NGramLM.load(args.lm) if args.lm else None


def decode(model, ck, wavs):
    """一批音頻 → (greedy 單位, LM 單位)。"""
    g_out, l_out = [], []
    for i in range(0, len(wavs), 8):
        for f in enc.frames(wavs[i:i + 8], model.feat_layers):
            lengths = torch.tensor([len(f)]).cuda()
            with torch.no_grad():
                lp = model(f[None].cuda().float(), lengths)
            g_out.append([ck["units"][j] for j in Reader.greedy(lp, lengths)[0]])
            if lm:
                l_out.append([ck["units"][j] for j in beam_search(lp[0].float().cpu().numpy(), ck["units"], lm, 0.7, 4.5)])
    return g_out, l_out


for path in args.ckpts:
    ck = torch.load(path, weights_only=False)
    model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
    model.load_state_dict(ck["model"])
    print(f"\n== {path}")
    total = {"seg": [0, 0, 0], "long": [0, 0, 0], "vad": [0, 0, 0]}
    for name, (wav, start, end, utts) in talkers.items():
        segs = [wav[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)] for a, b, _ in utts]
        refs = [clean(t) for _, _, t in utts]
        g, l = decode(model, ck, segs)
        g = [norm(h) for h in g]
        l = [norm(h) for h in l]
        e_g = sum(Levenshtein.distance(h, r) for h, r in zip(g, refs))
        e_l = sum(Levenshtein.distance(h, r) for h, r in zip(l, refs)) if lm else 0
        n = sum(map(len, refs))
        total["seg"] = [total["seg"][0] + e_g, total["seg"][1] + e_l, total["seg"][2] + n]

        seg = wav[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        chunks = [seg[i:i + CHUNK_SEC * SAMPLE_RATE] for i in range(0, len(seg), CHUNK_SEC * SAMPLE_RATE)]
        t0 = time.time()
        g_long, l_long = decode(model, ck, chunks)
        elapsed = time.time() - t0
        ref_long = [u for r in refs for u in r]
        hyp_g = norm([u for h in g_long for u in h])
        hyp_l = norm([u for h in l_long for u in h])
        le_g = Levenshtein.distance(hyp_g, ref_long)
        le_l = Levenshtein.distance(hyp_l, ref_long) if lm else 0
        total["long"] = [total["long"][0] + le_g, total["long"][1] + le_l, total["long"][2] + len(ref_long)]

        # VAD：只轉錄有人講嘢嘅段落
        t1 = time.time()
        vad_segs = [seg[a:b] for a, b in speech_segments(seg, args.vad_threshold)]
        g_vad, l_vad = decode(model, ck, vad_segs)
        vad_time = time.time() - t1
        ve_g = Levenshtein.distance(norm([u for h in g_vad for u in h]), ref_long)
        ve_l = Levenshtein.distance(norm([u for h in l_vad for u in h]), ref_long) if lm else 0
        total["vad"] = [total["vad"][0] + ve_g, total["vad"][1] + ve_l, total["vad"][2] + len(ref_long)]
        print(f"  {name} VAD：{len(vad_segs)} 段，共 {sum(map(len, vad_segs)) / SAMPLE_RATE / 60:.1f} 分鐘"
              f"｜greedy {ve_g / len(ref_long):.1%}" + (f" +LM {ve_l / len(ref_long):.1%}" if lm else "")
              + f"（用 {vad_time:.1f}s，連 VAD）")
        print(f"  {name}：逐句 greedy {e_g / n:.1%}" + (f" +LM {e_l / n:.1%}" if lm else "")
              + f"｜長音頻 greedy {le_g / len(ref_long):.1%}" + (f" +LM {le_l / len(ref_long):.1%}" if lm else "")
              + f"（{len(seg) / SAMPLE_RATE / 60:.1f} 分鐘用 {elapsed:.1f}s）")
        print(f"    例：參考 {join_units(refs[5])}\n        輸出 {join_units(l[5] if lm else g[5])}")
    for mode, (eg, el, n) in total.items():
        print(f"  合計 {mode}：greedy {eg / n:.2%}" + (f"，+LM {el / n:.2%}" if lm else ""))

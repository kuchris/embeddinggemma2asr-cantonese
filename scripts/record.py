"""錄真人測試集：逐句顯示，撳 Enter 開始錄，再撳 Enter 停。存去 real/（wav + refs.json）。

之後：python scripts/transcribe.py runs/frozen/best.pt real/*.wav [--lm data/lm.pkl]

用法：uv run --with sounddevice python scripts/record.py [--out real] [--sentences my.txt]
"""

import argparse
import json
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
SENTENCES = [
    "喂，你食咗飯未呀？",
    "我今日放工之後要去超級市場買啲嘢。",
    "唔該你幫我拎嗰杯奶茶過嚟吖。",
    "聽日落雨嘅話，我哋就改期啦。",
    "佢話琴晚瞓得唔好，成日都好眼瞓。",
    "呢間餐廳嘅叉燒飯真係好好食。",
    "你搭港鐵定係巴士返屋企呀？",
    "我啱啱send咗個email畀你，你check下先。",
    "今個weekend有冇時間一齊去行山？",
    "部電腦又死機，我要成個restart過。",
    "唔好意思，我遲咗十分鐘到。",
    "你覺得呢套戲好唔好睇呀？",
    "阿媽叫我早啲返屋企食飯。",
    "我哋公司下個月要搬去觀塘。",
    "其實我都唔係好明佢想點。",
    "你記唔記得我哋第一次見面係喺邊度？",
    "等陣我打電話畀你再傾。",
    "呢排天氣好凍，記得着多件衫。",
    "我想訂兩位，今晚七點半，得唔得？",
    "幫我同佢講聲生日快樂。",
]

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="real")
parser.add_argument("--sentences", help="自己嘅句子檔，每行一句")
args = parser.parse_args()

sentences = SENTENCES
if args.sentences:
    sentences = [l.strip() for l in open(args.sentences, encoding="utf-8") if l.strip()]
out = Path(args.out)
out.mkdir(exist_ok=True)
refs_path = out / "refs.json"
refs = json.loads(refs_path.read_text(encoding="utf-8")) if refs_path.exists() else {}

print(f"咪：{sd.query_devices(kind='input')['name']}\n用平時講嘢嘅語氣讀，唔使讀得好標準。打 s 跳過，q 離開。\n")
for i, text in enumerate(sentences, 1):
    name = f"real-{i:02d}"
    if name in refs:
        continue
    cmd = input(f"[{i}/{len(sentences)}] {text}\n  Enter 開始錄 ").strip().lower()
    if cmd == "q":
        break
    if cmd == "s":
        continue
    q = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=lambda d, *_: q.put(d.copy())):
        input("  🔴 錄緊…講完撳 Enter ")
    audio = np.concatenate(list(q.queue))[:, 0]
    sf.write(out / f"{name}.wav", audio, SAMPLE_RATE)
    refs[name] = {"text": text, "sec": round(len(audio) / SAMPLE_RATE, 1)}
    refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✓ {len(audio) / SAMPLE_RATE:.1f} 秒\n")

print(f"錄咗 {len(refs)} 句，喺 {out}/")

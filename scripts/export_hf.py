"""將 train 完嘅 checkpoint 轉做發佈用格式：model.safetensors + config.json（唔使 pickle）。

剔走 optimizer 狀態同粵拼 head（淨係 train 用）；PCA 參數（4 層 feature 用）係 buffer，會跟住存。
載入：g2asr.infer.load_reader("<資料夾>")

用法：python scripts/export_hf.py runs/learned_r6/best.pt --out hf/gemma2asr-cantonese
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import torch
from safetensors.torch import save_file

from g2asr.encoder import MODEL_ID
from g2asr.reader import EMB_DIM

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("--out", required=True)
args = parser.parse_args()

ck = torch.load(args.ckpt, weights_only=False)
config = dict(ck["config"], jy_classes=0)  # 推理唔使粵拼 head
state = {k: v.contiguous() for k, v in ck["model"].items() if not k.startswith("jy_head.")}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
save_file(state, out / "model.safetensors")
meta = {
    "architecture": "gemma2asr-reader",
    "base_model": MODEL_ID,
    "sample_rate": 16000,
    "frame_sec": 0.04,
    "emb_dim": EMB_DIM,
    "vocab_size": len(ck["units"]),
    "reader": {k: (list(v) if isinstance(v, tuple) else
                   {str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in config.items()},
    "units": ck["units"],
    "train": {"step": ck.get("step"), "epoch": ck.get("epoch"), "best_val_cer": ck.get("best_cer")},
}
(out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
size = (out / "model.safetensors").stat().st_size / 1e6
print(f"已存：{out}（model.safetensors {size:.0f}MB，字典 {len(ck['units'])}，feature 層 {config.get('feat_layers')}）")

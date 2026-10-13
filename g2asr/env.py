"""將模型 cache 指向專案 models/ 資料夾。必須喺 import transformers／huggingface_hub 之前 import。

Colab 上面可以用 G2ASR_MODELS 指去 Drive，費事每次重新落載。
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("G2ASR_MODELS", ROOT / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hf" / "hub"))

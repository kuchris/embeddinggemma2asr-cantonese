"""EmbeddingGemma 2 包裝：攞逐格 audio feature、成段 audio embedding、文字 embedding。

模型結構（google/embeddinggemma-2）：
  audio_tower（gemma4_audio，約 300M）→ embed_audio（投射去 512 維 LM 輸入空間，每 40ms 一格）
  → language_model（24 層，512 維）→ embedding_projection（768 維，逐 token）→ mean pooling → L2 normalize

reader 嘅輸入係 embed_audio 之後嗰啲逐格 512 維 feature（frames），
訓練目標係 768 維輸出空間入面每個字嘅 text embedding。
"""

from . import env  # noqa: F401  # 必須最先 import

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/embeddinggemma-2"
SAMPLE_RATE = 16000
FRAME_SEC = 0.04  # 每格 audio feature 40ms（25 格／秒）
AUDIO_LAYERS = 12  # audio_tower 有 12 層 conformer（1024 維）


def layer_dim(layer: int) -> int:
    """0 = embed_audio 之後 512 維；1–11 = 第幾層 conformer 輸出 1024 維；12 = 最後一層再經 output_proj，1536 維。"""
    return 512 if layer == 0 else 1536 if layer == AUDIO_LAYERS else 1024


def pick_dtype() -> torch.dtype:
    # T4 冇原生 bf16；fp16 有機會溢出，所以退返 fp32（encoder 得 300M，頂得順）
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class Encoder:
    def __init__(self, device: str = "cuda", dtype: torch.dtype | None = None, with_audio: bool = True):
        self.device = device
        self.dtype = dtype or pick_dtype()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        overrides = {"vision_config": None}  # 唔使 vision encoder，慳 170M
        if not with_audio:
            overrides["audio_config"] = None
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=self.dtype, **overrides).to(device).eval()

    # ---------- 逐格 feature（reader 嘅輸入） ----------

    @torch.inference_mode()
    def frames(self, wavs: list[np.ndarray], layers=(0,), pca: dict | None = None) -> list[torch.Tensor]:
        """每段 16kHz mono 音頻 → (T, D) 逐格 feature（已去 padding，放喺 CPU，fp16）。

        layers：要邊幾層，按次序駁埋。0 = embed_audio 之後嘅 512 維（原本用開嗰種）；
        1–11 = audio_tower 第幾層 conformer 嘅輸出（1024 維）；12 = 最後一層經 output_proj（1536 維）。
        （hidden_states 有 13 個：[0] 係入第 1 層之前，[1]–[11] 係第 1–11 層，[12] = last_hidden_state。）
        pca：{層: (mean (D,), comps (D, k))}，有就將嗰層壓縮做 k 維（(x − mean) @ comps）。
        """
        layers = tuple(layers)
        fe = self.processor.feature_extractor(
            wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest")
        out = self.model.get_audio_features(
            fe["input_features"].to(self.device, self.dtype),
            fe["input_features_mask"].to(self.device), return_dict=True,
            output_hidden_states=any(l > 0 for l in layers))
        hs = out.hidden_states
        parts = []
        for l in layers:
            x = out.pooler_output if l == 0 else hs[l]
            if pca and l in pca:
                mean, comps = pca[l]
                x = (x.float() - mean.to(x.device)) @ comps.to(x.device)
            parts.append(x.float())
        feats = torch.cat(parts, -1)
        return [f[m].to("cpu", torch.float16) for f, m in zip(feats, out.attention_mask)]

    # ---------- 768 維 embedding（同 sentence-transformers 一樣：mean pooling + normalize） ----------

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled.float(), dim=-1)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], prompt: str = "", batch_size: int = 256) -> torch.Tensor:
        """文字 → (N, 768) normalized embedding（放喺 CPU，fp32）。"""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [prompt + t for t in texts[i:i + batch_size]]
            tok = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            hidden = self.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
            out.append(self._pool(hidden, tok["attention_mask"]).cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def embed_audio(self, wavs: list[np.ndarray]) -> torch.Tensor:
        """成段音頻 → (N, 768) normalized embedding（同模型原本做 audio 檢索嘅方法一樣）。"""
        out = []
        for wav in wavs:  # 逐段做，費事 processor 處理唔同長度嘅 padding 出錯
            inputs = self.processor(text=[self.processor.audio_token], audio=[wav],
                                    sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            hidden = self.model(**inputs).last_hidden_state
            out.append(self._pool(hidden, inputs["attention_mask"]).cpu())
        return torch.cat(out)

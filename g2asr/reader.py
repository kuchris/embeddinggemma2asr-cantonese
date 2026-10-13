"""Reader：將逐格 audio feature（512 維）推去 768 維文字空間，再同字典每個字嘅 text embedding 比 cosine。

head 模式：
  frozen  — 字典 embedding 固定（原本方法，只 train reader）
  tuned   — 用 text embedding 做起點，但一齊 train（睇吓文字空間有幾大幫助）
  learned — 隨機初始化（普通 CTC 分類頭，做對照組）
輸出第 0 類係 CTC blank。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM, EMB_DIM = 512, 768


class Reader(nn.Module):
    def __init__(self, vocab_emb: torch.Tensor, d_model: int = 384, layers: int = 6, heads: int = 6,
                 dropout: float = 0.1, head: str = "frozen"):
        super().__init__()
        self.config = dict(d_model=d_model, layers=layers, heads=heads, dropout=dropout, head=head)
        self.inp = nn.Sequential(nn.LayerNorm(IN_DIM), nn.Linear(IN_DIM, d_model))
        # depthwise conv 做相對位置（Transformer 本身冇位置資訊）
        self.pos_conv = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, EMB_DIM)

        vocab_emb = F.normalize(vocab_emb.float(), dim=-1)
        if head == "frozen":
            self.register_buffer("vocab", vocab_emb)
        elif head == "tuned":
            self.vocab = nn.Parameter(vocab_emb.clone())
        elif head == "learned":
            self.vocab = nn.Parameter(torch.randn_like(vocab_emb) * 0.02)
        else:
            raise ValueError(head)
        self.blank = nn.Parameter(torch.randn(EMB_DIM) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, 512), (B,) → (B, T, V+1) log-probs。"""
        pad = torch.arange(feats.shape[1], device=feats.device)[None] >= lengths[:, None]
        x = self.inp(feats)
        x = x + self.pos_conv(x.masked_fill(pad[..., None], 0).transpose(1, 2)).transpose(1, 2)
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        y = F.normalize(self.out(x).float(), dim=-1)
        classes = F.normalize(torch.cat([self.blank[None], self.vocab]).float(), dim=-1)
        logits = self.log_scale.exp() * y @ classes.T
        return logits.log_softmax(-1)

    @staticmethod
    def greedy(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """CTC greedy 解碼 → 每句嘅字 id（已經減 1，對返 vocab index）。"""
        best = log_probs.argmax(-1).cpu()
        out = []
        for seq, n in zip(best, lengths.tolist()):
            ids, prev = [], 0
            for t in seq[:n].tolist():
                if t != prev and t != 0:
                    ids.append(t - 1)
                prev = t
            out.append(ids)
        return out

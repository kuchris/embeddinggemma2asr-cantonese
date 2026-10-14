"""Reader：將逐格 audio feature（512 維）推去 768 維文字空間，再同字典每個字嘅 text embedding 比 cosine。

head 模式：
  frozen  — 字典 embedding 固定（原本方法，只 train reader）
  tuned   — 用 text embedding 做起點，但一齊 train（睇吓文字空間有幾大幫助）
  learned — 隨機初始化（普通 CTC 分類頭，做對照組）
輸出第 0 類係 CTC blank。

feat_layers：輸入用 encoder 邊幾層（見 Encoder.frames）。多過一層就每層各自 LayerNorm + 投射，
再用一組可以學嘅權重（softmax）加埋（SUPERB 式 weighted layer sum）。
jy_classes > 0：喺第 jy_layer 層後面加一個粵拼 CTC head（輔助訓練，推理唔使用）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import layer_dim

EMB_DIM = 768


class Reader(nn.Module):
    def __init__(self, vocab_emb: torch.Tensor, d_model: int = 384, layers: int = 6, heads: int = 6,
                 dropout: float = 0.1, head: str = "frozen", feat_layers=(0,), jy_classes: int = 0,
                 jy_layer: int = 3):
        super().__init__()
        self.feat_layers = tuple(feat_layers)
        self.config = dict(d_model=d_model, layers=layers, heads=heads, dropout=dropout, head=head,
                           feat_layers=self.feat_layers, jy_classes=jy_classes, jy_layer=jy_layer)
        self.in_dims = [layer_dim(l) for l in self.feat_layers]
        if len(self.in_dims) == 1:  # 同舊 checkpoint 一樣嘅結構
            self.inp = nn.Sequential(nn.LayerNorm(self.in_dims[0]), nn.Linear(self.in_dims[0], d_model))
        else:
            self.inp_mix = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_model)) for d in self.in_dims)
            self.mix_logits = nn.Parameter(torch.zeros(len(self.in_dims)))
        self.jy_layer = jy_layer
        self.jy_head = nn.Linear(d_model, jy_classes + 1) if jy_classes else None
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

    def mix_weights(self) -> list[float]:
        return self.mix_logits.softmax(0).tolist() if len(self.in_dims) > 1 else [1.0]

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor, return_jy: bool = False):
        """(B, T, D), (B,) → (B, T, V+1) log-probs；return_jy 就再多回傳粵拼 log-probs (B, T, J+1)。"""
        pad = torch.arange(feats.shape[1], device=feats.device)[None] >= lengths[:, None]
        if len(self.in_dims) == 1:
            x = self.inp(feats)
        else:
            w = self.mix_logits.softmax(0)
            parts = feats.split(self.in_dims, -1)
            x = sum(w[i] * proj(part) for i, (proj, part) in enumerate(zip(self.inp_mix, parts)))
        x = x + self.pos_conv(x.masked_fill(pad[..., None], 0).transpose(1, 2)).transpose(1, 2)
        jy = None
        for i, layer in enumerate(self.encoder.layers, 1):
            x = layer(x, src_key_padding_mask=pad)
            if return_jy and i == self.jy_layer:
                jy = self.jy_head(x).float().log_softmax(-1)
        x = self.norm(x)
        y = F.normalize(self.out(x).float(), dim=-1)
        classes = F.normalize(torch.cat([self.blank[None], self.vocab]).float(), dim=-1)
        logits = self.log_scale.exp() * y @ classes.T
        return (logits.log_softmax(-1), jy) if return_jy else logits.log_softmax(-1)

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

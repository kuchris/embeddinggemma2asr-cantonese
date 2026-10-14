"""快速推理：按長度排序、encoder 同 reader 都成批行、bf16。

逐句行 reader 同埋長短句混埋做 batch（好多 padding）會慢兩三倍。
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .decode import beam_search, beam_search_nbest
from .encoder import Encoder
from .lm import NGramLM
from .reader import Reader

BATCH = 16  # 再大反而慢（長句多時頂爆 VRAM）


def load_reader(ckpt_path: str) -> tuple[Reader, list[str]]:
    ck = torch.load(ckpt_path, weights_only=False)
    model = Reader(ck["vocab_emb"], **ck["config"]).cuda().eval()
    model.load_state_dict(ck["model"])
    return model, ck["units"]


@torch.inference_mode()
def transcribe_batch(enc: Encoder, model: Reader, units: list[str], wavs: list[np.ndarray],
                     lm: NGramLM | None = None, alpha: float = 0.7, beta: float = 4.5,
                     batch: int = BATCH) -> list[list[str]]:
    """多段 16kHz 音頻 → 每段嘅單位（次序同輸入一樣）。"""
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    out: list[list[str]] = [[] for _ in wavs]
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        feats = enc.frames([wavs[i] for i in idx], model.feat_layers, model.pca())
        lengths = torch.tensor([len(f) for f in feats]).cuda()
        x = pad_sequence(feats, batch_first=True).cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            log_probs = model(x, lengths)
        if lm:
            for i, lp, n in zip(idx, log_probs.float().cpu().numpy(), lengths.tolist()):
                out[i] = [units[j] for j in beam_search(lp[:n], units, lm, alpha, beta)]
        else:
            for i, ids in zip(idx, Reader.greedy(log_probs, lengths)):
                out[i] = [units[j] for j in ids]
    return out


@torch.inference_mode()
def transcribe_nbest(enc: Encoder, model: Reader, units: list[str], wavs: list[np.ndarray], lm: NGramLM | None,
                     alpha: float = 0.7, beta: float = 4.5, beam: int = 8,
                     batch: int = BATCH) -> list[list[tuple[list[str], float]]]:
    """每段音頻 → beam 入面所有候選 [(單位, 分數)]，分數由高至低（俾外面嘅 LM 重新評分）。"""
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    out: list[list[tuple[list[str], float]]] = [[] for _ in wavs]
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        feats = enc.frames([wavs[i] for i in idx], model.feat_layers, model.pca())
        lengths = torch.tensor([len(f) for f in feats]).cuda()
        x = pad_sequence(feats, batch_first=True).cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            log_probs = model(x, lengths)
        for i, lp, n in zip(idx, log_probs.float().cpu().numpy(), lengths.tolist()):
            out[i] = [([units[j] for j in ids], score)
                      for ids, score in beam_search_nbest(lp[:n], units, lm, alpha, beta, beam)]
    return out

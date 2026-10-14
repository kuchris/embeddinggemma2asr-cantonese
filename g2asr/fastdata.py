"""快速訓練數據：shard 一次過載入 RAM、全部句子按長度分 batch、背景線程預先整好下一批。

逐個 shard 由硬碟讀（每個 epoch 重讀幾十 GB）加上主線程用 Python 砌 batch，會令 GPU 大部分時間等緊 CPU。
"""

import queue
import random
import threading

import torch

from .encoder import layer_dim


class Preloaded:
    """use_layers：shard 入面有多層 feature（shard["layers"]）嗰陣揀用邊幾層，只載入嗰幾層慳 RAM。
    jy_vocab：畀咗就順便整粵拼標籤（輔助 CTC），新音節會加入呢個 dict（grow=False 就略過）。"""

    def __init__(self, paths, unit_to_id: dict[str, int], use_layers=None, jy_vocab: dict | None = None,
                 grow: bool = True):
        self.shards: list[torch.Tensor] = []  # 每個 shard 一大塊連續 (總格數, D) fp16
        self.items: list[tuple] = []  # (shard, 開始格, 格數, 字 id, 粵拼 id 或 None)
        self.jy = jy_vocab is not None
        if self.jy:
            from .jyutping import jyutping_units  # 要裝 ToJyutping（uv sync --extra train）
        for path in paths:
            shard = torch.load(path)
            have = shard.get("layers", [0])
            self.dims = shard.get("dims") or [layer_dim(l) for l in have]
            cols = self._columns(have, self.dims, use_layers)
            offset, keep = 0, []
            for f, us in zip(shard["feats"], shard["units"]):
                ids = [unit_to_id[u] for u in us if u in unit_to_id]
                jy = None
                if self.jy:
                    jy = []
                    for j in jyutping_units(us):
                        if j not in jy_vocab and grow:
                            jy_vocab[j] = len(jy_vocab) + 1  # 0 留畀 blank
                        if j in jy_vocab:
                            jy.append(jy_vocab[j])
                    jy = torch.tensor(jy, dtype=torch.int32)
                if ids and len(f) >= 2 * len(ids):  # CTC 要夠格數
                    keep.append(f if cols is None else f[:, cols])
                    self.items.append((len(self.shards), offset, len(f), torch.tensor(ids, dtype=torch.int32), jy))
                    offset += len(f)
            self.shards.append(torch.cat(keep) if keep else torch.empty(0, 512, dtype=torch.float16))
            del shard, keep
        self.frames = sum(s.shape[0] for s in self.shards)

    @staticmethod
    def _columns(have: list[int], dims: list[int], want):
        """shard 有 have 呢幾層（每層 dims 維，按次序駁埋），想要 want 呢幾層 → 要攞嘅欄（全部要就 None）。"""
        if want is None or list(want) == list(have):
            return None
        starts, size, pos = {}, {}, 0
        for l, d in zip(have, dims):
            starts[l], size[l] = pos, d
            pos += d
        missing = [l for l in want if l not in starts]
        if missing:
            raise ValueError(f"shard 冇第 {missing} 層（只有 {have}）")
        return torch.cat([torch.arange(starts[l], starts[l] + size[l]) for l in want])

    def batches(self, max_frames: int, shuffle: bool, seed: int = 0) -> list[list[int]]:
        """全部句子按長度排序再切 batch（每批 padding 好少），再打亂 batch 次序。"""
        order = sorted(range(len(self.items)), key=lambda i: self.items[i][2])
        groups, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * self.items[i][2] > max_frames:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        if shuffle:
            random.Random(seed).shuffle(groups)
        return groups

    def collate(self, group: list[int]):
        lengths = torch.tensor([self.items[i][2] for i in group])
        feats = torch.zeros(len(group), int(lengths.max()), self.shards[0].shape[1], dtype=torch.float16)
        for row, i in enumerate(group):
            s, off, n = self.items[i][:3]
            feats[row, :n] = self.shards[s][off:off + n].to(torch.float16)  # 可能係 fp8
        targets = torch.cat([self.items[i][3] for i in group]).long()
        target_lengths = torch.tensor([len(self.items[i][3]) for i in group])
        if not self.jy:
            return feats.pin_memory(), lengths, targets, target_lengths
        jy = torch.cat([self.items[i][4] for i in group]).long()
        jy_lengths = torch.tensor([len(self.items[i][4]) for i in group])
        return feats.pin_memory(), lengths, targets, target_lengths, jy, jy_lengths


def prefetch(data: Preloaded, groups: list[list[int]], depth: int = 6):
    """背景線程砌 batch（同埋 pin memory），主線程 GPU 計緊上一批嗰陣下一批已經準備好。"""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        try:
            for g in groups:
                q.put(data.collate(g))
        except BaseException as e:  # 傳返畀主線程，唔係主線程會永遠等落去
            q.put(e)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (batch := q.get()) is not None:
        if isinstance(batch, BaseException):
            raise batch
        yield batch


def augment_gpu(feats: torch.Tensor, lengths: torch.Tensor, time_masks: int, time_width: int,
                feat_masks: int, feat_width: int) -> torch.Tensor:
    """SpecAugment 式遮罩，成批喺 GPU 一次過做（唔再逐句 Python 迴圈）。"""
    B, T, D = feats.shape
    dev = feats.device
    t = torch.arange(T, device=dev)
    mask = torch.zeros(B, T, D, dtype=torch.bool, device=dev)
    if time_masks:
        n = (time_masks * lengths.float() / 125).round().clamp(min=1).long()  # 每 5 秒 time_masks 段
        K = int(n.max())
        w = torch.randint(0, time_width + 1, (B, K), device=dev) * (torch.arange(K, device=dev) < n[:, None])
        s = (torch.rand(B, K, device=dev) * (lengths[:, None] - w).clamp(min=1)).long()
        tm = ((t[None, None] >= s[..., None]) & (t[None, None] < (s + w)[..., None])).any(1)  # (B, T)
        mask |= tm[:, :, None]
    if feat_masks:
        d = torch.arange(D, device=dev)
        w = torch.randint(0, feat_width + 1, (B, feat_masks), device=dev)
        s = (torch.rand(B, feat_masks, device=dev) * (D - w)).long()
        fm = ((d[None, None] >= s[..., None]) & (d[None, None] < (s + w)[..., None])).any(1)  # (B, D)
        mask |= fm[:, None, :]
    return feats.masked_fill(mask, 0)

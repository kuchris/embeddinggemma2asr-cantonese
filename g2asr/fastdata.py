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
    jy_vocab：畀咗就順便整粵拼標籤（輔助 CTC），新音節會加入呢個 dict（grow=False 就略過）。
    mmap：唔載入 RAM，用 torch.load(mmap=True) 由硬碟讀（數據大過 RAM 嗰陣用，靠 OS page cache）。"""

    def __init__(self, paths, unit_to_id: dict[str, int], use_layers=None, jy_vocab: dict | None = None,
                 grow: bool = True, mmap: bool = False):
        self.shards: list[torch.Tensor] = []  # 每個 shard 一大塊連續 (總格數, D) fp16
        self.mmap = mmap
        self.chunk_frames = 25_000_000  # mmap 模式每組約 2500 萬格（fp8 1024 維約 25GB，留位畀 page cache）
        self.item_feats: list[torch.Tensor] = []  # mmap 模式：每句自己嘅 feature（仲喺硬碟）
        self.cols = None
        self.paths = list(paths)
        self.chunk_plan: list[tuple[int, list[int]]] = []  # mmap：每組 (第一個 batch 位置, shard 編號)，畀 prefetch 預讀
        self.items: list[tuple] = []  # (shard, 開始格, 格數, 字 id, 粵拼 id 或 None)
        self.jy = jy_vocab is not None
        if self.jy:
            from .jyutping import jyutping_units  # 要裝 ToJyutping（uv sync --extra train）
        for path in paths:
            shard = torch.load(path, mmap=mmap)
            have = shard.get("layers", [0])
            self.dims = shard.get("dims") or [layer_dim(l) for l in have]
            cols = self.cols = self._columns(have, self.dims, use_layers)
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
                    if mmap:
                        self.item_feats.append(f)  # 揀欄留返 collate 先做，唔係會讀晒入 RAM
                    else:
                        keep.append(f if cols is None else f[:, cols])
                    self.items.append((len(self.shards), offset, len(f), torch.tensor(ids, dtype=torch.int32), jy))
                    offset += len(f)
            if not mmap:
                self.shards.append(torch.cat(keep) if keep else torch.empty(0, 512, dtype=torch.float16))
            else:
                self.shards.append(torch.empty(0))  # 佔位，令 items 嘅 shard 編號一致
            del shard, keep
        self.frames = sum(it[2] for it in self.items)
        self.dim = (len(self.cols) if self.cols is not None else sum(self.dims)) if mmap else self.shards[0].shape[1]

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
        """全部句子按長度排序再切 batch（每批 padding 好少），再打亂 batch 次序。

        mmap 模式：數據大過 RAM，全域隨機會令每個 batch 散落喺好多 shard，OS page cache 冇用（慢 3 倍）。
        所以先將 shard 隨機分組（每組約 chunk_frames 格，裝得落 RAM），一組做完先到下一組，組內照樣按長度分 batch 再打亂。
        """
        rng = random.Random(seed)
        if not self.mmap:
            chunks = [range(len(self.items))]
        else:
            by_shard: dict[int, list[int]] = {}
            for i, it in enumerate(self.items):
                by_shard.setdefault(it[0], []).append(i)
            shard_ids = list(by_shard)
            if shuffle:
                rng.shuffle(shard_ids)
            chunks, chunk_shards, cur, cur_s, size = [], [], [], [], 0
            for sid in shard_ids:
                cur += by_shard[sid]
                cur_s.append(sid)
                size += sum(self.items[i][2] for i in by_shard[sid])
                if size >= self.chunk_frames:
                    chunks.append(cur)
                    chunk_shards.append(cur_s)
                    cur, cur_s, size = [], [], 0
            if cur:
                chunks.append(cur)
                chunk_shards.append(cur_s)
            self.chunk_plan = []
        out = []
        for k, chunk in enumerate(chunks):
            if self.mmap:
                self.chunk_plan.append((len(out), chunk_shards[k]))
            groups, cur = [], []
            for i in sorted(chunk, key=lambda i: self.items[i][2]):
                if cur and (len(cur) + 1) * self.items[i][2] > max_frames:
                    groups.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                groups.append(cur)
            if shuffle:
                rng.shuffle(groups)
            out += groups
        return out

    def collate(self, group: list[int]):
        lengths = torch.tensor([self.items[i][2] for i in group])
        feats = torch.zeros(len(group), int(lengths.max()), self.dim, dtype=torch.float16)
        for row, i in enumerate(group):
            s, off, n = self.items[i][:3]
            if self.mmap:
                f = self.item_feats[i]
                feats[row, :n] = (f if self.cols is None else f[:, self.cols]).to(torch.float16)
            else:
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

    # mmap：做緊第 k 組嗰陣，另一條線程順序讀第 k+1 組嘅 shard 檔入 page cache（比 mmap 隨機讀快好多）
    plan = {start: sids for start, sids in data.chunk_plan} if data.mmap else {}
    starts = sorted(plan)

    def warm(sids):
        for sid in sids:
            with open(data.paths[sid], "rb") as f:
                while f.read(64 << 20):
                    pass

    def worker():
        try:
            if starts:
                threading.Thread(target=warm, args=(plan[starts[0]],), daemon=True).start()
            for n, g in enumerate(groups):
                if n in plan:
                    k = starts.index(n)
                    if k + 1 < len(starts):
                        threading.Thread(target=warm, args=(plan[starts[k + 1]],), daemon=True).start()
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

"""細 n-gram 語言模型（Witten-Bell 插值），單位同字典一樣（中文逐字、英文逐詞）。

用嚟喺 beam search 入面揀較通順嘅字（「老公」好過「路工」）。純 Python，唔使 KenLM。
"""

import math
import pickle
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, order: int = 4):
        self.order = order
        # counts[n][history][word]：history 長度 n（0 = unigram）
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.vocab_size = 0
        self._cache = {}

    def fit(self, sentences):
        vocab = set()
        for units in sentences:
            seq = [BOS] * (self.order - 1) + list(units) + [EOS]
            vocab.update(units)
            for i in range(self.order - 1, len(seq)):
                for n in range(self.order):
                    self.counts[n][tuple(seq[i - n:i])][seq[i]] += 1
        self.vocab_size = len(vocab) + 1  # 加 </s>
        # 預先計每個 history 嘅總數同種類數
        self.totals = [{h: (sum(c.values()), len(c)) for h, c in level.items()} for level in self.counts]
        return self

    def logp(self, history: tuple, word: str) -> float:
        """log P(word | 前面最多 order-1 個單位)。"""
        history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        if len(history) < self.order - 1:
            history = (BOS,) * (self.order - 1 - len(history)) + history
        key = (history, word)
        if key not in self._cache:
            p = 1.0 / self.vocab_size  # 均勻分佈做底
            for n in range(self.order):
                h = history[len(history) - n:] if n else ()
                total = self.totals[n].get(h)
                if not total:
                    continue
                c_h, types = total
                p = (self.counts[n][h][word] + types * p) / (c_h + types)
            self._cache[key] = math.log(p)
        return self._cache[key]

    def save(self, path):
        self._cache = {}
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "NGramLM":
        with open(path, "rb") as f:
            return pickle.load(f)

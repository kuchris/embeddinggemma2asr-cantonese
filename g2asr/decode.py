"""CTC prefix beam search，加 n-gram 語言模型（shallow fusion）。

分數 = CTC log 機率 + alpha × LM log 機率 + beta × 字數
alpha 越大越信 LM；beta 補償 LM 鍾意短句嘅傾向。
"""

import math

import numpy as np

from .lm import EOS, NGramLM

NEG_INF = -math.inf


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(log_probs: np.ndarray, units: list[str], lm: NGramLM | None = None, alpha: float = 0.5,
                beta: float = 1.0, beam: int = 8, top_k: int = 8, prune: float = -8.0) -> list[int]:
    """log_probs：(T, V+1)，第 0 類係 blank。回傳字 id（對應 units index）。"""
    return beam_search_nbest(log_probs, units, lm, alpha, beta, beam, top_k, prune)[0][0]


def beam_search_nbest(log_probs: np.ndarray, units: list[str], lm: NGramLM | None = None, alpha: float = 0.5,
                      beta: float = 1.0, beam: int = 8, top_k: int = 8,
                      prune: float = -8.0) -> list[tuple[list[int], float]]:
    """同 beam_search 一樣，但回傳成個 beam：[(字 id, 總分)]，分數由高至低。"""
    # beams：prefix（tuple of ids）→ [p_blank, p_nonblank, lm 分數]
    beams = {(): [0.0, NEG_INF, 0.0]}
    for frame in log_probs:
        k = min(top_k, len(frame) - 1)
        cand = np.argpartition(-frame, k)[:k]
        cand = [int(c) for c in cand if frame[c] > prune or c == 0]
        if 0 not in cand:
            cand.append(0)
        nxt = {}

        def get(prefix, lm_score):
            if prefix not in nxt:
                nxt[prefix] = [NEG_INF, NEG_INF, lm_score]
            return nxt[prefix]

        for prefix, (pb, pnb, lm_score) in beams.items():
            total = _logsumexp(pb, pnb)
            for c in cand:
                lp = float(frame[c])
                if c == 0:
                    e = get(prefix, lm_score)
                    e[0] = _logsumexp(e[0], total + lp)
                    continue
                uid = c - 1
                if prefix and prefix[-1] == uid:
                    # 重複字：唔經 blank 就係同一個字延續；經過 blank 先係新字
                    e = get(prefix, lm_score)
                    e[1] = _logsumexp(e[1], pnb + lp)
                    new_from = pb + lp
                else:
                    new_from = total + lp
                new_prefix = prefix + (uid,)
                if new_prefix not in nxt:
                    add = lm.logp(tuple(units[i] for i in prefix), units[uid]) if lm else 0.0
                    nxt[new_prefix] = [NEG_INF, NEG_INF, lm_score + alpha * add + beta]
                e = nxt[new_prefix]
                e[1] = _logsumexp(e[1], new_from)
        beams = dict(sorted(nxt.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]) + kv[1][2],
                            reverse=True)[:beam])

    def final(kv):
        prefix, (pb, pnb, lm_score) = kv
        score = _logsumexp(pb, pnb) + lm_score
        if lm:
            score += alpha * lm.logp(tuple(units[i] for i in prefix), EOS)
        return score

    return sorted(((list(kv[0]), final(kv)) for kv in beams.items()), key=lambda x: x[1], reverse=True)

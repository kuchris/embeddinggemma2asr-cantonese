"""文字正規化同切單位：中日文逐個字，英文／數字逐個詞。"""

import re
import unicodedata

import opencc

_to_hk = opencc.OpenCC("s2hk")
# 中日文字（CJK 統一漢字、擴展 A、假名）逐個切；拉丁字母／數字連埋一個詞
UNIT = re.compile(r"[㐀-䶿一-鿿぀-ヿ]|[a-z0-9]+(?:'[a-z]+)?")
LATIN = re.compile(r"[a-z0-9]")


def normalize(text: str, to_traditional: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _to_hk.convert(text) if to_traditional else text


def units(text: str, to_traditional: bool = True) -> list[str]:
    return UNIT.findall(normalize(text, to_traditional))


def join_units(us: list[str]) -> str:
    """單位 → 句子：英文詞之間加空格，中文字之間唔加。"""
    out = ""
    for u in us:
        if out and LATIN.match(u) and LATIN.match(out[-1]):
            out += " "
        out += u
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

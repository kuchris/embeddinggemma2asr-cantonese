"""SpiCE 字幕處理：讀 TextGrid、清走轉錄標記同語氣詞、統一寫法（所有系統用同一套）。"""

import re

from .text import units

FILLERS = {"um", "uh", "mm", "hmm", "er", "ah", "eh", "oh"}
VARIANTS = {"系": "係", "姐": "即"}  # 「系／係」「姐係／即係」只係寫法唔同，唔當錯


def norm(us: list[str]) -> list[str]:
    return [VARIANTS.get(u, u) for u in us if u not in FILLERS]


def clean(text: str) -> list[str]:
    """SpiCE 字幕 → 單位：刪走轉錄標記（&xx、xx@m、【】）同英文語氣詞（um、uh…）。"""
    return norm(units(re.sub(r"&\S+|@\S+", " ", text)))


def tier(textgrid: str, name: str):
    block = [b for b in re.split(r"item \[\d+\]:", textgrid)[1:] if f'name = "{name}"' in b][0]
    return [(float(a), float(b), t.strip()) for a, b, t in
            re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', block, re.S)]

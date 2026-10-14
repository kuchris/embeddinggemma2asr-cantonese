"""粵拼標籤（輔助 CTC 用）：每個中文字 → 一個「音節 + 聲調」單位（按上下文揀讀音），英文詞 → <en>。

例：["我", "去", "hiking"] → ["ngo5", "heoi3", "<en>"]
冇讀音嘅字（罕見字、符號）直接略過。
"""

import ToJyutping

EN = "<en>"


def jyutping_units(units: list[str]) -> list[str]:
    out, run = [], []

    def flush():
        if run:
            for _, jp in ToJyutping.get_jyutping_list("".join(run)):
                if jp:
                    out.append(jp.split()[0])  # 多讀音嗰陣攞第一個
            run.clear()

    for u in units:
        if len(u) == 1 and not u.isascii():
            run.append(u)
        else:
            flush()
            out.append(EN)
    flush()
    return out

"""長音頻切段：用 silero-vad 搵有人講嘢嘅部分，再合併成最長 MAX_SEC 秒嘅段落。

reader 只 train 過乾淨、有人講嘢、20 秒以內嘅句子；直接餵成段長音頻，靜音同背景人聲會變成亂碼。
"""

import numpy as np
import torch
from silero_vad import get_speech_timestamps_from_probs, load_silero_vad

SAMPLE_RATE = 16000
MAX_SEC = 20
WINDOW = 512  # silero 每次睇 32ms
STREAM_SEC = 30  # 長音頻切成每段 30 秒並排 batch 處理（逐窗口順序做太慢）
_model = None


@torch.no_grad()
def speech_probs(wav: np.ndarray) -> list[float]:
    """每 32ms 一個「有人講嘢」機率。將音頻切成多條 30 秒嘅流一齊行（batch），快好多。"""
    global _model
    if _model is None:
        _model = load_silero_vad()
    n_win = -(-len(wav) // WINDOW)
    per_stream = STREAM_SEC * SAMPLE_RATE // WINDOW
    n_streams = -(-n_win // per_stream)
    x = np.zeros(n_streams * per_stream * WINDOW, dtype=np.float32)
    x[:len(wav)] = wav
    x = torch.from_numpy(x).view(n_streams, per_stream, WINDOW)
    _model.reset_states()
    probs = torch.stack([_model(x[:, t], SAMPLE_RATE).view(-1) for t in range(per_stream)], dim=1)
    return probs.reshape(-1)[:n_win].tolist()


def speech_segments(wav: np.ndarray, threshold: float = 0.5, max_gap: float = 0.8,
                    max_sec: float = MAX_SEC) -> list[tuple[int, int]]:
    """16kHz 音頻 → [(開始 sample, 結束 sample)]，相隔唔夠 max_gap 秒嘅段會合併（唔超過 max_sec）。"""
    stamps = get_speech_timestamps_from_probs(speech_probs(wav), sampling_rate=SAMPLE_RATE, threshold=threshold,
                                              max_speech_duration_s=max_sec, min_silence_duration_ms=300,
                                              speech_pad_ms=150, audio_length_samples=len(wav))
    merged = []
    for s in stamps:
        a, b = s["start"], s["end"]
        if merged and a - merged[-1][1] < max_gap * SAMPLE_RATE and b - merged[-1][0] <= max_sec * SAMPLE_RATE:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged

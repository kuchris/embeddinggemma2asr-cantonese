# embeddinggemma2asr-cantonese 報告：用凍結嘅 EmbeddingGemma 2 做廣東話語音辨識

> 詳細數字同每次實驗設定見 [RESULTS.md](RESULTS.md)。呢份係總結。

## 一句講晒

凍結 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 嘅 audio encoder，喺上面 train 一個
2,080 萬參數嘅 CTC reader。最新版本（learned_r6b，1,238 小時數據）喺 RTX 5070 Ti 上約 **750–850 倍即時**，
比 SenseVoice-Small 快約 3 倍；有聲書錯誤率 **4.5%**，低過 SenseVoice（7.9%）；但自然對話仍然輸（34% vs 28%）。
權重：[kuchris/embeddinggemma2asr-cantonese](https://huggingface.co/kuchris/embeddinggemma2asr-cantonese)。

## 動機

[Embedding Gemma Transcribe](https://eg2.haku.sh/) 示範咗：EmbeddingGemma 2 雖然係 embedding model，但佢個 audio encoder
夠好，加個細 reader 就可以做英文語音辨識，而且好快。我哋想知粵語做唔做到，同埋可以追到幾近 Google／SenseVoice。

## 最終架構（learned_r6／r6b）

```
16kHz 音頻
 → EmbeddingGemma 2 audio_tower（約 3 億參數，凍結，每 40ms 一格）
 → 攞 4 層：第 6 層、第 9 層、第 12 層（經 output_proj）、embed_audio 輸出
 → 每層 PCA 壓縮做 256 維（參數存喺 checkpoint）
 → softmax 權重加埋 4 層（SUPERB 式 weighted layer sum）
 → reader：6 層 Transformer（384 維）+ depthwise conv
 → CTC，12,368 個單位（中文逐字、英文逐詞）
 （train 時另有粵拼 CTC 輔助 head，推理唔用）
```

可選：CTC prefix beam search + 4-gram LM（純 Python）。r6b 要用細啲嘅 LM 權重（α=0.3 β=2.5），全部測試集好約 1 點。

## 結果

錯誤率（CER，越低越好）。greedy ／ 加 LM。SenseVoice-Small 同一張 GPU、同一批音頻。

| 測試集 | learned_r4 | learned_r6 | **learned_r6b** | SenseVoice-Small |
|---|---|---|---|---|
| 原本 test（讀稿 + 粵英夾雜，1,030 句） | 12.21% ／ 11.57% | 11.51% ／ 11.13% | 10.79% ／ **9.59%** | 9.7% |
| MDCC test（有聲書，2,000 句） | 6.6% ／ 5.7% | 5.8% ／ 4.9% | **5.3% ／ 4.5%** | 7.9% |
| SpiCE（溫哥華雙語人士自然訪談） | 38.0% ／ 38.5% | 36.3% ／ 36.8% | **34.1% ／ 33.3%** | 28.0% |
| 速度（greedy，本機 RTX 5070 Ti） | 約 845× 即時 | 約 850× 即時 | 約 750–850× 即時 | 約 280–365× |

注意：
- r6b 嘅 LM 欄用 α=0.3 β=2.5（`scripts/lm_sweep.py` 喺三個測試集各一半揀）；原本 test 改用原始音頻量（greedy 10.79%，舊數 10.59% 係 feature shard）。r4／r6 仍然係 α=0.7。
- MDCC 對我哋係同類數據（train 過 MDCC train），對 SenseVoice 係 zero-shot。
- 電台數據（r6b 有 1,037 小時）嘅標籤係 SenseVoice 出嘅，即係部分係「蒸餾」SenseVoice，對話類好難超越佢。

## 每一輪做咗咩、學到咩

| 輪 | 改動 | 數據 | 重點結果 |
|---|---|---|---|
| 1–2 | frozen／tuned／learned 三種分類頭；4-gram LM | 143h（TTS 風格） | learned 準少少、學得快；LM 減少約 18% 錯字；大 reader 冇用 |
| 3 | 加真人讀稿 MDCC | 約 200h | MDCC 15.8% → 5.6%；SpiCE 47.8% |
| 4 | 電台錄音 + 兩個老師（Whisper、SenseVoice）互相驗證做標籤 | 720h | SpiCE 47.8% → 38.0%（最大改善） |
| — | LLM 改字（Qwen3.5、Gemma 4）、Gemma 4 直接轉錄、N-best rescoring | | 全部唔值得（變差或者只好 0.5 點，慢好多） |
| 5 | 粵拼輔助 CTC；細規模試中間層 | 720h | 粵拼單獨只好 0–0.3 點；多層相對好約 9%，第 6 層權重最高 |
| 6 | 4 層 + PCA + fp8 + 粵拼 | 720h | 全部測試集再好 0.2–0.8 點；MDCC +LM 首次低過 5% |
| 6b | 同上，加 518 小時電台 | 1,238h | 驗證 7.9% → 6.4%；對話 36.3% → 34.1%；LM 要重新調 α（0.7 → 0.3）先有用 |

### 有用
- **數據**：真人、自然對話數據係最大推動力。143h → 720h 令對話錯誤率跌 10 點。
- **encoder 中間層**：一直用嘅最後輸出係權重最低嗰層；第 6 層最高。只用第 6 層喺對話反而差，要幾層一齊用。
- **PCA 256 + fp8 壓縮**：4 層 feature 由 4,096 維 fp16 壓到 1,024 維 fp8（細 8 倍），準確度冇跌。
- **預先載入 RAM**：train 快約 1 倍（720h 由 1 小時 40 分降到約 55 分鐘）。

### 冇用（或者唔抵）
- 固定用 EmbeddingGemma 文字空間做分類頭（frozen head）：比普通分類頭差少少，只係對冇 train 過嘅語言好少少。
- 加大 reader（數據唔夠時）。
- 冇 fine-tune 嘅 LLM 改字：會刪字、改人名、譯英文，全部變差。
- Gemma 4（E2B／E4B）直接轉錄：本身唔識粵語（42–84% CER）。
- N-best rescoring（Qwen3.5-0.8B）：好 0.4–0.6 點，每句多 45ms。

## 成本

全部用 Colab（Pro）A100，約 5.3 compute units／小時。第四輪約 22 units；第五輪約 22 units（包括一次 VM 斷線重做）；
第六輪約 51 units（包括多抽 518 小時，同埋 r6b 用 mmap 慢咗）。

踩過嘅坑：Colab VM 冇 kernel 活動會被收機（要定時 ping）；數據大過 RAM 時 Colab 磁碟太慢（每秒約 150–200MB），
mmap train 慢 3 倍。

## 下一步

1. **3,000 小時**：電台數據集仲有大約 6,000 小時未用；要 RAM 大（> 200GB）或者磁碟快嘅機器。
2. 更準嘅老師（標籤品質決定上限）。
3. 數據多咗之後再試大啲嘅 reader；同埋換更大／更好嘅 LM（4-gram 重新調 α 之後仍然有用；n-best 上限仲有 2–5 點空間）。

## 授權

EmbeddingGemma 2 係 Apache-2.0；但部分 train 數據係 CC BY-NC-ND 4.0 或者冇註明授權（包括電台錄音），
所以 model 權重只建議作研究同非商業用途。

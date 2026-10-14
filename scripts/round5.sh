#!/usr/bin/env bash
# 第五輪（Colab A100）：(1) 粵拼輔助 CTC (2) 用 audio encoder 中間層（weighted layer sum）
#
# 細規模對照：MDCC train 30 小時，抽第 3／6／7／8／9／10／12 層 + 原本 512 維，同一批數據 train 4 個 reader：
#   p_final（原本）、p_mix（混合各層）、p_final_jy（+ 粵拼）、p_mix_jy（兩樣都有）
# 全數據：r4 嘅 720 小時 feature（由 Drive 抄返嚟）+ 粵拼，同 learned_r4 直接比較。
#
# 先要有：data/vocab_r4.pt、data/lm_r4.pkl、data/spice/prepared；全數據要 data/feats/r4.copied（抄完 Drive 嘅標記）
# 用法：bash scripts/round5.sh   （log 喺 runs/round5/）
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/round5
mkdir -p "$LOG"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }
PROBE=data/feats/mdcc_layers
LAYERS=3,6,7,8,9,10,12,0
JY=${JY:-0.3}

if [ ! -f "$PROBE/done" ]; then
  step "抽 MDCC 30 小時（第 $LAYERS 層）"
  python scripts/extract.py --dataset mdcc --out "$PROBE" --max-hours 30 --shard-size 2000 --layers "$LAYERS" \
    > "$LOG/extract_probe.log" 2>&1 && touch "$PROBE/done" || { step "抽 feature 失敗"; exit 1; }
fi
step "MDCC 對照 feature：$(du -sh "$PROBE" | cut -f1)"

probe() {
  local name=$1; shift
  [ -f "runs/$name/best.pt" ] && [ -f "$LOG/$name.done" ] && return
  step "train $name（$*）"
  python scripts/train.py "$PROBE" --vocab data/vocab_r4.pt --out "runs/$name" --head learned \
    --epochs 30 --max-frames 20000 --eval-every 500 "$@" > "$LOG/$name.log" 2>&1 \
    && touch "$LOG/$name.done" || { step "$name 失敗"; tail -20 "$LOG/$name.log" >> "$LOG/pipeline.log"; }
  step "$name：$(grep 完成 "$LOG/$name.log")"
}
probe p_final --use-layers 0
probe p_mix --use-layers "$LAYERS"
probe p_final_jy --use-layers 0 --jy-weight "$JY"
probe p_mix_jy --use-layers "$LAYERS" --jy-weight "$JY"

PROBES="runs/p_final/best.pt runs/p_mix/best.pt runs/p_final_jy/best.pt runs/p_mix_jy/best.pt"
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  --lm data/lm_r4.pkl --show 0 $PROBES > "$LOG/probe_mdcc.log" 2>&1
step "對照 MDCC test：$(grep -E '錯誤率' "$LOG/probe_mdcc.log" | tr '\n' ' ')"
python scripts/spice_eval.py $PROBES --lm data/lm_r4.pkl > "$LOG/probe_spice.log" 2>&1
step "對照 SpiCE：$(grep -E '^==|合計' "$LOG/probe_spice.log" | tr '\n' ' ')"

# ---------- 全數據 + 粵拼 ----------
while [ ! -f data/feats/r4.copied ]; do sleep 60; done
FEATS="data/feats/daily data/feats/mixed data/feats/wordshk data/feats/mdcc $(ls -d data/feats/radio_* | xargs)"
OLD="data/feats/daily data/feats/mixed data/feats/wordshk"
if [ ! -f "$LOG/learned_r5jy.done" ]; then
  step "train learned_r5jy（720 小時 + 粵拼 $JY）"
  python scripts/train.py $FEATS --vocab data/vocab_r4.pt --out runs/learned_r5jy --head learned \
    --epochs ${EPOCHS:-12} --max-frames ${MAX_FRAMES:-60000} --eval-every 1000 --jy-weight "$JY" \
    > "$LOG/train_r5jy.log" 2>&1 && touch "$LOG/learned_r5jy.done" \
    || { step "train 失敗"; tail -20 "$LOG/train_r5jy.log" >> "$LOG/pipeline.log"; exit 1; }
fi
step "train 完成：$(grep 完成 "$LOG/train_r5jy.log")"

python scripts/evaluate.py runs/learned_r5jy/best.pt $OLD --lm data/lm_r4.pkl --part test --alphas 0.7 --betas 4.0 \
  > "$LOG/eval_old_test.log" 2>&1
step "原本 test：$(grep -E '^greedy|^beam \+' "$LOG/eval_old_test.log" | tr '\n' ' ')"
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  --lm data/lm_r4.pkl --show 5 runs/learned_r5jy/best.pt > "$LOG/eval_mdcc.log" 2>&1
step "MDCC test：$(grep 錯誤率 "$LOG/eval_mdcc.log")"
python scripts/spice_eval.py runs/learned_r5jy/best.pt --lm data/lm_r4.pkl > "$LOG/eval_spice.log" 2>&1
step "SpiCE：$(grep -E '合計' "$LOG/eval_spice.log" | tr '\n' ' ')"
step "全部完成"

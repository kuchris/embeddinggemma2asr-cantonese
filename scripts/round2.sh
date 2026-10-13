#!/usr/bin/env bash
# 第二輪：等 feature 抽完 → 字典 → LM → train frozen（有遮罩）→ 評估 → train learned 對照組 → 評估
# 用法：bash scripts/round2.sh   （log 喺 runs/round2/*.log）
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
FEATS="data/feats/daily data/feats/mixed data/feats/wordshk"
LOG=runs/round2
mkdir -p "$LOG"

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }

step "等 feature 抽完"
# 等 wordshk 抽完：extract 完成之後由外面 touch data/feats/wordshk/DONE
until [ -f data/feats/wordshk/DONE ]; do sleep 30; done
step "feature 完成：$(du -sh data/feats | cut -f1)"

step "整字典"
uv run python scripts/build_vocab.py $FEATS --out data/vocab.pt --min-count 2 > "$LOG/vocab.log" 2>&1 || { step "字典失敗"; exit 1; }
step "$(grep 字典 "$LOG/vocab.log")"

step "整 LM"
uv run python scripts/build_lm.py $FEATS --out data/lm.pkl --order 4 > "$LOG/lm.log" 2>&1 || { step "LM 失敗"; exit 1; }
step "$(grep 已存 "$LOG/lm.log")"

for head in frozen learned; do
  step "train $head"
  uv run python scripts/train.py $FEATS --vocab data/vocab.pt --out runs/$head --head $head --epochs 30 \
    > "$LOG/train_$head.log" 2>&1 || { step "train $head 失敗"; tail -20 "$LOG/train_$head.log" >> "$LOG/pipeline.log"; continue; }
  step "train $head 完成：$(grep 完成 "$LOG/train_$head.log")"

  step "評估 $head"
  uv run python scripts/evaluate.py runs/$head/best.pt $FEATS --lm data/lm.pkl --max-utts 1500 \
    > "$LOG/eval_$head.log" 2>&1 || step "評估 $head 失敗"
  step "$(grep -E '^greedy|最好' "$LOG/eval_$head.log" | tr '\n' ' ')"

  uv run python scripts/transcribe.py runs/$head/best.pt samples/*.wav > "$LOG/samples_${head}_greedy.log" 2>&1
  uv run python scripts/transcribe.py runs/$head/best.pt samples/*.wav --lm data/lm.pkl > "$LOG/samples_${head}_lm.log" 2>&1
  step "samples $head 完成"
done
step "全部完成"

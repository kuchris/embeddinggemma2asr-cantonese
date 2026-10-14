#!/usr/bin/env bash
# 第三輪：加 MDCC 真人數據 → 字典 → LM → train learned → 評估（原本 test 句子 + MDCC test 2000 句）
# 用法：bash scripts/round3.sh   （等 data/feats/mdcc_test/DONE 出現先開始；log 喺 runs/round3/）
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
FEATS="data/feats/daily data/feats/mixed data/feats/wordshk data/feats/mdcc"
OLD="data/feats/daily data/feats/mixed data/feats/wordshk"
LOG=runs/round3
mkdir -p "$LOG"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }

step "等 MDCC feature 抽完"
until [ -f data/feats/mdcc_test/DONE ]; do sleep 30; done
step "feature 完成：$(du -sh data/feats/mdcc | cut -f1)"

uv run python scripts/build_vocab.py $FEATS --out data/vocab_r3.pt --min-count 2 > "$LOG/vocab.log" 2>&1 || { step "字典失敗"; exit 1; }
step "$(grep 字典 "$LOG/vocab.log")"
uv run python scripts/build_lm.py $FEATS --out data/lm_r3.pkl --order 4 > "$LOG/lm.log" 2>&1 || { step "LM 失敗"; exit 1; }
step "$(grep 已存 "$LOG/lm.log")"

step "train learned_r3"
uv run python scripts/train.py $FEATS --vocab data/vocab_r3.pt --out runs/learned_r3 --head learned --epochs 30 \
  > "$LOG/train.log" 2>&1 || { step "train 失敗"; tail -20 "$LOG/train.log" >> "$LOG/pipeline.log"; exit 1; }
step "train 完成：$(grep 完成 "$LOG/train.log")"

uv run python scripts/evaluate.py runs/learned_r3/best.pt $OLD --lm data/lm_r3.pkl --part test --alphas 0.7 --betas 4.0 \
  > "$LOG/eval_old_test.log" 2>&1
step "原本 test：$(grep -E '^greedy|^beam \+' "$LOG/eval_old_test.log" | tr '\n' ' ')"
uv run python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  --lm data/lm_r3.pkl --show 5 runs/learned_r3/best.pt > "$LOG/eval_mdcc.log" 2>&1
step "MDCC test：$(grep 錯誤率 "$LOG/eval_mdcc.log")"
step "全部完成"

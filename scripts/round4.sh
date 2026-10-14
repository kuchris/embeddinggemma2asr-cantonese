#!/usr/bin/env bash
# 第四輪（Colab A100）：老師標籤 radio + 原本 4 個數據集 → 字典 → LM → train learned → 評估
# 先要行完 extract_radio.py 同 4 個 extract.py（data/feats/{radio,daily,mixed,wordshk,mdcc}）
# 用法：bash scripts/round4.sh   （log 喺 runs/round4/）
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
FEATS="data/feats/daily data/feats/mixed data/feats/wordshk data/feats/mdcc $(ls -d data/feats/radio_* | xargs)"
OLD="data/feats/daily data/feats/mixed data/feats/wordshk"
LOG=runs/round4
mkdir -p "$LOG"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }

# SpiCE 測試數據（CC BY 4.0）直接由 Borealis 落載
mkdir -p data/spice
for p in "165805 VF19A_Cantonese_I2_20181114.TextGrid" "165798 VF19A_Cantonese_I2_20181114.wav" \
         "165810 VM19A_Cantonese_I2_20191031.TextGrid" "165808 VM19A_Cantonese_I2_20191031.wav"; do
  set -- $p; [ -f "data/spice/$2" ] || curl -sL -o "data/spice/$2" "https://borealisdata.ca/api/access/datafile/$1"
done
python scripts/spice_eval.py --prepare > "$LOG/spice_prepare.log" 2>&1
step "SpiCE 準備好"

python scripts/build_vocab.py $FEATS --out data/vocab_r4.pt --min-count 2 > "$LOG/vocab.log" 2>&1 || { step "字典失敗"; exit 1; }
step "$(grep 字典 "$LOG/vocab.log")"
python scripts/build_lm.py $FEATS --out data/lm_r4.pkl --order 4 > "$LOG/lm.log" 2>&1 || { step "LM 失敗"; exit 1; }
step "$(grep 已存 "$LOG/lm.log")"

step "train learned_r4"
python scripts/train.py $FEATS --vocab data/vocab_r4.pt --out runs/learned_r4 --head learned \
  --epochs ${EPOCHS:-12} --max-frames ${MAX_FRAMES:-120000} --eval-every 1000 \
  > "$LOG/train.log" 2>&1 || { step "train 失敗"; tail -20 "$LOG/train.log" >> "$LOG/pipeline.log"; exit 1; }
step "train 完成：$(grep 完成 "$LOG/train.log")"

python scripts/evaluate.py runs/learned_r4/best.pt $OLD --lm data/lm_r4.pkl --part test --alphas 0.7 --betas 4.0 \
  > "$LOG/eval_old_test.log" 2>&1
step "原本 test：$(grep -E '^greedy|^beam \+' "$LOG/eval_old_test.log" | tr '\n' ' ')"
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  --lm data/lm_r4.pkl --show 5 runs/learned_r4/best.pt > "$LOG/eval_mdcc.log" 2>&1
step "MDCC test：$(grep 錯誤率 "$LOG/eval_mdcc.log")"
python scripts/spice_eval.py runs/learned_r4/best.pt --lm data/lm_r4.pkl > "$LOG/eval_spice.log" 2>&1
step "SpiCE：$(grep -E '合計' "$LOG/eval_spice.log" | tr '\n' ' ')"
step "全部完成"

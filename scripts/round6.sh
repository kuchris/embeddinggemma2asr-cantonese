#!/usr/bin/env bash
# 第六輪（Colab A100）：4 層 encoder feature（第 6／9／12 層 + 原本），PCA 每層 256 維 + fp8，全數據 + 粵拼。
#
# A. 細規模：MDCC 30 小時抽原始 4 層 → 計 PCA → 再抽壓縮版，同一批數據比較 p_raw4 vs p_pca4（壓縮有冇蝕）
# B. 壓縮冇蝕（MDCC +LM 差 ≤ 0.3 點）先做全數據：5 個數據集用 PCA + fp8 重抽（同第四輪一樣嘅數據），
#    train learned_r6（12 epoch、粵拼 0.3），評估同第四／五輪一樣。
#
# 先要有：data/vocab_r4.pt、data/lm_r4.pkl、data/spice/prepared；radio 係 gated，要先 `hf auth login`
# 用法：bash scripts/round6.sh   （log 喺 runs/round6/）；STOP_AFTER_PROBE=1 只做 A
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/round6
mkdir -p "$LOG"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }
LAYERS=6,9,12,0
PCA=data/pca_6_9_12_0.pt
JY=${JY:-0.3}
VOCAB=${VOCAB:-data/vocab_r4.pt}
if [ ! -f "$VOCAB" ]; then  # 冇 Drive 嗰陣：用 MDCC 30 小時自己建一個（只夠細規模對照用）
  VOCAB=data/vocab_mdcc30.pt; NEED_VOCAB=1
fi
LMARG=""; [ -f data/lm_r4.pkl ] && LMARG="--lm data/lm_r4.pkl"

# ---------- A. 細規模：壓縮有冇蝕 ----------
if [ ! -f data/feats/mdcc_l4/done ]; then
  step "抽 MDCC 30 小時原始 4 層"
  python scripts/extract.py --dataset mdcc --out data/feats/mdcc_l4 --max-hours 30 --shard-size 2000 --layers $LAYERS \
    > "$LOG/extract_l4.log" 2>&1 && touch data/feats/mdcc_l4/done || { step "抽 feature 失敗"; exit 1; }
fi
[ -f $PCA ] || python scripts/fit_pca.py data/feats/mdcc_l4 --layers $LAYERS --k 256 --out $PCA > "$LOG/pca.log" 2>&1 \
  || { step "PCA 失敗"; exit 1; }
step "PCA：$(grep -E '保留' "$LOG/pca.log" | tr '\n' ' ')"
if [ ! -f data/feats/mdcc_pca/done ]; then
  python scripts/extract.py --dataset mdcc --out data/feats/mdcc_pca --max-hours 30 --shard-size 2000 --layers $LAYERS \
    --pca $PCA --fp8 > "$LOG/extract_pca.log" 2>&1 && touch data/feats/mdcc_pca/done || { step "抽壓縮 feature 失敗"; exit 1; }
fi
[ "${NEED_VOCAB:-0}" = 1 ] && [ ! -f "$VOCAB" ] && python scripts/build_vocab.py data/feats/mdcc_l4 --out "$VOCAB" --min-count 1 > "$LOG/vocab.log" 2>&1
step "MDCC 30 小時：原始 $(du -sh data/feats/mdcc_l4 | cut -f1)，壓縮 $(du -sh data/feats/mdcc_pca | cut -f1)"

probe() {
  local name=$1 dir=$2
  [ -f "$LOG/$name.done" ] && return
  python scripts/train.py "$dir" --vocab "$VOCAB" --out "runs/$name" --head learned \
    --epochs 30 --max-frames 20000 --eval-every 500 > "$LOG/$name.log" 2>&1 && touch "$LOG/$name.done" \
    || { step "$name 失敗"; tail -20 "$LOG/$name.log" >> "$LOG/pipeline.log"; exit 1; }
  step "$name：$(grep 完成 "$LOG/$name.log")"
}
probe p_raw4 data/feats/mdcc_l4
probe p_pca4 data/feats/mdcc_pca
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  $LMARG --show 0 runs/p_raw4/best.pt runs/p_pca4/best.pt > "$LOG/probe_mdcc.log" 2>&1
step "對照 MDCC test：$(grep 錯誤率 "$LOG/probe_mdcc.log" | sed 's/（head[^）]*）//' | tr '\n' ' ')"
python scripts/spice_eval.py runs/p_raw4/best.pt runs/p_pca4/best.pt $LMARG > "$LOG/probe_spice.log" 2>&1
step "對照 SpiCE：$(grep -E '^==|合計 vad' "$LOG/probe_spice.log" | tr '\n' ' ')"

KEY='+LM'; [ -z "$LMARG" ] && KEY='錯誤率'  # 冇 LM 就比 greedy
LM_RAW=$(grep -o 'p_raw4/best.pt.*' "$LOG/probe_mdcc.log" | grep -o "$KEY [0-9.]*" | grep -o '[0-9.]*$')
LM_PCA=$(grep -o 'p_pca4/best.pt.*' "$LOG/probe_mdcc.log" | grep -o "$KEY [0-9.]*" | grep -o '[0-9.]*$')
if ! python -c "import sys; sys.exit(0 if float('$LM_PCA') - float('$LM_RAW') <= 0.3 else 1)"; then
  step "壓縮蝕咗（MDCC +LM 原始 $LM_RAW% vs 壓縮 $LM_PCA%），唔做全數據"; exit 0
fi
step "壓縮 OK（MDCC +LM 原始 $LM_RAW% vs 壓縮 $LM_PCA%）"
[ "${STOP_AFTER_PROBE:-0}" = 1 ] && { step "STOP_AFTER_PROBE，停喺度"; exit 0; }

# ---------- B. 全數據重抽（同第四輪一樣嘅數據） ----------
for d in daily mixed wordshk mdcc; do
  [ -f data/feats6/$d/done ] && continue
  step "抽 $d"
  python scripts/extract.py --dataset $d --out data/feats6/$d --layers $LAYERS --pca $PCA --fp8 \
    > "$LOG/extract_$d.log" 2>&1 && touch data/feats6/$d/done || { step "$d 失敗"; exit 1; }
done &
OLD=$!
for k in 0 1 2; do
  [ -f data/feats6/radio_$k/done ] && continue
  python scripts/extract_radio.py --out data/feats6/radio_$k --offset $((k * 4)) --stride 12 --agree 0.25 --max-hours 170 \
    --layers $LAYERS --pca $PCA --fp8 > "$LOG/extract_radio_$k.log" 2>&1 && touch data/feats6/radio_$k/done &
done
wait
for d in daily mixed wordshk mdcc radio_0 radio_1 radio_2; do
  [ -f data/feats6/$d/done ] || { step "抽 $d 未完成"; exit 1; }
done
step "全數據 feature：$(du -sh data/feats6 | cut -f1)"

FEATS="data/feats6/daily data/feats6/mixed data/feats6/wordshk data/feats6/mdcc data/feats6/radio_0 data/feats6/radio_1 data/feats6/radio_2"
OLDT="data/feats6/daily data/feats6/mixed data/feats6/wordshk"
if [ ! -f "$LOG/learned_r6.done" ]; then
  step "train learned_r6（4 層 PCA + 粵拼 $JY）"
  python scripts/train.py $FEATS --vocab data/vocab_r4.pt --out runs/learned_r6 --head learned \
    --epochs ${EPOCHS:-12} --max-frames ${MAX_FRAMES:-60000} --eval-every 1000 --jy-weight "$JY" \
    > "$LOG/train_r6.log" 2>&1 && touch "$LOG/learned_r6.done" \
    || { step "train 失敗"; tail -20 "$LOG/train_r6.log" >> "$LOG/pipeline.log"; exit 1; }
fi
step "train 完成：$(grep 完成 "$LOG/train_r6.log")"
python scripts/evaluate.py runs/learned_r6/best.pt $OLDT --lm data/lm_r4.pkl --part test --alphas 0.7 --betas 4.0 \
  > "$LOG/eval_old_test.log" 2>&1
step "原本 test：$(grep -E '^greedy|^beam \+' "$LOG/eval_old_test.log" | tr '\n' ' ')"
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 \
  --lm data/lm_r4.pkl --show 5 runs/learned_r6/best.pt > "$LOG/eval_mdcc.log" 2>&1
step "MDCC test：$(grep 錯誤率 "$LOG/eval_mdcc.log")"
python scripts/spice_eval.py runs/learned_r6/best.pt --lm data/lm_r4.pkl > "$LOG/eval_spice.log" 2>&1
step "SpiCE：$(grep -E '合計' "$LOG/eval_spice.log" | tr '\n' ' ')"
step "全部完成"

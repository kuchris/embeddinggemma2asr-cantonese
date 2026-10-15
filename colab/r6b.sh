cd /content/gemma2asr && tar -xzf /content/patch6b.tar.gz
export PYTHONIOENCODING=utf-8 G2ASR_MODELS=/content/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/round6; F=data/feats6; DR=/content/drive/MyDrive/gemma2asr
s() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L/pipeline.log; }
rm -rf runs/learned_r6b
FEATS="$F/daily $F/mixed $F/wordshk $F/mdcc $F/radio_0 $F/radio_1 $F/radio_2 $F/radio_3 $F/radio_4 $F/radio_5"
s "重新 train learned_r6b（mmap 分組讀 + 預讀下一組）"
python scripts/train.py $FEATS --vocab data/vocab_r4.pt --out runs/learned_r6b --head learned --mmap \
  --epochs 12 --max-frames 60000 --eval-every 1000 --jy-weight 0.3 > $L/train_r6b.log 2>&1 \
  || { s "r6b train 失敗"; tail -20 $L/train_r6b.log >> $L/pipeline.log; }
s "r6b train 完成：$(grep 完成 $L/train_r6b.log)"
OLDT="$F/daily $F/mixed $F/wordshk"
python scripts/evaluate.py runs/learned_r6b/best.pt $OLDT --lm data/lm_r4.pkl --part test --alphas 0.7 --betas 4.0 > $L/eval_r6b_old.log 2>&1
s "r6b 原本 test：$(grep -E '^greedy|^beam \+' $L/eval_r6b_old.log | tr '\n' ' ')"
python scripts/eval_external.py --repo ming030890/mdcc --split test --text-col transcript --n 2000 --lm data/lm_r4.pkl --show 0 runs/learned_r6b/best.pt > $L/eval_r6b_mdcc.log 2>&1
s "r6b MDCC test：$(grep 錯誤率 $L/eval_r6b_mdcc.log)"
python scripts/spice_eval.py runs/learned_r6b/best.pt --lm data/lm_r4.pkl > $L/eval_r6b_spice.log 2>&1
s "r6b SpiCE：$(grep -E '合計' $L/eval_r6b_spice.log | tr '\n' ' ')"
mkdir -p $DR/round6/learned_r6b $DR/round6/learned_r6; cp runs/learned_r6b/best.pt runs/learned_r6b/log.jsonl $DR/round6/learned_r6b/
cp -u runs/learned_r6/best.pt runs/learned_r6/log.jsonl $DR/round6/learned_r6/; mkdir -p $DR/round6/logs; cp $L/* $DR/round6/logs/
s "R6B_DONE"

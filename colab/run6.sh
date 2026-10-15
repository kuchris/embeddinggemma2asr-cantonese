mkdir -p /content/gemma2asr /content/logs && tar -xzf /content/code.tar.gz -C /content/gemma2asr && mkdir -p /content/gemma2asr/data/feats
pip install -q -U 'transformers>=5.19.0' opencc-python-reimplemented rapidfuzz silero-vad soundfile librosa pyarrow 'huggingface_hub>=1.0' 'ToJyutping>=3.2' 2>&1 | tail -1
cd /content/gemma2asr && export PYTHONIOENCODING=utf-8 G2ASR_MODELS=/content/models
mkdir -p data/spice runs/round6
for p in "165805 VF19A_Cantonese_I2_20181114.TextGrid" "165798 VF19A_Cantonese_I2_20181114.wav" "165810 VM19A_Cantonese_I2_20191031.TextGrid" "165808 VM19A_Cantonese_I2_20191031.wav"; do set -- $p; [ -f "data/spice/$2" ] || curl -sL -o "data/spice/$2" "https://borealisdata.ca/api/access/datafile/$1"; done
python scripts/spice_eval.py --prepare 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] setup 完成，等 Drive（vocab_r4／lm_r4）" >> runs/round6/pipeline.log
R=/content/drive/MyDrive/gemma2asr/round4
until [ -f $R/vocab_r4.pt ]; do sleep 20; done
cp $R/vocab_r4.pt $R/lm_r4.pkl data/
echo "[$(date +%H:%M:%S)] 攞到 vocab_r4／lm_r4" >> runs/round6/pipeline.log
D=/content/drive/MyDrive/gemma2asr/round6; mkdir -p $D
sync_drive() {
  mkdir -p $D/logs; cp runs/round6/* $D/logs/ 2>/dev/null; cp -u data/pca_6_9_12_0.pt $D/ 2>/dev/null
  for r in p_raw4 p_pca4 learned_r6; do [ -f runs/$r/best.pt ] && { mkdir -p $D/$r; cp -u runs/$r/best.pt runs/$r/log.jsonl $D/$r/; }; done
}
( while true; do sleep 600; sync_drive; done ) &
SYNC=$!
bash scripts/round6.sh > /content/logs/round6.log 2>&1
kill $SYNC; sync_drive
echo "[$(date +%H:%M:%S)] ROUND6_ALL_DONE" >> runs/round6/pipeline.log

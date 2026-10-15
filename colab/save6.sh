# 每個 feats6 數據集抽完（有 done）就抄去 Drive；PCA 檔亦抄埋
F=/content/gemma2asr/data/feats6; D=/content/drive/MyDrive/gemma2asr/feats_r6; mkdir -p $D
cp -u /content/gemma2asr/data/pca_6_9_12_0.pt $D/ && echo "pca copied $(date +%H:%M)"
left="daily mixed wordshk mdcc radio_0 radio_1 radio_2"
while [ -n "$left" ]; do
  next=""
  for d in $left; do
    if [ -f $F/$d/done ]; then nice -n 19 ionice -c3 cp -r $F/$d $D/ && echo "$d copied $(date +%H:%M) $(du -sh $D/$d | cut -f1)"
    else next="$next $d"; fi
  done
  left=$(echo $next); [ -n "$left" ] && sleep 60
done
echo "SAVE6_DONE $(du -sh $D | cut -f1)"

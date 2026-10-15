# 用法（喺 WSL root）：bash colab/up.sh <檔案>...  —— 將 colab/ 入面嘅檔案上載去 VM 嘅 /content/
D="$(cd "$(dirname "$0")" && pwd)"
for f in "$@"; do timeout --foreground 300 /root/.local/bin/colab upload -s g2asr "$D/$f" "/content/$f" < /dev/null 2>&1 | tail -1 | sed 's/.*to /uploaded to /'; done

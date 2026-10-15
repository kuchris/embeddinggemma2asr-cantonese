# 用法：bash vm.sh "<shell 指令>"  —— 經 console 喺 VM 行，輸出寫去 /tmp/vm_out.txt 再下載返嚟
CMD="$1"; WAIT=${2:-10}
(sleep 4; printf '%s > /tmp/vm_out.txt 2>&1\n' "$CMD"; sleep "$WAIT"; printf 'exit\n') | timeout --foreground $((WAIT + 40)) /root/.local/bin/colab console -s g2asr > /dev/null 2>&1
rm -f /tmp/vm_out.txt; timeout --foreground 60 /root/.local/bin/colab download -s g2asr tmp/vm_out.txt /tmp/vm_out.txt < /dev/null > /dev/null 2>&1 || echo "(download failed)"
cut -c1-250 /tmp/vm_out.txt 2>/dev/null

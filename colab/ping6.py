import subprocess
print(subprocess.run(["bash", "-c", "cat /content/gemma2asr/runs/round6/pipeline.log 2>/dev/null; grep -h -E 'Traceback|Error:|GatedRepoError|Killed' /content/gemma2asr/runs/round6/*.log 2>/dev/null | tail -2 | sed 's/^/ERR /'"], capture_output=True, text=True).stdout)

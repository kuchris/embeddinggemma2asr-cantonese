import subprocess
from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True, timeout_ms=180000)
except Exception as e:
    print("ERR", repr(e)[:300])
print(subprocess.run(["bash", "-c", "ls /content/drive/MyDrive/gemma2asr 2>&1 | head"], capture_output=True, text=True).stdout)

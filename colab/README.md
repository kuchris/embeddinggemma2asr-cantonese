# Colab helpers

Used with [google-colab-cli](https://github.com/googlecolab/google-colab-cli) installed in WSL (Ubuntu, as root, at
`/root/.local/bin/colab`). Run them from PowerShell as `wsl -d Ubuntu -u root -- bash colab/<script>` (with the repo
path under `/mnt/c/...`).

| File | What it does |
|---|---|
| `vm.sh "<cmd>" [wait]` | Run a shell command on the VM through `colab console` and print its output. Does not use the Jupyter kernel, so it works while the kernel is busy. |
| `up.sh <files>` | Upload files from this folder to `/content/` on the VM. |
| `ping6.py` | `colab exec -f colab/ping6.py`: print the round-6 pipeline log. **Run it every few minutes**: Colab stops a VM whose kernel is idle, even if background jobs are running. |
| `enter.sh` | After you approve Drive access in the browser, send Enter to the waiting `colab drivemount` (it reads the TTY, not stdin). |
| `remount.py` | `colab exec -f colab/remount.py`: `drive.mount(force_remount=True)`. The first mount after approval always fails; this one works. |
| `run6.sh`, `save6.sh`, `r6b.sh` | Round-6 pipeline wrappers (setup, copy features to Drive, train learned_r6b with `--mmap`). |

Notes:
- `timeout` must be `timeout --foreground`, otherwise `colab drivemount` is stopped by SIGTTIN.
- Write long jobs as `nohup setsid bash … &` on the VM and poll their logs; `colab exec` has a reply timeout.
- Write `.sh` files with LF line endings (`.gitattributes` enforces this on commit).

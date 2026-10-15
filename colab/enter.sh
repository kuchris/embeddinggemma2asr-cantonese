P=$(pgrep -f '^/root/.local/share/uv/tools/google-colab-cli/bin/python /root/.local/bin/colab drivemount' | head -1)
T=$(ps -o tty= -p "$P" | tr -d ' ')
echo "pid=$P tty=$T"
python3 -c "import fcntl,termios,os,sys; fd=os.open('/dev/'+sys.argv[1],os.O_RDWR); fcntl.ioctl(fd,termios.TIOCSTI,b'\n'); print('sent')" "$T"

#!/usr/bin/env bash
# airsim_headless.sh: run AirSim's Blocks environment (release v1.8.1, Linux) headless, on a machine without a GPU or
# a display, such as a cloud VM. Run it as root from the repository folder.
#
#   bash airsim_headless.sh setup [--set NAME=VALUE ...]   create the user airsim (Unreal refuses to run as root),
#                                                           download Blocks.zip into its home and unpack it, and write
#                                                           its settings.json with `drone_jcc.py settings --write`
#   bash airsim_headless.sh start                           start Blocks with -nullrhi in the tmux session "airsim" and
#                                                           wait for the RPC port; the log is /home/airsim/blocks.log
#   bash airsim_headless.sh stop                            stop it
#   bash airsim_headless.sh status                          is it running, and is the RPC port open
#
# After `setup` with new --set values (for example CLOCK_SPEED), run `stop` and `start` so that Blocks reads them.
# drone_jcc.py itself runs as root as usual and connects to 127.0.0.1:41451.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
U=airsim
UH=/home/$U
URL=https://github.com/microsoft/AirSim/releases/download/v1.8.1/Blocks.zip
SHA256=563e1da1b5d7303b9405c8736243ea92ab0a011b96a561f26f17563d105a404f
ROOT=$UH/LinuxBlocks1.8.1/LinuxNoEditor
LOG=$UH/blocks.log
PORT=41451

running() { pgrep -u "$U" -f "Binaries/Linux/Blocks" >/dev/null 2>&1; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") >/dev/null 2>&1; }

case "${1:-}" in
  setup)
    shift
    id "$U" >/dev/null 2>&1 || useradd -m -s /bin/bash "$U"
    if [ ! -f "$ROOT/Blocks.sh" ]; then
      mkdir -p "$UH/dl"
      echo "downloading $URL"
      curl -sSL --fail -o "$UH/dl/Blocks.zip" "$URL"
      echo "$SHA256  $UH/dl/Blocks.zip" | sha256sum -c -
      unzip -q -o "$UH/dl/Blocks.zip" -d "$UH"
    fi
    chmod +x "$ROOT/Blocks.sh" "$ROOT/Blocks/Binaries/Linux/Blocks"  # unzip drops the execute bits
    python "$HERE/drone_jcc.py" settings --write "$UH/Documents/AirSim/settings.json" "$@"
    chown -R "$U:$U" "$UH"
    ;;
  start)
    if running; then echo "Blocks is already running"; exit 0; fi
    [ -f "$ROOT/Blocks.sh" ] || { echo "run 'bash $0 setup' first"; exit 1; }
    tmux new-session -d -s airsim "sudo -u $U -H $ROOT/Blocks.sh -nullrhi -nosound -unattended -nosplash -stdout \
-FullStdOutLogOutput > $LOG 2>&1; echo \"Blocks exited with code \$?\" >> $LOG"
    for _ in $(seq 1 60); do
      if port_open; then echo "Blocks is up: RPC port $PORT open; log $LOG"; exit 0; fi
      if ! running && grep -q "Blocks exited" "$LOG" 2>/dev/null; then break; fi
      sleep 1
    done
    echo "Blocks did not open port $PORT; the end of $LOG:"; tail -20 "$LOG"; exit 1
    ;;
  stop)
    tmux kill-session -t airsim 2>/dev/null || true
    pkill -u "$U" -f "Binaries/Linux/Blocks" 2>/dev/null || true
    for _ in $(seq 1 20); do running || break; sleep 0.5; done
    if running; then echo "Blocks is still running"; exit 1; fi
    echo "Blocks stopped"
    ;;
  status)
    if running; then echo "Blocks is running"; else echo "Blocks is not running"; fi
    if port_open; then echo "RPC port $PORT is open"; else echo "RPC port $PORT is closed"; fi
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac

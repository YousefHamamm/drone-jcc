#!/usr/bin/env bash
# pipeline.sh: the AirSim training run on this branch, as it was run (as root, from the repository, in tmux, with Blocks
# running from airsim_headless.sh).
#
#   bash results/airsim_train/pipeline.sh run      stop always_send (runs/air_always_send, started from the surrogate
#                                                  session's always_send) once it passes STOP_AT slots, push its
#                                                  checkpoint, train joint_push and then joint_pull from it for
#                                                  JOINT_SLOTS slots each, and evaluate always send, push and pull in
#                                                  AirSim on common random numbers (eval_three.py)
#   bash results/airsim_train/pipeline.sh pusher   every 30 minutes, copy the checkpoints, train logs and learning
#                                                  curves into results/airsim_train/ and push them
#   bash results/airsim_train/pipeline.sh push     do that once, now
#
# Every AirSim slot is checked against the sim clock and topped up to DT (SLOT_TOPUPS); the train logs report the
# counts ("slots topped up N/total, off M").
set -u
REPO=/home/user/drone-jcc
cd "$REPO" || exit 1
S=runs/air                      # the AirSim session folder: always_send, joint_push, joint_pull, eval
STOP_AT=50000                   # always_send slots
JOINT_SLOTS=40960               # slots per joint phase: 80 updates of 512
COMMON=(--set TORCH_THREADS=1 --set DRAW=False)
EVAL_SETS=(--set EVAL_REPS_AIRSIM=6 --set EVAL_SLOTS_AIRSIM=3000)
LOCK=$REPO/.git/airsim_push.lock
mkdir -p "$S"
LOG=$S/pipeline.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

src() {  # the run folder of a phase: always_send is runs/air_always_send until the pipeline stops and moves it
  if [ -d "$S/$1" ]; then echo "$S/$1"; elif [ "$1" = always_send ] && [ -d runs/air_always_send ]; then
    echo runs/air_always_send; fi
}

push_results() {  # $1: what the commit is about
  (
    flock 9
    plots=""
    for ph in always_send joint_push joint_pull; do
      r=$(src "$ph")
      [ -n "$r" ] || continue
      plots="$plots $r"
      d=results/airsim_train/$ph
      mkdir -p "$d"
      cp "$r"/ckpt_*.pt "$r/train_log.csv" "$r/config.json" "$d/" 2>/dev/null
      [ -f "$r.log" ] && tr '\r' '\n' < "$r.log" | grep -aoE "\[$ph\].*|\[airsim\] (sim time|a slot).*" > "$d/console.log"
    done
    [ -n "$plots" ] && nice -n 19 python drone_jcc.py plot $plots --out results/airsim_train/plots >/dev/null 2>&1
    if [ -f "$S/eval/eval.json" ]; then
      d=results/airsim_train/eval
      mkdir -p "$d"
      cp "$S/eval/eval.json" "$S/eval/eval.csv" "$d/" 2>/dev/null
      cp "$S/eval/summary.md" "$S"/eval/plots/*.png "$d/" 2>/dev/null
      [ -f "$S/eval.log" ] && tr '\r' '\n' < "$S/eval.log" | grep -av "^eval:\|^\s*$" > "$d/console.log"
    fi
    git add results/airsim_train
    git diff --cached --quiet -- results/airsim_train && exit 0
    git commit -q -m "AirSim training: $1" \
      -m "Checkpoints, train logs and learning curves of the AirSim run (results/airsim_train/pipeline.sh)." \
      -m $'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_0171LPFsSA1s7tkLz6j1gbj5' \
      -- results/airsim_train
    for w in 0 2 4 8 16; do
      sleep $w
      git push -q -u origin throughput 2>>"$LOG" && { echo "[$(date -u +%H:%M:%S)] pushed: $1" >> "$LOG"; exit 0; }
    done
    echo "[$(date -u +%H:%M:%S)] push failed: $1" >> "$LOG"
  ) 9>"$LOCK"
}

phase_run() {  # $1: joint_push or joint_pull
  log "start $1 from $S/always_send/ckpt_latest.pt for $JOINT_SLOTS slots"
  python drone_jcc.py train --phase "$1" --run "air/$1" --init "$S/always_send/ckpt_latest.pt" --no-dashboard \
    "${COMMON[@]}" --set TOTAL_SLOTS_AIRSIM=$JOINT_SLOTS > "$S/$1.log" 2>&1
  log "$1 exited with code $?"
  push_results "$1 done"
}

case "${1:-}" in
  run)
    A=runs/air_always_send
    log "waiting for always_send to pass $STOP_AT slots"
    until [ "$(tail -1 "$A/train_log.csv" 2>/dev/null | cut -d, -f2)" -ge "$STOP_AT" ] 2>/dev/null; do sleep 15; done
    log "always_send at $(tail -1 "$A/train_log.csv" | cut -d, -f2) slots: stopping it with Ctrl+C"
    tmux send-keys -t train C-c
    for _ in $(seq 1 120); do pgrep -f "^python drone_jcc.py train" >/dev/null || break; sleep 1; done
    pgrep -f "^python drone_jcc.py train" >/dev/null && { log "always_send did not stop"; exit 1; }
    mv "$A" "$S/always_send" && mv runs/air_always_send.log "$S/always_send.log"
    tmux kill-session -t dash 2>/dev/null
    tmux new-session -d -s dash "cd $REPO && exec nice -n 19 python drone_jcc.py dashboard $S --port 8765 \
--host 0.0.0.0 --refresh 2 > $S/dashboard.log 2>&1"
    push_results "always_send stopped at $(tail -1 "$S/always_send/train_log.csv" | cut -d, -f2) slots"
    phase_run joint_push
    phase_run joint_pull
    log "evaluate always send, joint push and joint pull in AirSim"
    python results/airsim_train/eval_three.py "$S" "${COMMON[@]}" "${EVAL_SETS[@]}" > "$S/eval.log" 2>&1
    log "evaluation exited with code $?"
    push_results "evaluation of push, pull and always send"
    log "pipeline done"
    ;;
  push)
    push_results "${2:-checkpoints and learning curves at $(date -u +%H:%M) UTC}"
    ;;
  pusher)
    while true; do
      sleep 1800
      push_results "checkpoints and learning curves at $(date -u +%H:%M) UTC"
    done
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac

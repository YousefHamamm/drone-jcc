#!/usr/bin/env bash
# pipeline.sh: the AirSim training run on this branch, as it was run (as root, from the repository, in tmux, with Blocks
# running from airsim_headless.sh).
#
#   bash results/airsim_train/pipeline.sh run      stop always_send (runs/air_always_send, started from the surrogate
#                                                  session's always_send) once it passes STOP_AT slots and push its
#                                                  checkpoint; fine-tune joint_push and then joint_pull in AirSim from
#                                                  the surrogate session's joint checkpoints, JOINT_SLOTS slots each,
#                                                  with a start check (below); then evaluate five policies in AirSim on
#                                                  common random numbers (eval_five.py)
#   bash results/airsim_train/pipeline.sh pusher   every 30 minutes, copy the checkpoints, train logs and learning
#                                                  curves into results/airsim_train/ and push them
#   bash results/airsim_train/pipeline.sh push     do that once, now
#
# Start check: when a joint phase has CHECK_UPDATES updates, the mean J of those updates is compared with the
# surrogate's J for the same policy with sampled actions (its evaluation). If it is more than CHECK_DROP below, the
# phase is stopped and the pipeline halts.
#
# Every AirSim slot is checked against the sim clock and topped up to DT (SLOT_TOPUPS); the train logs report the
# counts ("slots topped up N/total, off M").
set -u
REPO=/home/user/drone-jcc
cd "$REPO" || exit 1
S=runs/air                      # the AirSim session folder: always_send, joint_push, joint_pull, eval
SURR=runs/surr_tau155           # the surrogate session the joint phases start from
STOP_AT=50000                   # always_send slots
JOINT_SLOTS=40960               # slots per joint phase: 80 updates of 512
CHECK_UPDATES=3
CHECK_DROP=0.20
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
    for ph in joint_push joint_pull; do  # the surrogate checkpoints the fine-tuning starts from
      d=results/airsim_train/surrogate_init/$ph
      [ -f "$d/ckpt_latest.pt" ] && continue
      mkdir -p "$d"
      cp "$SURR/$ph/ckpt_latest.pt" "$SURR/$ph/train_log.csv" "$SURR/$ph/config.json" "$d/" 2>/dev/null
    done
    [ -f results/airsim_train/surrogate_init/eval.csv ] || cp "$SURR/eval/eval.csv" results/airsim_train/surrogate_init/
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

ref_J() {  # the surrogate's J for a joint phase's policy with sampled actions, from the surrogate's evaluation
  python - "$SURR/eval/eval.json" "$1" <<'EOF'
import json, sys
name = {"joint_push": "joint push (learned, sampled actions)", "joint_pull": "joint pull (learned, sampled actions)"}
print(round(next(r["J"] for r in json.load(open(sys.argv[1]))["results"] if r["label"] == name[sys.argv[2]]), 5))
EOF
}

start_check() {  # in the background while a joint phase trains; $1: the phase
  local ph=$1 tl=$S/$1/train_log.csv ref n m
  ref=$(ref_J "$ph")
  while :; do
    n=$(( $(wc -l < "$tl" 2>/dev/null || echo 1) - 1 ))
    [ "$n" -ge "$CHECK_UPDATES" ] && break
    pgrep -f "^python drone_jcc.py train --phase $ph" >/dev/null || { [ "$n" -gt 0 ] || sleep 20; }
    pgrep -f "^python drone_jcc.py train --phase $ph" >/dev/null || return 0
    sleep 10
  done
  m=$(python -c "import csv; r = list(csv.DictReader(open('$tl')))[:$CHECK_UPDATES]; \
print(round(sum(float(x['J']) for x in r) / len(r), 5))")
  if python -c "import sys; sys.exit(0 if $m >= (1 - $CHECK_DROP) * $ref else 1)"; then
    log "$ph start check passed: mean J of the first $CHECK_UPDATES updates $m; surrogate $ref (sampled actions)"
  else
    log "STOPPED: $ph start check failed: mean J of the first $CHECK_UPDATES updates $m is more than" \
        "$CHECK_DROP below the surrogate's $ref (sampled actions)"
    touch "$S/$ph.stopped"
    pkill -INT -f "^python drone_jcc.py train --phase $ph"
  fi
}

phase_run() {  # $1: joint_push or joint_pull; returns 1 when the start check stopped it
  local ph=$1 ck=$SURR/$1/ckpt_latest.pt
  rm -f "$S/$ph.stopped"
  log "start $ph in AirSim from $ck for $JOINT_SLOTS slots"
  start_check "$ph" &
  python drone_jcc.py train --phase "$ph" --run "air/$ph" --init "$ck" --no-dashboard "${COMMON[@]}" \
    --set TOTAL_SLOTS_AIRSIM=$JOINT_SLOTS > "$S/$ph.log" 2>&1
  log "$ph exited with code $?"
  wait
  if [ -f "$S/$ph.stopped" ]; then
    push_results "$ph stopped by the start check"
    return 1
  fi
  push_results "$ph done"
}

case "${1:-}" in
  run)
    A=runs/air_always_send
    if [ -d "$A" ]; then
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
    fi
    phase_run joint_push || { log "pipeline halted after the joint_push start check"; exit 1; }
    phase_run joint_pull || { log "pipeline halted after the joint_pull start check"; exit 1; }
    log "evaluate five policies in AirSim"
    python results/airsim_train/eval_five.py "$S" "$SURR" "${COMMON[@]}" "${EVAL_SETS[@]}" > "$S/eval.log" 2>&1
    log "evaluation exited with code $?"
    push_results "evaluation of five policies"
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
    sed -n '2,22p' "$0"
    exit 1
    ;;
esac

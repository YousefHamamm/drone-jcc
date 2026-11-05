# drone_jcc: joint communication and control for a drone that hits targets (AirSim)

Research code for a paper on task-completion throughput over an unreliable channel. `drone_jcc.py` implements the
paper's model with an AirSim SimpleFlight drone. All parameters are in the PARAMETERS block at the top. README.md
describes the model, the training phases (always_send, joint_push, joint_pull) and the outputs.

Throughput is J = (successes - COST_C x transmissions) / slots, with held and reset slots counted.

## Commands
- No simulator needed: `python drone_jcc.py selftest` (1 to 2 min) and `python test_offline.py` (the AirSim code
  path against `fake_airsim_server.py`)
- AirSim preflight, with the simulator running: `python drone_jcc.py check`
- Settings file for AirSim: `python drone_jcc.py settings --write`
- Surrogate session: `python drone_jcc.py all --backend surrogate --run <name>`
- AirSim session that starts from a surrogate session: `python drone_jcc.py all --run <name> --init-session runs/<surrogate>`
- Watch a policy: `python drone_jcc.py watch --ckpt <ckpt>`; without `--ckpt` the PD controller flies
- Evaluate and plot: `python drone_jcc.py eval --session runs/<name>`, `python drone_jcc.py plot runs/<name>`
- Any parameter: `--set NAME=VALUE`

## Rules
- The paper's model stays the default. New behavior goes behind a new parameter in the PARAMETERS block, default
  off. Ask me before changing a default of the model (HOLD_SLOTS, RESET_SLOTS, the target rule, the channel).
- A claim that something raises throughput needs `eval` on common random numbers: the paired difference to the
  baseline with its 95% CI. Report J per slot and tasks per second (J / DT).
- Every long-running script shows live tqdm progress and prints partial results as each group of work completes.
- Runs longer than a few minutes go in the background (tmux or nohup) with output to a log file; check the log
  instead of waiting on the command.
- After code changes, `python drone_jcc.py selftest` and `python test_offline.py` must pass.
- Make targeted edits; do not rewrite files. Commit to git after each working step.
- Text written for me (README, reports): plain direct sentences, no em dashes.

## AirSim facts that matter
- SimpleFlight hovers when no API command arrives for 60 ms (`api_goal_timeout`). That is why stepping is
  lockstep: the simulator stays paused, each command lasts DT - CMD_MARGIN, then `simContinueForTime(DT)`.
- AirSim's RPC server has (vehicles + 4) worker threads, and a move command holds one until it ends. Never send
  commands that outlast their slot.
- A vehicle's position is relative to its own start point; markers are drawn in the world frame. The start offsets
  come from the simulator's settings.
- `pip install airsim` fails on Python 3.10 and later (it needs tornado 4); the script's built-in client needs only
  `msgpack`.
- So far the code has been tested only against `fake_airsim_server.py`, not against the real simulator.

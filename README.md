# Drone experiment: joint communication and control in AirSim

`drone_jcc.py` runs the paper's task-throughput model with an AirSim drone as the robot. The drone flies to the
current target, a red marker. When it is within `HIT_RADIUS` of the target at the end of a slot, the marker turns
green and a new red marker appears. The objective is the throughput
J = (successes − C × transmissions) / slots, with held and reset slots counted.

All parameters are at the top of `drone_jcc.py`. Any of them can also be set from the command line with
`--set NAME=VALUE`.

## Files

| file | what it is |
|---|---|
| `drone_jcc.py` | the experiment: AirSim and surrogate backends, the task and channel model, PPO, evaluation, dashboard, plots |
| `fake_airsim_server.py` | a stand-in for the AirSim RPC server, so the AirSim code path can run without Unreal |
| `test_offline.py` | end-to-end tests: the surrogate, then the full AirSim path against the fake server |
| `example_results/` | plots and the evaluation table of one surrogate session, and a dashboard frame |

## Model (the paper's, with a drone)

* **Slot** (`DT` = 0.25 s). The order within a slot is (1) the scheduler decides whether the state is sent, (2) the
  channel delivers it with probability `P_S` at cost `COST_C`, and both sides learn the outcome (silent, lost,
  delivered), (3) the controller sends a velocity command, which always arrives, and (4) the drone moves.
* **Drone.** SimpleFlight in AirSim, driven with `moveByVelocityZ` at altitude `ALTITUDE` (or `moveByVelocity` when
  `DIM = 3`). Commands are bounded by `U_MAX`. A disturbance `NOISE_STD` (m/s, i.i.d. per slot) is added to the
  command; this is the paper's w_n. Wind can be added with `WIND_STD`.
* **Task.** A success is |x − g| ≤ `HIT_RADIUS` at the end of a slot. The drone is then held for `HOLD_SLOTS` slots
  and the next task starts where it is. A failure is leaving the safe box (`SAFE_HALF`), a collision, or
  `MAX_TASK_SLOTS` slots without a hit. After a failure the drone is put back at `START_POS` and held for
  `RESET_SLOTS` slots.
* **Information.** Push: the drone decides from its own state and the common record. Pull: the controller decides
  from its record. The command depends on the controller's record after the outcome. The record is the last
  delivered state, propagated through the commands sent since (time constant `PRED_TAU`), plus its age, the losses
  and silences since, the last outcomes and the task time. After a success the estimate is projected onto the target
  window. After a failure the controller knows the drone is at `START_POS`. `LOSS_VISIBLE = False` makes a loss look
  like silence to the controller.

## Learning

PPO with two heads that see only what they are allowed to see:

* the scheduler: a Bernoulli policy on the drone's inputs (push) or the controller's inputs (pull);
* the controller: a Gaussian velocity policy on the controller's record after the outcome.

The critic sees everything and is used only in training. The ratio in the PPO objective is the joint ratio of the
two heads. The reward of a slot is successes − C × transmissions − λ × slots, with a Dinkelbach price λ that
follows the running throughput (`PRICE_MODE = "adaptive"`). At the fixed point λ = J, a policy that maximizes
this reward maximizes the ratio J, held and reset slots included (up to discounting, `GAMMA` = 0.99).
Potential-based shaping (`SHAPING`) speeds up the first phase and does not change the optimal policy.

Phases:

1. `always_send`: send in every task slot and train the controller.
2. `joint_push`: train the drone's scheduler and the controller together, starting from phase 1.
3. `joint_pull`: train the controller's scheduler and the controller together, starting from phase 1.

The evaluation compares the learned policies with rule baselines on common random numbers, with 95% Student-t
confidence intervals. The baselines are periodic sending, an estimation-error threshold, sending near the target,
an age threshold, never sending, a PD controller, and a free-information reference.

## Install

```
pip install numpy torch tqdm matplotlib msgpack scipy tensorboard
pip install airsim          # optional; see the note below
```

On Python 3.10 and later, `pip install airsim` often fails because its dependency `msgpack-rpc-python` needs
tornado 4. That is fine: with `AIRSIM_CLIENT = "auto"` the script uses its built-in client, which needs only
`msgpack`. In tests against the fake server it was also about 10 times faster per call (0.5 ms against 5 ms). To
use it even when `airsim` is installed, set `AIRSIM_CLIENT="'lite'"`.

## AirSim setup

1. Get an AirSim environment. The Blocks binary from the AirSim releases is the lightest. Any Multirotor
   environment works if the arena (±`SAFE_HALF` m around the start, at `ALTITUDE` m) is free of obstacles. A
   collision counts as a failure, so if a block is in the way, raise `ALTITUDE` or move the PlayerStart.
2. Write the settings file: `python drone_jcc.py settings --write`. It goes to `~/Documents/AirSim/settings.json`,
   and an existing file is kept as a backup. It sets `SimMode` Multirotor, `NUM_DRONES` SimpleFlight vehicles spaced
   by `ARENA_SPACING` m, `ClockSpeed` = `CLOCK_SPEED`, the SteppableClock and `ViewMode`.
3. Start the simulator (on Linux, for example, `./Blocks.sh -windowed -ResX=1280 -ResY=720`; on a machine
   without a display, add `-RenderOffscreen` and watch through the dashboard). Then run
   `python drone_jcc.py check`. It reports the RPC round trip, the sim time and wall time per slot, and the
   measured velocity time constant; set `PRED_TAU` (and `TAU` for the surrogate) to that value. It then draws a
   red and a green test marker.

**Headless, without a GPU (a cloud VM).** `bash airsim_headless.sh setup` creates the user `airsim` (Unreal refuses
to run as root), downloads the Blocks v1.8.1 Linux release into its home and writes its settings file.
`bash airsim_headless.sh start` starts Blocks with `-nullrhi` (no rendering) in a tmux session and waits for the RPC
port. Then run `check` and the other commands as usual. Nothing is drawn, so markers and camera images cannot be
seen.

**Stepping.** With `STEP_MODE = "lockstep"` the simulator stays paused. In each slot every drone gets one command
lasting `DT − CMD_MARGIN`, and then the simulator advances DT. AirSim 1.8.1 runs `simContinueForTime(x)` for x
seconds of wall time, so with the SteppableClock the sim clock moves ClockSpeed × x; each slot therefore calls
`simContinueForTime(DT / ClockSpeed)` (`CONTINUE_SCALE = "auto"`), and `check` shows the sim time per slot. There are two reasons for
this. SimpleFlight switches to hover if no command arrives for 60 ms, and AirSim's RPC server has only
(vehicles + 4) worker threads. With these settings every command ends inside its slot, the next one starts before
the clock moves, and the server never runs out of threads for any number of drones. Training pauses do not move the
drones. `ClockSpeed` 5 to 10 is usually stable.

## Run

Quick check of the installation (one to two minutes):
```
python drone_jcc.py selftest
python test_offline.py          # also exercises the AirSim path against fake_airsim_server.py
```

Everything on the surrogate (minutes):
```
python drone_jcc.py all --backend surrogate --run pre
```

Everything in AirSim, each phase starting from the surrogate session (recommended):
```
python drone_jcc.py all --run air --init-session runs/pre
```

Or from scratch in AirSim (`TOTAL_SLOTS_AIRSIM` slots per phase):
```
python drone_jcc.py all --run air
```

One phase at a time:
```
python drone_jcc.py train --phase always_send --run a1
python drone_jcc.py train --phase joint_push --init runs/a1/ckpt_latest.pt --run p1
python drone_jcc.py train --phase joint_pull --init runs/a1/ckpt_latest.pt --run q1
```
(Without `--init`, a joint phase starts from the newest `always_send` run under `runs/`.)

Evaluate and plot:
```
python drone_jcc.py eval --session runs/air
python drone_jcc.py eval --ckpt runs/p1/ckpt_latest.pt --sched learned
python drone_jcc.py eval --ckpt runs/a1/ckpt_latest.pt --sched err --param 0.4
python drone_jcc.py plot runs/air
```

Watch a policy fly (AirSim window plus dashboard):
```
python drone_jcc.py watch --ckpt runs/air/joint_push/ckpt_latest.pt
python drone_jcc.py watch                       # PD controller, always sending: tests the setup, no training
```

Sweeps are command-line overrides, e.g. `--set P_S=0.3 --set COST_C=0.02`.

## Watching and tracking

* **AirSim window.** The current target is a red dot with a red ring of radius `HIT_RADIUS`. Hit targets turn
  green; the last `SHOW_HITS` are kept. The safe box is grey. Drone 1 leaves a blue trail, with cyan where its state
  was delivered and orange where it was lost. The yellow dot is where the controller thinks the drone is. Live
  numbers are printed in the window: slots, J, tasks, sends per task.
* **Dashboard.** `train`, `all` and `watch` start it at `http://127.0.0.1:8765/`. It shows a top view of drone 1
  (trajectory, the controller's estimate, deliveries and losses, targets) and curves of J, the price, sends and
  deliveries per task, success rate, slots per task and send probability. On a remote machine, use
  `ssh -L 8765:127.0.0.1:8765 host`. With `CAMERA_STREAM = True` it also shows drone 1's camera. To reopen it:
  `python drone_jcc.py dashboard runs/<run>`, or `--mode window` for a matplotlib window.
* **Console.** A tqdm bar with live J, success rate, sends per task, send probability, price and KL, and a summary
  line every `LOG_EVERY` updates.
* **Run folder** (`runs/<run>/`):
  * `config.json`: every parameter, the phase, the init checkpoint, the command line
  * `train_log.csv`: one row per update (J, success, failures by kind, sends and deliveries per task, slots per
    task, send probability, price, KL, learning rate, entropies, losses, explained variance)
  * `tasks.csv`: one row per task of every drone (outcome, slots, sends, deliveries, final distance, target, start)
  * `trace.csv`: every slot of drone 1 (state, estimate, target, decision, outcome, command, age)
  * `live.json`, read by the dashboard
  * `tb/`: TensorBoard scalars (`tensorboard --logdir runs`)
  * `ckpt_latest.pt`, `ckpt_best.pt`, `ckpt_000NN.pt`: networks, input normalization, price, parameters
  * `eval/eval.json`, `eval/eval.csv`, `plots/*.png` for sessions

Ctrl+C stops a run and keeps the latest checkpoint and the logs.

## Main parameters

| parameter | default | meaning |
|---|---|---|
| `BACKEND` | `airsim` | `airsim` or `surrogate` |
| `NUM_DRONES` | 1 | drones trained in parallel (must match settings.json) |
| `P_S`, `COST_C` | 0.6, 0.01 | channel |
| `DT`, `U_MAX`, `NOISE_STD` | 0.25 s, 3 m/s, 0.9 m/s | slot, command bound, disturbance |
| `HIT_RADIUS`, `TARGET_MIN_DIST`, `TARGET_MAX_DIST` | 0.5, 4, 12 m | target window and spacing |
| `HOLD_SLOTS`, `RESET_SLOTS`, `MAX_TASK_SLOTS` | 2, 20, 120 | Δ_S, Δ_F, timeout |
| `LOSS_VISIBLE`, `SENDER_KNOWS_TARGET` | True, True | information structure |
| `TOTAL_SLOTS`, `TOTAL_SLOTS_AIRSIM` | 4M, 300k | training length per phase |
| `PRED_TAU`, `TAU` | 1.55 s, 1.55 s | the controller's model of the velocity response, and the surrogate's (`check` measured 1.55 s for SimpleFlight) |

## Reference numbers (surrogate, default parameters)

One `all` session on the surrogate: 3M slots per phase (the default is now 4M), p_s = 0.6, C = 0.01. The
evaluation used 16 replications of 20,000 slots on common random numbers; the files are in `example_results/`.
These numbers are from before `TAU` and `PRED_TAU` were set to SimpleFlight's measured 1.55 s; they used 0.35 s.

| policy | J (95% CI) | vs always send | sends per task | slots per task |
|---|---|---|---|---|
| free state every slot (C = 0, p_s = 1), reference | 0.0677 ± 0.0003 | +0.0098 | 12.8 (free) | 12.8 |
| joint push (learned) | 0.0625 ± 0.0002 | +0.0046 ± 0.0001 | 4.8 | 13.2 |
| joint pull (learned) | 0.0623 ± 0.0002 | +0.0045 ± 0.0001 | 4.8 | 13.3 |
| push rule: send within 3 m of the target | 0.0616 ± 0.0003 | +0.0038 ± 0.0003 | 4.1 | 13.4 |
| pull rule: ask when the age is ≥ 2 | 0.0589 ± 0.0003 | +0.0011 ± 0.0001 | 8.7 | 13.5 |
| always send (learned controller) | 0.0579 ± 0.0003 | 0 | 13.0 | 13.0 |
| push rule: estimation error > 0.3 m | 0.0577 ± 0.0002 | −0.0002 ± 0.0002 | 4.9 | 14.5 |
| periodic, every 2 slots | 0.0574 ± 0.0002 | −0.0005 ± 0.0002 | 7.3 | 14.1 |
| PD controller, always send | 0.0551 ± 0.0003 | −0.0028 ± 0.0002 | 13.7 | 13.7 |
| never send | 0.0081 ± 0.0004 | −0.0498 ± 0.0006 | 0 | 25.3 |

The success rate was 0.999 or higher, except for the near-target rule with a 1.5 m radius (0.94) and for never
sending (0.59). The rule baselines use the always-send controller, which was not trained for them.

`scheduler_maps.png` shows where each scheduler sends. The learned push scheduler sends close to the target
whatever the error, and farther away only when the error is large; its error threshold falls as the drone gets
closer. Neither an error threshold nor a distance rule has that shape.

Push beats pull by 0.00014 ± 0.00009 (paired), much less than in the paper's scalar model. PPO finds good
policies, not certified optimal ones, so a small gap can come from the learning as well as from the setup. A
sweep over C and p_s, and longer joint training, would tell the two apart.

## Notes

* The surrogate models the drone with a first-order velocity response and an acceleration limit. Set its `TAU` to
  the value `check` measures, so that the policies trained on it transfer. For SimpleFlight in Blocks 1.8.1 it is
  1.55 s; the measured response also starts more slowly than a first-order one (0.20 m/s after the first slot of a
  2 m/s step, against 0.30 m/s for the fit).
* In AirSim, a vehicle's position is relative to its own start point, while markers are drawn in the world frame.
  The script reads each vehicle's start offset from the simulator's settings.
* A deterministic evaluation (`EVAL_DETERMINISTIC`) sends if and only if the learned probability exceeds 0.5. The
  default evaluation also runs the learned policies with sampled actions.

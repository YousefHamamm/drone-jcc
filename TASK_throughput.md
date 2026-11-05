# Task: run the drone experiment in AirSim and raise its throughput

## Where this runs
You are in a Claude Code cloud session: an Ubuntu VM with 4 vCPUs and 16 GB of RAM, no GPU, no display and no
AirSim. Do everything that runs without the simulator: the surrogate backend and `fake_airsim_server.py`.

The steps marked **(local)** need the real simulator on my machine. For those, write the exact commands into
`RUN_LOCAL.md`, with what to look for in the output. I will run them after `git pull` and paste the results back
to you.

Commit and push the branch after each step, so I can pull it in VS Code.

## Where things stand
Default parameters, surrogate backend, 3M training slots per phase, evaluated on 16 x 20,000 slots:

| policy | J per slot | sends per task | slots per task |
|---|---|---|---|
| free state every slot (C = 0, p_s = 1) | 0.0677 | | 12.8 |
| joint push (learned) | 0.0625 | 4.8 | 13.2 |
| joint pull (learned) | 0.0623 | 4.8 | 13.3 |
| always send | 0.0579 | 13.0 | 13.0 |

The joint policies already get 92% of the free-information value. J is low because a task is long:
- the targets are 8.25 m apart on average;
- at U_MAX = 3 m/s a straight line needs 10.8 slots of 0.25 s, and the drone takes 13.3, so about 2.5 slots per task
  go to accelerating from rest and slowing down to land in the 0.5 m window;
- each success is followed by HOLD_SLOTS = 2 held slots.

So 0.062 tasks per slot is 0.25 tasks per second. Throughput rises if tasks get shorter.

## Step 1: make sure it works (before anything else)
1. Install the dependencies if they are missing (`pip install numpy torch tqdm matplotlib msgpack scipy
   tensorboard`). Then `python drone_jcc.py selftest` and `python test_offline.py` must pass.
2. **(local)** Put these in `RUN_LOCAL.md`:
   - `python drone_jcc.py settings --write`, then start AirSim, then `python drone_jcc.py check`. I will report the
     RPC round trip, the sim time per slot against DT, the measured velocity time constant, and whether the red and
     green markers appeared.
   - `python drone_jcc.py watch --slots 500` with the PD controller. I will check that the markers turn from red to
     green, that there are no collisions, and that the AirSim window never shows "API call was not received,
     entering hover mode for safety".
3. When I paste those results, fix whatever breaks against the real simulator (the code was tested only against
   `fake_airsim_server.py`), and set `PRED_TAU` and `TAU` to the measured time constant.

## Step 2: measure where the slots go
Write `analyze_tasks.py`. It reads a run's `tasks.csv` and `trace.csv` and reports, per task and averaged:
- the target distance;
- the slots used;
- the straight-line bound ceil((d - HIT_RADIUS) / (U_MAX x DT));
- the slots spent accelerating, cruising and landing (from speed and distance in `trace.csv`);
- the sends and the held slots;
- J per slot and per second.

Run it on a surrogate session and show me the table. **(local)** Add the command that runs it on my AirSim watch
run to `RUN_LOCAL.md`. Use the table to rank the levers below by how many slots each can save.

## Step 3: improvements, one at a time
Add each one behind a new parameter that defaults to the current behavior. Compare it to the baseline with `eval` on
common random numbers: the paired difference in J and its 95% CI. Do this on the surrogate. **(local)** Put the
commands that confirm the best ones in AirSim in `RUN_LOCAL.md`.

a. **Next-target preview** (`TARGET_PREVIEW = 1`). Draw the next target ahead of time and show it to the controller
   and the push scheduler (in the paper both sides know the targets). The drone can then go through the current
   target already heading for the next one instead of stopping. Show it as a second, dimmer red marker in AirSim and
   on the dashboard.
b. **Hit along the path** (`HIT_CHECK = "segment"`). A success when the path during the slot passes within
   HIT_RADIUS of the target, not only when the end point does. This matters once the drone flies through targets at
   speed.
c. **Speed.** Try `U_MAX` = 4.5 and 6 m/s on the surrogate. **(local)** Add a `check` option that measures whether
   SimpleFlight tracks those speeds and what `ACC_MAX` and `TAU` the surrogate should use, and put the command in
   `RUN_LOCAL.md`.
d. **No hold** (`HOLD_SLOTS = 0`) for targets that are only hit, not docked at. This changes the model, so ask me
   before making it a default.
e. **Training.** The joint phases were still reducing sends at 3M slots. Try 8M, and `ENT_SCHED` 0 against 0.003.
f. Anything else that step 2 shows, explained with numbers.

## Step 4: report
Write `REPORT.md`. It should have:
- a table of every variant: J per slot, tasks per second, sends per task, success rate, slots per task, and the paired
  difference to the baseline with its CI;
- what changed in the code;
- the settings you recommend.

**(local)** Put in `RUN_LOCAL.md` the commands for a full AirSim session with those settings, starting from a
surrogate session (`all --init-session`). When I paste its evaluation, add it to the report.

## Constraints
- Every claim comes with the command that produced it and its numbers.
- Training runs take minutes to tens of minutes on 4 vCPUs. Start them in the background with output to a log file,
  check the log, and keep each one short enough to finish inside the session's command time limits.
- Commit and push after each step.

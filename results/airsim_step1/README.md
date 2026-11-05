# Step 1 in AirSim: Blocks v1.8.1, headless in the cloud VM

Setup: `bash airsim_headless.sh setup`, then `bash airsim_headless.sh start`. Blocks ran with `-nullrhi`, ClockSpeed 5
and the SteppableClock. `check` and `watch` ran with the parameters of that time: `PRED_TAU = 0.35`.

| file | command | what it shows |
|---|---|---|
| `check_before_fix.log` | `python drone_jcc.py check`, before `CONTINUE_SCALE` | 1.2486 s of sim time per slot instead of 0.25 s, and 0.03 m/s at the end of every slot of the 2 m/s step |
| `continue_probe.py` | `python results/airsim_step1/continue_probe.py` | `simContinueForTime(x)` moves the sim clock by 4.98 x to 5.00 x at ClockSpeed 5, in x seconds of wall time |
| `check.log` | `python drone_jcc.py check` | 0.2499 s of sim time per slot (0.2490 to 0.2520 s); time constant 1.55 s, fit RMS 0.034 m/s, largest error 0.099 m/s |
| `response_probe.py`, `response_probe_clockspeed5.log` | `python results/airsim_step1/response_probe.py` | 1.55 s with a command per slot, 1.56 s with one 4 s command, 1.57 s for a 3 m/s step |
| `response_probe_clockspeed1.log` | the same, after `bash airsim_headless.sh setup --set CLOCK_SPEED=1.0` and a restart | the same response at ClockSpeed 1 |
| `watch_pd_500.log` | `python drone_jcc.py watch --slots 500 --run air_watch_pd --no-dashboard --set WATCH_REALTIME=False` | 16 successes, 0 failures, J 0.0226 |
| `analyze_air_watch_pd.md`, `air_watch_pd_tasks.csv` | `python analyze_tasks.py runs/air_watch_pd` | 26.88 slots per successful task: 10.69 accelerating, 4.31 cruising, 11.88 landing |

With `TAU = PRED_TAU = 1.55` (the new defaults), the PD controller was flown again in both:

| file | command | what it shows |
|---|---|---|
| `watch_pd_500_tau155.log`, `air_watch_pd_tau155_tasks.csv` | `python drone_jcc.py watch --slots 500 --run air_watch_pd_tau155 --no-dashboard --set WATCH_REALTIME=False` | AirSim: 19 successes, 0 failures, J 0.0288 |
| `analyze_air_vs_surr_tau155.md` | `python analyze_tasks.py runs/air_watch_pd_tau155 runs/surr_watch_pd_tau155`, the second from `python drone_jcc.py watch --backend surrogate --slots 4000 --run surr_watch_pd_tau155 --no-dashboard --set WATCH_REALTIME=False` | 23.47 slots per task in AirSim against 22.64 on the surrogate; accelerating 11.21 against 11.43, landing 6.74 against 5.09 |

**Slot timing.** The 500-slot runs above ran before the backend checked the sim time of each slot. Under CPU load
AirSim's physics thread falls behind and, because `simContinueForTime` counts wall time, a slot can come out short:
the first AirSim training run had 84 of 2560 slots more than 10% short, some at 0.036 s. `topup_probe.py` (written
before `SLOT_TOPUPS`, so it tops up at most 5 times) showed the physics running at about 0.5x to 1.3x real time in
those episodes instead of 5x. Since `SLOT_TOPUPS`, a short slot is continued until it reaches DT. So the 500-slot
numbers may include short slots; this run with the check is the one to use:

| file | command | what it shows |
|---|---|---|
| `watch_pd_1000_topup.log`, `air_watch_pd_topup_tasks.csv` | `python drone_jcc.py watch --slots 1000 --run air_watch_pd_topup2 --no-dashboard --set WATCH_REALTIME=False` | 40 successes, 0 failures, J 0.0308; 51 of 1000 slots topped up, none left more than 10% off (0.234 to 0.252 s) |
| `analyze_air_watch_pd_topup.md` | `python analyze_tasks.py runs/air_watch_pd_topup2` | 22.95 slots per task: 11.30 accelerating, 6.38 cruising, 5.28 landing; J 0.0309 per slot, 0.124 tasks per second |

With correct slots the PD controller matches the surrogate with `TAU = PRED_TAU = 1.55` closely: 22.95 slots per
task against 22.64, and 11.30 / 6.38 / 5.28 against 11.43 / 6.12 / 5.09.

The same 500-slot PD watch on the surrogate (with `TAU = PRED_TAU = 0.35`) gave 30 to 33 successes and J from 0.051 to
0.057 over seeds 1 to 10, and 13.76 slots per task over 4000 slots.

Nothing is rendered with `-nullrhi`, so the red and green markers could not be seen. The marker calls returned without
errors.

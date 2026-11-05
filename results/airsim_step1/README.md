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

The same 500-slot PD watch on the surrogate (with `TAU = PRED_TAU = 0.35`) gave 30 to 33 successes and J from 0.051 to
0.057 over seeds 1 to 10, and 13.76 slots per task over 4000 slots.

Nothing is rendered with `-nullrhi`, so the red and green markers could not be seen. The marker calls returned without
errors.

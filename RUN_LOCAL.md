# Commands to run with AirSim

These steps need the real simulator, so they run on your machine. Run them from the repository folder on the
`throughput` branch and paste the output back to me. Each step says what to look for.

## Before each step

```
git checkout throughput
git pull
```

The first time, install the dependencies (the same as in README.md):

```
pip install numpy torch tqdm matplotlib msgpack scipy tensorboard
```

## Step 1: check that the setup works

### 1a. Settings file and preflight check

1. Write the settings file. It goes to `~/Documents/AirSim/settings.json` (the command prints the path), and an
   existing file is kept as a backup.

   ```
   python drone_jcc.py settings --write
   ```

2. Start AirSim, or restart it if it is already running, so that it reads the new file. For the Blocks binary on
   Linux: `./Blocks.sh -windowed -ResX=1280 -ResY=720`.

3. With the simulator running:

   ```
   python drone_jcc.py check
   ```

Paste the whole output. This is what I look for:

| output line | expected |
|---|---|
| `client: ...; vehicles: [...]; start offsets: [...]` | one vehicle, `Drone1`, with start offset `[[0.0, 0.0, 0.0]]` |
| `settings: ClockSpeed ..., ClockType ...` | `ClockSpeed 5.0, ClockType SteppableClock` |
| `RPC round trip ... ms` | the number; it sets how fast training runs in AirSim |
| `sim time per slot ... s (DT 0.25), wall time per slot ... s` | sim time per slot close to 0.25 s, and no `[warning]` line after it |
| `sim time per slot: shortest ... s, longest ... s` | both close to 0.25 s; their difference is the slot jitter |
| `velocity after 1, 2, 4, 8 slots: ...` and `velocity after each of the 16 slots: ...` | speeds that rise toward 2 m/s |
| `first-order fit: time constant ... s` | the measured time constant; I will set `PRED_TAU` and `TAU` to it |
| `fit error: RMS ... m/s, largest ... m/s` | how far the measured speeds are from the fitted first-order response |

Also tell me whether a red ring appeared 4 m north of the drone and a green ring 4 m south, with the text
"drone_jcc check" in the AirSim window.

### 1b. Fly the PD controller for 500 slots

```
python drone_jcc.py watch --slots 500 --run air_watch_pd
```

This flies 500 slots with the PD controller, sending the state in every slot. It is paced to real time, so it takes
about 2 minutes. A browser tab opens with the dashboard. The logs go to `runs/air_watch_pd/`, and step 2 analyzes
them.

What to look for while it flies:

- Each red marker turns green when the drone reaches it, and a new red marker appears.
- The drone does not hit anything.
- The AirSim window does not show "API call was not received, entering hover mode for safety". One such message
  after the command has ended is expected: the script leaves the drone hovering and stops sending commands.

When it ends, paste the last line, which starts with `[watch] 500 slots:`, and the output of this command:

```
python -c "import csv, collections; print(collections.Counter(r['outcome'] for r in csv.DictReader(open('runs/air_watch_pd/tasks.csv'))))"
```

It should count only `success`. A `collision` means something in the arena is in the drone's way: raise the plane
of the targets with `--set ALTITUDE=10`, or move the PlayerStart.

For comparison, the same 500 slots on the surrogate gave 30 to 33 successes, no failures and J from 0.051 to 0.057
(mean 0.054) over seeds 1 to 10:

```
python drone_jcc.py watch --backend surrogate --slots 500 --no-dashboard --set WATCH_REALTIME=False --set SEED=1
```

Numbers well below that range in AirSim mean AirSim and the surrogate differ. A slower velocity response is one
possible cause, and the time constant from 1a shows whether it is the cause.

When you paste these results, I will fix whatever breaks against the real simulator and set `PRED_TAU` and `TAU`
to the measured time constant.

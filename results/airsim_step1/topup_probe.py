"""Written before SLOT_TOPUPS: watch with the PD controller in AirSim and log every top-up: the slot's sim time before and after each continue."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import drone_jcc as dj

LOG = []
orig = dj.AirSimBackend._track_slot

def traced(self):
    c, nm, dt = self.c, self.names[0], float(self.cfg.DT)
    ts = c.getMultirotorState(nm).timestamp
    if self._ts is None:
        self._ts = ts
        return
    d0 = (ts - self._ts) / 1e9
    steps = [round(d0, 4)]
    d, tops = d0, 0
    while dt - d > 0.006 and tops < 5:
        w = time.time()
        paused_before = c.simIsPause()
        c.simContinueForTime((dt - d) * self.cont_scale)
        self._wait_paused()
        ts = c.getMultirotorState(nm).timestamp
        d, tops = (ts - self._ts) / 1e9, tops + 1
        steps.append((round(d, 4), paused_before, round(time.time() - w, 4)))
    if tops:
        LOG.append((self.slot_dt[0] + 1, steps))
    s = self.slot_dt
    s[0] += 1
    s[1], s[2] = min(s[1], d), max(s[2], d)
    s[4] += tops > 0
    s[3] += abs(d - dt) > 0.1 * dt
    self._ts = ts

dj.AirSimBackend._track_slot = traced
cfg = dj.make_cfg(["BACKEND='airsim'", "WATCH_REALTIME=False",
                   "RUN_ROOT='runs'"])
dj.cmd_watch(cfg, slots=1000, dashboard=False, name="topup_probe")
print(f"{len(LOG)} slots topped up")
for k, steps in LOG:
    bad = steps[-1][0] < 0.244 if isinstance(steps[-1], tuple) else True
    if bad or len(steps) > 2:
        print("slot", k, steps)
print("first 5:", LOG[:5])

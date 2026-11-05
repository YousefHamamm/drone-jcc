"""Measure how far simContinueForTime(x) moves the AirSim clock, with the simulator paused in between."""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import drone_jcc as dj
c = dj.LiteAirSimClient("127.0.0.1", 41451, 30)
s = json.loads(c.getSettingsString())
print("ClockSpeed", s.get("ClockSpeed"), "ClockType", s.get("ClockType"))
c.simPause(True)
time.sleep(0.2)
for x in (0.05, 0.1, 0.25, 0.5, 1.0):
    out = []
    for _ in range(3):
        t0 = c.getMultirotorState("Drone1").timestamp
        w0 = time.time()
        c.simContinueForTime(x)
        while not c.simIsPause():
            time.sleep(0.0005)
        t1 = c.getMultirotorState("Drone1").timestamp
        out.append(((t1 - t0) / 1e9, time.time() - w0))
    print(f"continueForTime({x}): sim advance " + ", ".join(f"{a:.4f}" for a, _ in out)
          + f" s  (ratio {sum(a for a, _ in out) / (3 * x):.3f}); wall " + ", ".join(f"{w:.3f}" for _, w in out) + " s")
c.simPause(False)

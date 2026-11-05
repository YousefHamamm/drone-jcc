"""SimpleFlight velocity response in lockstep: per-slot commands at 2 and 3 m/s, and one long 2 m/s command."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import drone_jcc as dj

cfg = dj.make_cfg(["BACKEND='airsim'"])
b = dj.AirSimBackend(cfg, 1)
c, nm, DT = b.c, b.names[0], float(cfg.DT)

def fit(vs, amp):
    taus = np.linspace(0.05, 4.0, 791)
    k = np.arange(len(vs)) + 1
    errs = [np.sum((vs - amp * (1 - np.exp(-DT * k / t))) ** 2) for t in taus]
    t = float(taus[int(np.argmin(errs))])
    return t, float(np.sqrt(min(errs) / len(vs)))

def rest():
    b.teleport(0, cfg.start)
    for _ in range(8):
        b.step(np.zeros((1, 2)))

def per_slot(amp, n=16):
    rest()
    return np.array([b.step(np.array([[amp, 0.0]]))[1][0, 0] for _ in range(n)])

def one_command(amp, n=16):
    rest()
    f = c.moveByVelocityZAsync(amp, 0.0, b.z_alt, n * DT, dj_drive, b.yaw, vehicle_name=nm)
    out = []
    for _ in range(n):
        c.simContinueForTime(DT * b.cont_scale)
        while not c.simIsPause():
            time.sleep(0.0005)
        out.append(b.read()[1][0, 0])
    del f
    return np.array(out)

dj_drive = b.A.DrivetrainType.MaxDegreeOfFreedom
for name, vs, amp in [("per-slot commands, 2 m/s", per_slot(2.0), 2.0),
                      ("one 4 s command, 2 m/s", one_command(2.0), 2.0),
                      ("per-slot commands, 3 m/s", per_slot(3.0), 3.0)]:
    t, rms = fit(vs, amp)
    print(f"{name:28s} tau {t:.2f} s, fit RMS {rms:.3f} m/s | " + " ".join(f"{v:.2f}" for v in vs), flush=True)
rest()
b.close()

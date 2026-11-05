#!/usr/bin/env python3
"""
test_offline.py: tests drone_jcc.py end to end without Unreal Engine.

Surrogate: a short always_send training, joint_push and joint_pull from it, and an evaluation.
AirSim path: a fake AirSim server (fake_airsim_server.py, two drones) and the real command line of drone_jcc.py:
check, watch with the PD controller, a joint_push training on the two drones, and an evaluation. After each step it
asks the fake server what happened and checks that
  * SimpleFlight's 60 ms API goal timeout never fired during the lockstep slots (every command ended inside its
    slot and the next one started before the clock moved),
  * the server never ran out of worker threads (no waits for a free thread),
  * red and green markers were drawn, each drone's in its own arena (world frame = start offset + local position),
  * resets used simSetKinematics, and the run folders have their logs and checkpoints.

Usage: python test_offline.py [--python /path/to/python] [--keep]
  --python  the interpreter that runs drone_jcc.py for the AirSim part (e.g. one with the official airsim package)
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "drone_jcc.py")
sys.path.insert(0, HERE)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(py, args, root, extra_env=None, timeout=1800):
    cmd = [py, SCRIPT] + args + ["--set", f"RUN_ROOT={root!r}", "--set", "TENSORBOARD=False"]
    t0 = time.time()
    env = dict(os.environ, **(extra_env or {}))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-6:])
    print(f"$ {' '.join(os.path.basename(a) if a == SCRIPT else a for a in cmd[1:3])} ... "
          f"[{time.time() - t0:.0f}s, exit {p.returncode}]\n{tail}\n", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
    if not cond:
        raise SystemExit(f"check failed: {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    import drone_jcc as dj
    root = tempfile.mkdtemp(prefix="drone_jcc_test_")
    print(f"test folder {root}\n", flush=True)
    py = sys.executable
    small = ["--set", "N_ENVS=16", "--set", "ROLLOUT_SLOTS=128", "--set", "DASHBOARD='none'"]

    # ---- information structure: what each side may see ---------------------------------------------------------
    import numpy as np
    cfg = dj.make_cfg(["BACKEND='surrogate'"])
    env = dj.DroneTaskEnv(cfg, dj.SurrogateBackend(cfg, 4, [1, 2, 3, 4]), [1, 2, 3, 4])
    for _ in range(5):
        env.transmit(env.rule("periodic", 3))
        env.advance(np.random.default_rng(0).uniform(-1, 1, (4, 2)))
    c0, pull0, push0 = env.ctrl_obs(), env.sched_obs("pull"), env.sched_obs("push")
    env.pos = env.pos + 0.7
    env.vel = env.vel - 0.4
    check(np.array_equal(c0, env.ctrl_obs()), "the controller's inputs do not depend on the true state")
    check(np.array_equal(pull0, env.sched_obs("pull")), "the pull scheduler's inputs do not depend on the true state")
    check(not np.array_equal(push0, env.sched_obs("push")), "the push scheduler sees the drone's state")
    env.pos = env.pos - 0.7
    env.vel = env.vel + 0.4
    d = np.ones(4)
    seen = env.transmit(d)[:, -3:]
    check(np.all((seen[:, 2] == 1) == env.deliv), "a delivery shows in the controller's inputs")
    check(np.allclose(env.xh[env.deliv], env.pos[env.deliv]), "a delivery sets the estimate to the true state")

    # ---- surrogate ----------------------------------------------------------------------------------------------
    run(py, ["train", "--backend", "surrogate", "--phase", "always_send", "--run", "s_always", "--set",
             "TOTAL_SLOTS=60000"] + small, root)
    ck = os.path.join(root, "s_always", "ckpt_latest.pt")
    for ph in ("joint_push", "joint_pull"):
        run(py, ["train", "--backend", "surrogate", "--phase", ph, "--run", f"s_{ph}", "--init", ck, "--set",
                 "TOTAL_SLOTS=20000"] + small, root)
    for d in ("s_always", "s_joint_push", "s_joint_pull"):
        for f in ("ckpt_latest.pt", "train_log.csv", "tasks.csv", "trace.csv", "live.json", "config.json"):
            check(os.path.isfile(os.path.join(root, d, f)), f"{d}/{f} exists")
    run(py, ["eval", "--backend", "surrogate", "--ckpt", os.path.join(root, "s_joint_push", "ckpt_latest.pt"),
                   "--ckpt", os.path.join(root, "s_joint_pull", "ckpt_latest.pt"), "--out",
                   os.path.join(root, "s_eval"), "--set", "EVAL_SLOTS=2000", "--set", "EVAL_REPS=4"], root)
    ev = json.load(open(os.path.join(root, "s_eval", "eval.json")))
    check(len(ev["results"]) == 2 and all(r["J_ci"] == r["J_ci"] for r in ev["results"]), "surrogate eval with CIs")

    # ---- AirSim path, fake server ---------------------------------------------------------------------------------
    port = free_port()
    srv = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_airsim_server.py"), "--drones", "2", "--port",
                            str(port), "--clock", "20"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(1.5)
        air = ["--set", "NUM_DRONES=2", "--set", f"AIRSIM_PORT={port}", "--set", "DASHBOARD='none'"]
        cli = dj.LiteAirSimClient("127.0.0.1", port, 30)

        def stats():
            return cli.client.call("fakeStats")

        def timeouts():
            return sum(stats()["hover_timeouts"].values())

        run(a.python, ["check"] + air, root)
        s = stats()
        check(timeouts() == 0, f"no SimpleFlight goal timeouts in check (got {s['hover_timeouts']})")
        check(s["set_kinematics"] >= 2, "resets use simSetKinematics")
        red = sum(v for k, v in s["plots"].items() if k.startswith("points:1.0,0.1,0.1"))
        green = sum(v for k, v in s["plots"].items() if k.startswith("points:0.1,1.0,0.2"))
        check(red > 0 and green > 0, f"red and green markers drawn ({red} red, {green} green)")

        run(a.python, ["watch", "--run", "a_watch", "--slots", "400", "--set", "WATCH_REALTIME=False", "--set",
                       "CAMERA_STREAM=True", "--set", "CAMERA_PERIOD=0.2"] + air, root)
        s = stats()
        check(timeouts() == 0, f"no goal timeouts while flying 400 slots (got {s['hover_timeouts']})")
        check(s["max_queue_wait"] < 0.5, f"no wait for a free RPC thread (max {s['max_queue_wait']:.3f}s)")
        check(os.path.isfile(os.path.join(root, "a_watch", "camera.png")), "camera images saved")
        markers = cli.client.call("fakeMarkers")
        reds = [m for m in markers if m["kind"] == "points" and m["color"][:3] == [1.0, 0.1, 0.1]]
        ys = sorted(p[1] for m in reds for p in m["points"])
        check(len(ys) == 2 and ys[0] < 14.5 and 25.5 < ys[1] < 54.5,
              f"drone 1 and drone 2 targets drawn in their own arenas (y = {[round(y, 1) for y in ys]})")
        tasks = open(os.path.join(root, "a_watch", "tasks.csv")).read().splitlines()
        check(len(tasks) > 20, f"tasks completed in the fake simulator ({len(tasks) - 1} rows)")

        run(a.python, ["watch", "--run", "a_watch3d", "--slots", "200", "--set", "WATCH_REALTIME=False", "--set",
                       "DIM=3"] + air, root)
        s = stats()
        check(timeouts() == 0, f"no goal timeouts with 3-D targets (got {s['hover_timeouts']})")
        check(s["calls"].get("moveByVelocity", 0) >= 400, "3-D flight uses moveByVelocity")
        tasks = open(os.path.join(root, "a_watch3d", "tasks.csv")).read().splitlines()
        check(len(tasks) > 5, f"3-D tasks completed ({len(tasks) - 1} rows)")

        if a.python == sys.executable:  # training and evaluation need torch
            run(py, ["train", "--phase", "joint_push", "--run", "a_push", "--init", ck, "--set",
                     "ROLLOUT_SLOTS_AIRSIM=128", "--set", "TOTAL_SLOTS_AIRSIM=512"] + air, root)
            s = stats()
            check(timeouts() == 0, f"no goal timeouts while training (got {s['hover_timeouts']})")
            check(os.path.isfile(os.path.join(root, "a_push", "ckpt_latest.pt")), "AirSim training checkpoint")
            run(py, ["eval", "--ckpt", os.path.join(root, "a_push", "ckpt_latest.pt"), "--out",
                     os.path.join(root, "a_eval"), "--set", "EVAL_SLOTS_AIRSIM=150", "--set", "EVAL_REPS_AIRSIM=2"]
                + air, root)
            ev = json.load(open(os.path.join(root, "a_eval", "eval.json")))
            check(ev["results"][0]["reps"] == 4, "AirSim eval: 2 rounds x 2 drones")
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    print("\nall offline tests passed", flush=True)
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

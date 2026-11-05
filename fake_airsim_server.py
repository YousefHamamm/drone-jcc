#!/usr/bin/env python3
"""
fake_airsim_server.py: a stand-in for the AirSim RPC server, to run drone_jcc.py without Unreal Engine.

It speaks msgpack-rpc on port 41451 (like AirSim's rpclib server), answers the calls drone_jcc.py makes, and
simulates SimpleFlight multirotors closely enough to exercise the code paths:
  * a steppable clock: simPause, simIsPaused, simContinueForTime (blocks until the time has run; as in AirSim 1.8.1
    with the SteppableClock, simContinueForTime(x) moves the sim clock by ClockSpeed x x);
  * velocity and position goals with a first-order response, SimpleFlight's 60 ms API goal timeout (after which the
    drone holds its position), takeoff, hover, moveToPosition;
  * AirSim's task semantics: a new move cancels the running one and waits for it to stop, and a running command
    only notices the cancel when the clock moves; the server has (vehicles + 4) worker threads, as AirSim does;
  * positions relative to each vehicle's start point, markers in the world frame, collisions with the ground and
    with one pillar, wind, set kinematics / pose, camera images (a small rendered top view);
  * a fakeStats call that reports what happened (goal timeouts, markers by color, waits for a worker thread, ...).

Usage:  python fake_airsim_server.py [--drones 2] [--port 41451] [--clock 5] [--spacing 40]
"""
import argparse
import io
import json
import math
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import msgpack
import numpy as np

H = 0.005  # s of sim time per physics step
MOVES = {"moveByVelocityZ", "moveByVelocity", "moveToPosition", "moveToZ", "takeoff", "hover"}
_local = threading.local()
CMD_PERIOD = 0.02  # SimpleFlight command period (50 Hz)
GOAL_TIMEOUT = 0.06  # SimpleFlight api_goal_timeout
TAU = 0.40  # velocity response time constant
ACC = 7.0  # m/s^2


def vec(x=0.0, y=0.0, z=0.0):
    return {"x_val": float(x), "y_val": float(y), "z_val": float(z)}


def quat():
    return {"w_val": 1.0, "x_val": 0.0, "y_val": 0.0, "z_val": 0.0}


class Token:
    def __init__(self):
        self.cancelled = False


class Vehicle:
    def __init__(self, name, offset):
        self.name, self.offset = name, np.array(offset, float)
        self.cv = threading.Condition()
        self.task = None
        self.reset()

    def reset(self):
        self.p, self.v = np.zeros(3), np.zeros(3)
        self.api = self.armed = self.flying = False
        self.mode, self.goal_v, self.goal_z, self.goal_p, self.vmax = "none", np.zeros(3), None, np.zeros(3), 3.0
        self.goal_ts, self.timed_out = 0.0, False
        self.col = dict(has_collided=False, normal=vec(), impact_point=vec(), position=vec(), penetration_depth=0.0,
                        time_stamp=0, object_name="", object_id=-1)


class FakeAirSim:
    def __init__(self, n, clock, spacing, pillar=(12.0, -12.0, 1.0)):
        self.vehicles = {f"Drone{i + 1}": Vehicle(f"Drone{i + 1}", (0.0, i * spacing, 0.0)) for i in range(n)}
        self.clock_speed, self.spacing, self.pillar = float(clock), float(spacing), pillar
        self.t, self.paused = 0.0, False
        self.lock = threading.RLock()  # state
        self.step_lock = threading.Lock()  # physics steps
        self.tick = threading.Condition()
        self.wind = np.zeros(3)
        self.stats = dict(hover_timeouts={}, plots={}, persistent_markers=0, flushes=0, messages={}, set_kinematics=0,
                          set_pose=0, collisions=0, continues=0, max_busy=0, max_queue_wait=0.0, calls={},
                          wind_sets=0, images=0)
        self.busy = 0
        self.persistent = []
        self.waiters = {}
        self.pending = set()
        threading.Thread(target=self._free_run, daemon=True).start()

    # ---- physics -------------------------------------------------------------------------------------------------
    def _step(self):
        with self.step_lock, self.lock:
            a = 1.0 - math.exp(-H / TAU)
            for veh in self.vehicles.values():
                if veh.api and veh.flying and veh.mode == "vel" and self.t - veh.goal_ts > GOAL_TIMEOUT:
                    veh.mode, veh.goal_p, veh.vmax = "pos", veh.p.copy(), 3.0  # SimpleFlight: hover for safety
                    self.stats["hover_timeouts"][veh.name] = self.stats["hover_timeouts"].get(veh.name, 0) + 1
                if veh.mode == "vel":
                    vd = veh.goal_v.copy()
                    if veh.goal_z is not None:
                        vd[2] = float(np.clip(2.0 * (veh.goal_z - veh.p[2]), -3.0, 3.0))
                elif veh.mode == "pos":
                    e = veh.goal_p - veh.p
                    vd = 1.5 * e
                    nv = np.linalg.norm(vd)
                    if nv > veh.vmax:
                        vd *= veh.vmax / nv
                else:
                    vd = np.zeros(3) if veh.flying else -veh.v
                dv = (vd - veh.v) * a
                nd = np.linalg.norm(dv)
                if nd > ACC * H:
                    dv *= ACC * H / nd
                veh.v = veh.v + dv
                veh.p = veh.p + (veh.v + 0.25 * self.wind) * H
                if veh.p[2] > 0.0:  # the ground is at the start height
                    if veh.flying and veh.v[2] > 0.5:
                        self._collide(veh, "Ground")
                    veh.p[2], veh.v[2] = 0.0, min(veh.v[2], 0.0)
                px, py, pr = self.pillar
                d = math.hypot(veh.p[0] - px, veh.p[1] - py)
                if d < pr and veh.p[2] < 0.0:
                    k = pr / max(d, 1e-6)
                    veh.p[0], veh.p[1] = px + (veh.p[0] - px) * k, py + (veh.p[1] - py) * k
                    veh.v[:2] = 0.0
                    self._collide(veh, "Pillar")
            self.t += H

    def _advance(self):
        """One physics step, then let every command whose period is due run first (AirSim's C++ command loops
        spin on the clock and react within a physics step; Python threads need to be waited for)."""
        self._step()
        with self.tick:
            self.tick.notify_all()
            deadline = time.time() + 2.0
            while (self.pending or any(tg <= self.t + 1e-9 for tg in self.waiters.values())) \
                    and time.time() < deadline:
                self.tick.wait(0.002)

    def _collide(self, veh, what):
        veh.col = dict(has_collided=True, normal=vec(0, 0, -1), impact_point=vec(*veh.p), position=vec(*veh.p),
                       penetration_depth=0.01, time_stamp=int(self.t * 1e9), object_name=what, object_id=1)
        self.stats["collisions"] += 1

    def _free_run(self):
        last = time.time()
        while True:
            now = time.time()
            if self.paused:
                last = now
                time.sleep(0.001)
                continue
            n = int((now - last) * self.clock_speed / H)
            for _ in range(min(n, 200)):
                if self.paused:
                    break
                self._advance()
            if n:
                last = now
            time.sleep(0.001)

    def _sleep_sim(self, dt, tok):
        """Wait until the sim clock has moved by dt (a paused clock never moves, as in AirSim's SteppableClock)."""
        target = self.t + dt
        with self.tick:
            self.waiters[tok] = target
            while self.t < target - 1e-9:
                self.tick.wait(timeout=0.5)

    # ---- AirSim's task semantics ------------------------------------------------------------------------------------
    def _task(self, veh, body, duration):
        tok = Token()
        key = getattr(_local, "key", None)
        with veh.cv:
            if veh.task is not None:
                veh.task.cancelled = True
                while veh.task is not None:
                    veh.cv.wait()
            veh.task = tok
        try:
            t0 = self.t
            while True:
                if tok.cancelled:
                    return False
                with self.lock:
                    done = body()
                with self.tick:
                    self.waiters.pop(tok, None)
                    self.pending.discard(key)
                    self.tick.notify_all()
                if done:
                    return True
                if duration is not None and self.t - t0 >= duration - 1e-9:
                    return True
                self._sleep_sim(CMD_PERIOD, tok)  # the cancel is seen only once the clock moves
        finally:
            with self.tick:
                self.waiters.pop(tok, None)
                self.pending.discard(key)
                self.tick.notify_all()
            with veh.cv:
                veh.task = None
                veh.cv.notify_all()

    def _vel_goal(self, veh, vx, vy, vz=None, z=None):
        veh.mode, veh.goal_v, veh.goal_z, veh.goal_ts = "vel", np.array([vx, vy, vz or 0.0], float), z, self.t

    # ---- RPC methods -------------------------------------------------------------------------------------------------
    def ping(self):
        return True

    def getServerVersion(self):
        return 1

    def getMinRequiredClientVersion(self):
        return 1

    def listVehicles(self):
        return list(self.vehicles)

    def getSettingsString(self):
        return json.dumps({"SettingsVersion": 1.2, "SimMode": "Multirotor", "ClockSpeed": self.clock_speed,
                           "ClockType": "SteppableClock",
                           "Vehicles": {nm: {"VehicleType": "SimpleFlight", "X": float(v.offset[0]),
                                             "Y": float(v.offset[1]), "Z": float(v.offset[2])}
                                        for nm, v in self.vehicles.items()}})

    def _veh(self, name):
        if name == "" and len(self.vehicles) == 1:
            return next(iter(self.vehicles.values()))
        if name == "":
            return self.vehicles["Drone1"]
        if name not in self.vehicles:
            raise RuntimeError(f"Vehicle API for '{name}' is not available. This could either because this is "
                               "simulation-only API or this vehicle does not exist")
        return self.vehicles[name]

    def enableApiControl(self, is_enabled, name):
        self._veh(name).api = bool(is_enabled)

    def isApiControlEnabled(self, name):
        return self._veh(name).api

    def armDisarm(self, arm, name):
        self._veh(name).armed = bool(arm)
        return True

    def simPause(self, p):
        self.paused = bool(p)

    def simIsPaused(self):
        return self.paused

    def simContinueForTime(self, seconds):
        """AirSim 1.8.1 with the SteppableClock runs this for `seconds` of wall time, so the sim clock moves
        ClockSpeed x seconds. Here the steps run as fast as they can; only the sim time matters."""
        self.stats["continues"] += 1
        self.paused = True
        for _ in range(int(round(float(seconds) * self.clock_speed / H))):
            self._advance()
        self.paused = True

    def simContinueForFrames(self, frames):
        self.simContinueForTime(frames / 60.0)

    def reset(self):
        with self.lock:
            for v in self.vehicles.values():
                v.reset()

    def takeoff(self, timeout, name):
        veh = self._veh(name)
        with self.lock:
            veh.flying = True

        def body():
            veh.mode, veh.goal_p, veh.vmax = "pos", np.array([veh.p[0], veh.p[1], -3.0]), 2.0
            return abs(veh.p[2] + 3.0) < 0.1 and np.linalg.norm(veh.v) < 0.2
        return self._task(veh, body, timeout)

    def hover(self, name):
        veh = self._veh(name)
        hold = veh.p.copy()

        def body():
            veh.mode, veh.goal_p, veh.vmax = "pos", hold, 3.0
            return np.linalg.norm(veh.v) < 0.2
        return self._task(veh, body, 10.0)

    def moveToPosition(self, x, y, z, velocity, timeout, drivetrain, yaw_mode, lookahead, adaptive, name):
        veh = self._veh(name)
        goal = np.array([x, y, z], float)

        def body():
            veh.mode, veh.goal_p, veh.vmax = "pos", goal, float(velocity)
            return np.linalg.norm(veh.p - goal) < 0.3 and np.linalg.norm(veh.v) < 0.3
        return self._task(veh, body, min(float(timeout), 60.0))

    def moveToZ(self, z, velocity, timeout, yaw_mode, lookahead, adaptive, name):
        veh = self._veh(name)
        return self.moveToPosition(veh.p[0], veh.p[1], z, velocity, timeout, 0, yaw_mode, -1, 1, name)

    def moveByVelocityZ(self, vx, vy, z, duration, drivetrain, yaw_mode, name):
        veh = self._veh(name)
        if duration <= 0:
            return False

        def body():
            self._vel_goal(veh, vx, vy, z=float(z))
            return False
        return self._task(veh, body, float(duration))

    def moveByVelocity(self, vx, vy, vz, duration, drivetrain, yaw_mode, name):
        veh = self._veh(name)

        def body():
            self._vel_goal(veh, vx, vy, vz=float(vz))
            return False
        return self._task(veh, body, float(duration))

    def cancelLastTask(self, name):
        veh = self._veh(name)
        with veh.cv:
            if veh.task is not None:
                veh.task.cancelled = True

    def _kin(self, veh):
        return dict(position=vec(*veh.p), orientation=quat(), linear_velocity=vec(*veh.v), angular_velocity=vec(),
                    linear_acceleration=vec(), angular_acceleration=vec())

    def simGetGroundTruthKinematics(self, name):
        with self.lock:
            return self._kin(self._veh(name))

    def getMultirotorState(self, name):
        with self.lock:
            veh = self._veh(name)
            return dict(collision=dict(veh.col), kinematics_estimated=self._kin(veh),
                        gps_location=dict(latitude=47.64, longitude=-122.14, altitude=122.0),
                        timestamp=int(self.t * 1e9), landed_state=1 if veh.flying else 0,
                        rc_data=dict(timestamp=0, pitch=0.0, roll=0.0, throttle=0.0, yaw=0.0, switch1=0, switch2=0,
                                     switch3=0, switch4=0, switch5=0, switch6=0, switch7=0, switch8=0,
                                     is_initialized=False, is_valid=False),
                        ready=True, ready_message="", can_arm=True)

    def simGetCollisionInfo(self, name):
        with self.lock:
            return dict(self._veh(name).col)

    def simSetKinematics(self, state, ignore_collision, name):
        with self.lock:
            veh = self._veh(name)
            p, v = state["position"], state["linear_velocity"]
            veh.p = np.array([p["x_val"], p["y_val"], p["z_val"]], float)
            veh.v = np.array([v["x_val"], v["y_val"], v["z_val"]], float)
            veh.flying = veh.p[2] < -0.05 or veh.flying
            self.stats["set_kinematics"] += 1

    def simSetVehiclePose(self, pose, ignore_collision, name):
        with self.lock:
            veh = self._veh(name)
            p = pose["position"]
            veh.p = np.array([p["x_val"], p["y_val"], p["z_val"]], float)  # the velocity stays, as in AirSim
            self.stats["set_pose"] += 1

    def simGetVehiclePose(self, name):
        with self.lock:
            return dict(position=vec(*self._veh(name).p), orientation=quat())

    def simGetObjectPose(self, name):
        with self.lock:
            if name in self.vehicles:
                veh = self.vehicles[name]
                return dict(position=vec(*(veh.p + veh.offset)), orientation=quat())
            nan = float("nan")
            return dict(position=vec(nan, nan, nan), orientation={"w_val": nan, "x_val": nan, "y_val": nan,
                                                                  "z_val": nan})

    def simSetWind(self, wind):
        self.wind = np.array([wind["x_val"], wind["y_val"], wind["z_val"]], float)
        self.stats["wind_sets"] += 1

    def _plot(self, kind, n, color, persistent, points=None):
        key = f"{kind}:{','.join(f'{c:.1f}' for c in color[:3])}"
        self.stats["plots"][key] = self.stats["plots"].get(key, 0) + n
        if persistent:
            self.stats["persistent_markers"] += n
            if points is not None:
                self.persistent.append((kind, color, [(p["x_val"], p["y_val"]) for p in points]))

    def simPlotPoints(self, points, color, size, duration, persistent):
        self._plot("points", len(points), color, persistent, points)

    def simPlotLineStrip(self, points, color, thickness, duration, persistent):
        self._plot("strip", len(points), color, persistent, points)

    def simPlotLineList(self, points, color, thickness, duration, persistent):
        self._plot("list", len(points), color, persistent)

    def simPlotStrings(self, strings, positions, scale, color, duration):
        self._plot("strings", len(strings), color, False)

    def simFlushPersistentMarkers(self):
        self.stats["flushes"] += 1
        self.persistent = []

    def simPrintLogMessage(self, message, param, severity):
        self.stats["messages"][str(message)] = str(param)

    def simGetImages(self, requests, name, external):
        self.stats["images"] += 1
        veh = self._veh(name)
        return [dict(image_data_uint8=self._render(veh), image_data_float=[], camera_position=vec(*veh.p),
                     camera_orientation=quat(), time_stamp=int(self.t * 1e9), message="", pixels_as_float=False,
                     compress=True, width=240, height=240, image_type=0, camera_name=r.get("camera_name", "0"))
                for r in requests]

    def _render(self, veh):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3, 3), dpi=80)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_facecolor("#5a7d4f")
            w = veh.p + veh.offset
            for kind, color, pts in list(self.persistent):
                xs, ys = [p[1] for p in pts], [p[0] for p in pts]
                if kind == "points":
                    ax.scatter(xs, ys, color=color[:3], s=60)
                else:
                    ax.plot(xs, ys, color=color[:3], lw=1.5)
            ax.plot([w[1]], [w[0]], "k^", ms=10)
            ax.set_xlim(w[1] - 12, w[1] + 12)
            ax.set_ylim(w[0] - 12, w[0] + 12)
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            return buf.getvalue()
        except Exception:
            return b""

    def fakeMarkers(self):
        return [dict(kind=k, color=list(c), points=[list(p) for p in pts]) for k, c, pts in self.persistent]

    def fakeStats(self):
        with self.lock:
            out = json.loads(json.dumps(self.stats))
            out["sim_time"] = self.t
            out["positions"] = {nm: v.p.tolist() for nm, v in self.vehicles.items()}
            return out


def serve(sim, host, port, workers):
    pool = ThreadPoolExecutor(max_workers=workers)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f"fake AirSim on {host}:{port}: {list(sim.vehicles)}, ClockSpeed {sim.clock_speed}, {workers} RPC threads",
          flush=True)

    def handle(conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        unp = msgpack.Unpacker(raw=False, strict_map_key=False)
        wlock = threading.Lock()

        def run(msgid, method, params, t_sub):
            _local.key = (id(conn), msgid) if method in MOVES else None
            with sim.lock:
                sim.busy += 1
                sim.stats["max_busy"] = max(sim.stats["max_busy"], sim.busy)
                sim.stats["max_queue_wait"] = max(sim.stats["max_queue_wait"], time.time() - t_sub)
                sim.stats["calls"][method] = sim.stats["calls"].get(method, 0) + 1
            err, res = None, None
            try:
                fn = getattr(sim, method, None)
                if fn is None or method.startswith("_"):
                    err = f"rpclib: server could not find function '{method}' with argument count {len(params)}."
                else:
                    res = fn(*params)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                with sim.lock:
                    sim.busy -= 1
                if _local.key is not None:
                    with sim.tick:
                        sim.pending.discard(_local.key)
                        sim.tick.notify_all()
            data = msgpack.packb([1, msgid, err, res], use_bin_type=True)
            with wlock:
                try:
                    conn.sendall(data)
                except OSError:
                    pass

        try:
            while True:
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                unp.feed(chunk)
                for msg in unp:
                    if msg[0] == 0:
                        if msg[2] in MOVES:
                            with sim.tick:
                                sim.pending.add((id(conn), msg[1]))
                        pool.submit(run, msg[1], msg[2], list(msg[3]), time.time())
        except OSError:
            pass
        finally:
            conn.close()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drones", type=int, default=1)
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--clock", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=40.0)
    a = ap.parse_args()
    serve(FakeAirSim(a.drones, a.clock, a.spacing), a.host, a.port, a.drones + 4)

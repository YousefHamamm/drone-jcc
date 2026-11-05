#!/usr/bin/env python3
"""
drone_jcc.py: joint communication and control for a drone that hits targets, in AirSim.

The model is the paper's, with a drone as the robot:
  * The drone flies to the current target, a red marker. A task ends with a success when the drone is within
    HIT_RADIUS of the target at the end of a slot: the marker turns green and a new red marker appears. It ends with
    a failure when the drone leaves the safe box, collides, or needs more than MAX_TASK_SLOTS slots.
  * After a success the drone is held for HOLD_SLOTS slots and the next task starts where it is. After a failure it
    is brought back to START_POS and held for RESET_SLOTS slots.
  * In each slot of a task the drone's state (position and velocity) can be sent once. A transmission costs COST_C
    and is delivered with probability P_S. Both sides learn the outcome (silent, lost, delivered). The controller
    then sends a velocity command, which always arrives.
  * Push: the drone decides when to send, from its own state and the common record.
    Pull: the controller decides when to ask, from its own record.
  * Throughput J = (successes - COST_C * transmissions) / slots, held and reset slots included.

Training is PPO with two heads that see only what they are allowed to see: the scheduler (on the drone for push,
on the controller for pull) and the controller (its record after the outcome of the slot). The critic is used only
in training and sees everything.
  always_send : the scheduler sends in every task slot; the controller is trained.
  joint_push  : the drone's scheduler and the controller are trained together, starting from always_send.
  joint_pull  : the controller's scheduler and the controller are trained together, starting from always_send.

Commands (python drone_jcc.py <command> -h shows the options):
  all        the three phases, then the evaluation and the plots, in one session folder
  train      one phase
  eval       compare policies on common random numbers, with 95% confidence intervals
  watch      fly a trained policy (or the PD controller) and watch it
  dashboard  live dashboard of a run (web page or window); train, all and watch start it themselves
  settings   print or write the AirSim settings.json that matches these parameters
  check      AirSim preflight: connection, lockstep timing, velocity response time, red and green markers
  plot       learning curves and the evaluation chart
  selftest   a few minutes of everything on the surrogate, to check the installation

Requirements: numpy, torch, tqdm, matplotlib. For AirSim: the `airsim` package, or only `msgpack`
(a built-in client is used when the official one does not import, e.g. on Python 3.10 and later).
"""

# ======================================================================================================================
# PARAMETERS. Edit them here, or override any of them from the command line with --set NAME=VALUE (repeatable), e.g.
#     python drone_jcc.py train --phase joint_push --set P_S=0.3 --set COST_C=0.02
# ======================================================================================================================

# ---- what to run -------------------------------------------------------------------------------------------------------
PHASE = "always_send"            # for `train`: always_send | joint_push | joint_pull
BACKEND = "airsim"               # airsim | surrogate (a fast Python model of the same drone and task)
RUN_ROOT = "runs"                # every run or session gets its own folder in here
RUN_NAME = ""                    # "" -> automatic name with the date and time
SEED = 1
DEVICE = "cpu"                   # cpu | cuda | auto (the networks are small; cpu is usually the fastest)
TORCH_THREADS = 4

# ---- AirSim connection and stepping ------------------------------------------------------------------------------
AIRSIM_IP = "127.0.0.1"
AIRSIM_PORT = 41451
AIRSIM_CLIENT = "auto"           # auto | official | lite (built-in client; needs only `pip install msgpack`)
NUM_DRONES = 1                   # drones flown in parallel, each in its own arena; must match settings.json
VEHICLE_PREFIX = "Drone"         # vehicle names Drone1, Drone2, ... (with one drone, a single vehicle of any name works)
ARENA_SPACING = 40.0             # m between the drones' start points, along y (used by `settings`)
ALTITUDE = 6.0                   # m above the start point: the plane of the targets
STEP_MODE = "lockstep"           # lockstep: paused simulator advanced DT per slot | realtime: commands run in real time
CMD_MARGIN = 0.03                # s; a command lasts DT - CMD_MARGIN; keep it below 0.06 (SimpleFlight's API timeout)
CMD_SETTLE = 0.001               # s (wall) between sending the commands and advancing the simulator
CLOCK_SPEED = 5.0                # settings.json ClockSpeed (sim seconds per wall second); too high makes physics unstable
CONTINUE_SCALE = "auto"          # lockstep: each slot calls simContinueForTime(DT x scale); auto: 1 / ClockSpeed with
                                 # the SteppableClock (AirSim 1.8.1 runs the call for that long in wall time), else 1
VIEW_MODE = "SpringArmChase"     # settings.json ViewMode: SpringArmChase | FlyWithMe | Manual | GroundObserver | NoDisplay
TAKEOFF_TIMEOUT = 20.0           # s
RPC_TIMEOUT = 60.0               # s for one AirSim call
USE_SET_KINEMATICS = True        # resets use simSetKinematics (zero velocity); else simSetVehiclePose

# ---- slot, drone and task ------------------------------------------------------------------------------------------
DT = 0.25                        # s per slot
DIM = 2                          # 2: targets in a horizontal plane at ALTITUDE | 3: targets also vary in altitude
U_MAX = 3.0                      # m/s, bound on each axis of the velocity command
U_NORM_CAP = True                # also cap the norm of the command at U_MAX
NOISE_STD = 0.9                  # m/s per axis: i.i.d. disturbance added to the command in each task slot (the paper's w_n)
WIND_STD = 0.0                   # m/s: Ornstein-Uhlenbeck wind (AirSim: simSetWind; surrogate: a drift); 0 = off
WIND_TAU = 3.0                   # s, correlation time of the wind
HIT_RADIUS = 0.5                 # m: success when |x - g| <= HIT_RADIUS at the end of a slot (the target window)
HIT_MAX_SPEED = 0.0              # m/s: > 0 also requires |v| <= HIT_MAX_SPEED at the hit; 0 = no condition
TARGET_MODE = "random"           # random | cycle (go through TARGET_LIST in turn, like the paper's alternating targets)
TARGET_HALF = 10.0               # m: random targets lie in the square |x|, |y| <= TARGET_HALF around START_POS
TARGET_MIN_DIST = 4.0            # m: distance of a new random target from the previous one, min and max
TARGET_MAX_DIST = 12.0
TARGET_LIST = ((-5.0, 0.0), (5.0, 0.0))   # m, relative to START_POS (cycle mode); a third entry is an altitude
Z_RANGE = (4.0, 10.0)            # m above the start point: altitude range of the targets when DIM = 3
SAFE_HALF = 14.0                 # m: safe box |x|, |y| <= SAFE_HALF around START_POS; leaving it is a failure
SAFE_ALT = (1.5, 15.0)           # m above the start point: safe altitude band when DIM = 3
START_POS = (0.0, 0.0)           # m: the reset point x_0, relative to each drone's own start point
HOLD_SLOTS = 2                   # Delta_S: slots held after a success
RESET_SLOTS = 20                 # Delta_F: slots to bring the drone back after a failure
MAX_TASK_SLOTS = 120             # a task that takes longer is a failure (0 = no limit)
COLLISION_IS_FAILURE = True
PHYSICAL_PENALTY = True          # True: held and reset slots are flown (hover); False: they are only counted
TAU = 1.55                       # s: velocity response time constant of the surrogate drone (SimpleFlight in Blocks
                                 # 1.8.1, measured with `check`; it was 0.35 before that measurement)
ACC_MAX = 8.0                    # m/s^2: acceleration limit of the surrogate drone
PRED_TAU = 1.55                  # s: time constant the controller uses to predict (`check` measures AirSim's)

# ---- channel and information ---------------------------------------------------------------------------------------
P_S = 0.6                        # delivery probability of a transmission
COST_C = 0.01                    # cost of a transmission, in units of one completed task
LOSS_VISIBLE = True              # the controller tells a lost transmission from silence (the paper's assumption)
SENDER_KNOWS_TARGET = True       # the push scheduler sees the target (paper: both sides know the targets)

# ---- learning (PPO) -------------------------------------------------------------------------------------------------
N_ENVS = 64                      # surrogate environments in parallel (AirSim uses NUM_DRONES)
ROLLOUT_SLOTS = 128              # slots per environment per update (surrogate)
ROLLOUT_SLOTS_AIRSIM = 512       # slots per drone per update (AirSim)
TOTAL_SLOTS = 4_000_000          # slots per phase, all environments together (surrogate)
TOTAL_SLOTS_AIRSIM = 300_000     # slots per phase, all drones together (AirSim)
GAMMA = 0.99                     # discount per slot
GAE_LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 5
MINIBATCHES = 4
LR = 3e-4
LR_SCHEDULE = "kl_adaptive"      # kl_adaptive | linear | constant
LR_MIN = 2e-5
LR_MAX = 1e-3
TARGET_KL = 0.02                 # for kl_adaptive, and epochs stop early above 2.5 x TARGET_KL
ENT_CTRL = 0.0                   # entropy bonus of the controller (Gaussian)
ENT_SCHED = 0.003                # entropy bonus of the scheduler (Bernoulli)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN = (128, 128)
INIT_LOG_STD = -0.7              # initial log std of the command, in units of U_MAX
SCHED_INIT_SEND_PROB = 0.9       # a new scheduler starts close to always sending
OBS_NORM = True                  # running mean/std normalization of the inputs
PRICE_MODE = "adaptive"          # adaptive: Dinkelbach price = running throughput | fixed: PRICE_INIT | none
PRICE_INIT = 0.0                 # price per slot (the joint phases start from the price of their init checkpoint)
PRICE_RATE = 0.05                # adaptive price step per update
SHAPING = 0.05                   # potential-based shaping: SHAPING x (gamma^k Phi' - Phi), Phi = -distance / (U_MAX DT)
INIT_FROM = "auto"               # joint phases start from: auto (newest always_send run) | a checkpoint path | none
CKPT_FOR_INIT = "latest"         # latest | best: which always_send checkpoint the joint phases and the evaluation use

# ---- evaluation ------------------------------------------------------------------------------------------------------
EVAL_SLOTS = 20_000              # slots per replication (surrogate)
EVAL_REPS = 16                   # replications; every policy sees the same random numbers (surrogate)
EVAL_SLOTS_AIRSIM = 3_000        # slots per replication and drone (AirSim)
EVAL_REPS_AIRSIM = 3
EVAL_SEED = 5000
EVAL_DETERMINISTIC = True        # mean command and send iff probability > 0.5
PERIODS = (2, 3, 4)              # baseline: send every K slots
ERR_THRESHOLDS = (0.3, 0.6)      # baseline (push): send when |x - x_hat| > threshold (m)
NEAR_RADII = (1.5, 3.0)          # baseline (push): send when within this distance of the target (m)
AGE_THRESHOLDS = (2, 3, 4)       # baseline (pull): ask when the last delivered state is this many slots old
PD_KP = 2.0                      # PD controller on the controller's estimate (a classical reference)
PD_KD = 0.4

# ---- tracking and live watching ------------------------------------------------------------------------------------
LOG_EVERY = 5                    # updates between printed summaries
CKPT_EVERY = 25                  # updates between numbered checkpoints (latest and best are always kept)
LIVE_PERIOD = 0.5                # s (wall) between writes of live.json, read by the dashboard
LIVE_TRAJ = 300                  # slots of drone-1 trajectory shown on the dashboard map
TRACE = True                     # trace.csv: every slot of drone 1 (state, estimate, decision, outcome, command)
TRACE_MAX_ROWS = 2_000_000
TENSORBOARD = True               # scalars to <run>/tb when tensorboard is installed
DASHBOARD = "web"                # web (http://127.0.0.1:PORT) | window (matplotlib) | none
DASHBOARD_PORT = 8765
DASHBOARD_HOST = "127.0.0.1"     # use SSH port forwarding to watch a remote run
DASHBOARD_REFRESH = 1.0          # s
DASHBOARD_KEEP = False           # keep the dashboard running after the command ends
OPEN_BROWSER = True
DRAW = True                      # AirSim markers: red current target, green hit targets, safe box, trail
DRAW_ALL_DRONES = True           # markers for every drone (the trail and the estimate are drawn for drone 1)
TRAIL_SECONDS = 6.0              # sim seconds a trail point stays
SHOW_HITS = 8                    # green markers kept per drone
SHOW_ESTIMATE = True             # yellow dot: where the controller thinks drone 1 is
ONSCREEN_STATS = True            # live numbers in the AirSim window
STATS_PERIOD = 1.0               # s (wall) between on-screen updates
CAMERA_STREAM = False            # save drone-1 camera images for the dashboard (AirSim)
CAMERA_NAME = "bottom_center"
CAMERA_PERIOD = 1.0              # s (wall) between camera images
WATCH_REALTIME = True            # watch: pace the slots to DT of wall time

# ======================================================================================================================
# END OF PARAMETERS
# ======================================================================================================================
_PARAMS = {k: v for k, v in dict(globals()).items() if k.isupper()}

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from collections import defaultdict, deque  # noqa: E402

import numpy as np  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # a small stand-in, so that the script still runs
    class tqdm:  # noqa: N801
        def __init__(self, total=None, desc="", unit="it", **kw):
            self.n, self.total, self.desc, self.post, self._t = 0, total, desc, "", 0.0

        def update(self, k=1):
            self.n += k
            if time.time() - self._t > 10:
                self._t = time.time()
                print(f"{self.desc}: {self.n}/{self.total} {self.post}", flush=True)

        def set_postfix(self, d=None, **kw):
            self.post = " ".join(f"{a}={b}" for a, b in dict(d or {}, **kw).items())

        def close(self):
            pass

        @staticmethod
        def write(s):
            print(s, flush=True)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = nn = F = None

_VERSION = "1.0"
_SCRIPT = os.path.abspath(__file__)
_TASK, _HOLD, _RESET = 0, 1, 2
_MODES = ("task", "hold", "reset")
_SILENT, _LOST, _DELIV, _NONE = 0, 1, 2, 3
_FAIL_NAMES = {1: "out_of_bounds", 2: "timeout", 3: "collision"}
_AGE_CAP = 20
_HIST = 4
_PHASE_SCHED = {"always_send": "always", "joint_push": "push", "joint_pull": "pull"}
_RULES = ("always", "never", "periodic", "age", "err", "near", "near_est")
_PROBE_COLS = ("dist", "err", "dist_est", "age", "speed", "task_slot", "send", "p")


# ======================================================================================================================
# Configuration
# ======================================================================================================================
class Cfg(dict):
    """The parameters above (UPPER CASE) plus derived values (lower case), with attribute access."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


def _parse_value(text, default):
    try:
        v = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        v = text
    if isinstance(default, bool):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(default, tuple) and isinstance(v, list):
        return tuple(v)
    return v


def make_cfg(sets=(), base=None, **direct):
    """Parameters from the block above (or from `base`), then --set NAME=VALUE items, then keyword overrides."""
    src = base if base is not None else _PARAMS
    cfg = Cfg({k: copy.deepcopy(v) for k, v in src.items() if k.isupper()})
    for item in sets or ():
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip().upper()
        if k not in _PARAMS:
            raise SystemExit(f"unknown parameter {k!r} (see the PARAMETERS block at the top of {_SCRIPT})")
        cfg[k] = _parse_value(v.strip(), _PARAMS[k])
    for k, v in direct.items():
        if v is not None:
            if k not in _PARAMS:
                raise SystemExit(f"unknown parameter {k!r}")
            cfg[k] = v
    return derive(cfg)


def derive(cfg):
    D = int(cfg.DIM)
    if D not in (2, 3):
        raise SystemExit("DIM must be 2 or 3")
    air = cfg.BACKEND == "airsim"
    if cfg.BACKEND not in ("airsim", "surrogate"):
        raise SystemExit("BACKEND must be airsim or surrogate")
    sx, sy = float(cfg.START_POS[0]), float(cfg.START_POS[1])
    cfg.start = np.array([sx, sy] + ([-float(cfg.ALTITUDE)] if D == 3 else []))
    h = float(cfg.SAFE_HALF)
    lo, hi = [sx - h, sy - h], [sx + h, sy + h]
    if D == 3:
        lo.append(-float(cfg.SAFE_ALT[1]))
        hi.append(-float(cfg.SAFE_ALT[0]))
    cfg.safe_lo, cfg.safe_hi = np.array(lo), np.array(hi)
    cfg.n_envs = int(cfg.NUM_DRONES) if air else int(cfg.N_ENVS)
    cfg.rollout = int(cfg.ROLLOUT_SLOTS_AIRSIM if air else cfg.ROLLOUT_SLOTS)
    cfg.total = int(cfg.TOTAL_SLOTS_AIRSIM if air else cfg.TOTAL_SLOTS)
    cfg.eval_slots = int(cfg.EVAL_SLOTS_AIRSIM if air else cfg.EVAL_SLOTS)
    cfg.eval_reps = int(cfg.EVAL_REPS_AIRSIM if air else cfg.EVAL_REPS)
    cfg.cmd_duration = max(0.02, float(cfg.DT) - float(cfg.CMD_MARGIN))
    problems = []
    if not 0.0 < float(cfg.P_S) <= 1.0:
        problems.append("P_S must be in (0, 1]")
    if float(cfg.TARGET_HALF) + float(cfg.HIT_RADIUS) >= h:
        problems.append("TARGET_HALF + HIT_RADIUS must be smaller than SAFE_HALF")
    if cfg.TARGET_MODE not in ("random", "cycle"):
        problems.append("TARGET_MODE must be random or cycle")
    if cfg.TARGET_MODE == "random" and float(cfg.TARGET_MIN_DIST) > float(cfg.TARGET_MAX_DIST):
        problems.append("TARGET_MIN_DIST must not exceed TARGET_MAX_DIST")
    if cfg.STEP_MODE not in ("lockstep", "realtime"):
        problems.append("STEP_MODE must be lockstep or realtime")
    if problems:
        raise SystemExit("parameter error: " + "; ".join(problems))
    if air and cfg.STEP_MODE == "lockstep" and not 0.0 < float(cfg.CMD_MARGIN) < 0.06:
        print("[warning] CMD_MARGIN should be in (0, 0.06): SimpleFlight hovers 60 ms after a command ends", flush=True)
    return cfg


def cfg_dump(cfg):
    return {k: _jsonable(v) for k, v in cfg.items() if k.isupper()}


# ======================================================================================================================
# Small utilities
# ======================================================================================================================
def _now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, deque)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path, obj, indent=None):
    """Atomic write; retries because on Windows a reader can briefly block the rename."""
    text = json.dumps(_jsonable(obj), indent=indent)
    tmp = f"{path}.{os.getpid()}.tmp"
    for attempt in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.02 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def write_bytes(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    for attempt in range(5):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.02 * (attempt + 1))
    return False


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t95(df):
    """Two-sided 95% Student t quantile."""
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df))
    except Exception:
        keys = sorted(_T95)
        for k in keys:
            if df <= k:
                return _T95[k]
        return 1.96


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(t95(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def _csv_open(path, header):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(header)
    return f, w


def _free_port(host, port):
    for p in range(int(port), int(port) + 50):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    return int(port)


def seed_all(seed):
    np.random.seed(seed % (2 ** 32))
    if torch is not None:
        torch.manual_seed(seed)


def _need_torch():
    if torch is None:
        raise SystemExit("this command needs PyTorch: pip install torch")


def _device(cfg):
    if cfg.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.DEVICE


# ======================================================================================================================
# AirSim client: the official `airsim` package if it imports, else a small built-in msgpack-rpc client with the same
# method names and arguments (only the calls used here)
# ======================================================================================================================
class _LMsg:
    def to_msgpack(self, *args, **kwargs):
        return self.__dict__


class _LVector3r(_LMsg):
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0):
        self.x_val, self.y_val, self.z_val = float(x_val), float(y_val), float(z_val)


class _LQuaternionr(_LMsg):
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val, self.y_val, self.z_val, self.w_val = float(x_val), float(y_val), float(z_val), float(w_val)


class _LPose(_LMsg):
    def __init__(self, position_val=None, orientation_val=None):
        self.position = position_val if position_val is not None else _LVector3r()
        self.orientation = orientation_val if orientation_val is not None else _LQuaternionr()


class _LKinematicsState(_LMsg):
    def __init__(self):
        self.position, self.orientation = _LVector3r(), _LQuaternionr()
        self.linear_velocity, self.angular_velocity = _LVector3r(), _LVector3r()
        self.linear_acceleration, self.angular_acceleration = _LVector3r(), _LVector3r()


class _LYawMode(_LMsg):
    def __init__(self, is_rate=True, yaw_or_rate=0.0):
        self.is_rate, self.yaw_or_rate = bool(is_rate), float(yaw_or_rate)


class _LImageRequest(_LMsg):
    def __init__(self, camera_name, image_type, pixels_as_float=False, compress=True):
        self.camera_name, self.image_type = str(camera_name), int(image_type)
        self.pixels_as_float, self.compress = bool(pixels_as_float), bool(compress)


LITE_TYPES = types.SimpleNamespace(
    Vector3r=_LVector3r, Quaternionr=_LQuaternionr, Pose=_LPose, KinematicsState=_LKinematicsState,
    YawMode=_LYawMode, ImageRequest=_LImageRequest,
    DrivetrainType=types.SimpleNamespace(MaxDegreeOfFreedom=0, ForwardOnly=1), ImageType=types.SimpleNamespace(Scene=0))


class _Obj:
    """Attribute access to an AirSim reply (a dict from msgpack)."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _wrap(v))

    def __repr__(self):
        return f"_Obj({self.__dict__})"


def _wrap(v):
    if isinstance(v, dict):
        return _Obj(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _to_msgpack(o):
    if hasattr(o, "to_msgpack"):
        return o.to_msgpack()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot send {type(o).__name__} to AirSim")


class _LiteFuture:
    def __init__(self, rpc, msgid):
        self._rpc, self._id = rpc, msgid
        rpc._wanted.add(msgid)

    def join(self):
        return self._rpc._wait(self._id)

    get = join

    def __del__(self):  # a future that is dropped without join: its reply is discarded when it comes
        try:
            self._rpc._wanted.discard(self._id)
            self._rpc._results.pop(self._id, None)
        except Exception:
            pass


class LiteRpc:
    """msgpack-rpc over TCP, as spoken by AirSim's rpclib server: request [0, id, method, params],
    reply [1, id, error, result]. Single-threaded: replies are read while some call waits for its own."""

    def __init__(self, ip, port, timeout):
        import msgpack
        self._mp = msgpack
        self._sock = socket.create_connection((ip, int(port)), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._unp = msgpack.Unpacker(raw=False, strict_map_key=False)
        self._id, self._results, self._wanted, self._warned = 0, {}, set(), 0
        self._lock = threading.RLock()

    def _send(self, method, args):
        with self._lock:
            self._id += 1
            self._sock.sendall(self._mp.packb([0, self._id, method, list(args)], default=_to_msgpack,
                                              use_bin_type=False))
            return self._id

    def call(self, method, *args):
        mid = self._send(method, args)
        self._wanted.add(mid)
        return self._wait(mid)

    def call_async(self, method, *args):
        return _LiteFuture(self, self._send(method, args))

    def _wait(self, mid):
        with self._lock:
            while mid not in self._results:
                chunk = self._sock.recv(1 << 16)
                if not chunk:
                    raise ConnectionError("AirSim closed the connection")
                self._unp.feed(chunk)
                for msg in self._unp:
                    if not isinstance(msg, (list, tuple)) or len(msg) != 4 or msg[0] != 1:
                        continue
                    _, rid, err, res = msg
                    if rid in self._wanted:
                        self._results[rid] = (err, res)
                    elif err is not None and self._warned < 20:
                        self._warned += 1
                        print(f"[airsim] an asynchronous call failed: {err}", flush=True)
            self._wanted.discard(mid)
            err, res = self._results.pop(mid)
        if err is not None:
            raise RuntimeError(f"AirSim RPC error: {err}")
        return res


class LiteAirSimClient:
    """The subset of airsim.MultirotorClient used here, with the same names and arguments."""

    def __init__(self, ip="", port=41451, timeout_value=3600):
        self.client = LiteRpc(ip or "127.0.0.1", port, timeout_value)

    def ping(self):
        return self.client.call("ping")

    def getServerVersion(self):
        return self.client.call("getServerVersion")

    def confirmConnection(self):
        print("Connected!" if self.ping() else "Ping returned false!", flush=True)

    def getSettingsString(self):
        return self.client.call("getSettingsString")

    def listVehicles(self):
        return self.client.call("listVehicles")

    def reset(self):
        self.client.call("reset")

    def enableApiControl(self, is_enabled, vehicle_name=""):
        self.client.call("enableApiControl", bool(is_enabled), vehicle_name)

    def armDisarm(self, arm, vehicle_name=""):
        return self.client.call("armDisarm", bool(arm), vehicle_name)

    def simPause(self, is_paused):
        self.client.call("simPause", bool(is_paused))

    def simIsPause(self):
        return self.client.call("simIsPaused")

    def simContinueForTime(self, seconds):
        self.client.call("simContinueForTime", float(seconds))

    def takeoffAsync(self, timeout_sec=20, vehicle_name=""):
        return self.client.call_async("takeoff", timeout_sec, vehicle_name)

    def hoverAsync(self, vehicle_name=""):
        return self.client.call_async("hover", vehicle_name)

    def moveToPositionAsync(self, x, y, z, velocity, timeout_sec=3e+38, drivetrain=0, yaw_mode=None, lookahead=-1,
                            adaptive_lookahead=1, vehicle_name=""):
        return self.client.call_async("moveToPosition", x, y, z, velocity, timeout_sec, drivetrain,
                                      yaw_mode if yaw_mode is not None else _LYawMode(), lookahead,
                                      adaptive_lookahead, vehicle_name)

    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain=0, yaw_mode=None, vehicle_name=""):
        return self.client.call_async("moveByVelocityZ", vx, vy, z, duration, drivetrain,
                                      yaw_mode if yaw_mode is not None else _LYawMode(), vehicle_name)

    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain=0, yaw_mode=None, vehicle_name=""):
        return self.client.call_async("moveByVelocity", vx, vy, vz, duration, drivetrain,
                                      yaw_mode if yaw_mode is not None else _LYawMode(), vehicle_name)

    def getMultirotorState(self, vehicle_name=""):
        return _wrap(self.client.call("getMultirotorState", vehicle_name))

    def simGetGroundTruthKinematics(self, vehicle_name=""):
        return _wrap(self.client.call("simGetGroundTruthKinematics", vehicle_name))

    def simGetCollisionInfo(self, vehicle_name=""):
        return _wrap(self.client.call("simGetCollisionInfo", vehicle_name))

    def simSetKinematics(self, state, ignore_collision, vehicle_name=""):
        self.client.call("simSetKinematics", state, bool(ignore_collision), vehicle_name)

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=""):
        self.client.call("simSetVehiclePose", pose, bool(ignore_collision), vehicle_name)

    def simGetVehiclePose(self, vehicle_name=""):
        return _wrap(self.client.call("simGetVehiclePose", vehicle_name))

    def simGetObjectPose(self, object_name):
        return _wrap(self.client.call("simGetObjectPose", object_name))

    def simSetWind(self, wind):
        self.client.call("simSetWind", wind)

    def simFlushPersistentMarkers(self):
        self.client.call("simFlushPersistentMarkers")

    def simPlotPoints(self, points, color_rgba=(1.0, 0.0, 0.0, 1.0), size=10.0, duration=-1.0, is_persistent=False):
        self.client.call("simPlotPoints", points, list(color_rgba), float(size), float(duration), bool(is_persistent))

    def simPlotLineStrip(self, points, color_rgba=(1.0, 0.0, 0.0, 1.0), thickness=5.0, duration=-1.0,
                         is_persistent=False):
        self.client.call("simPlotLineStrip", points, list(color_rgba), float(thickness), float(duration),
                         bool(is_persistent))

    def simPrintLogMessage(self, message, message_param="", severity=0):
        self.client.call("simPrintLogMessage", message, message_param, severity)

    def simGetImages(self, requests, vehicle_name="", external=False):
        return [_wrap(r) for r in self.client.call("simGetImages", requests, vehicle_name, external)]


def connect_airsim(cfg):
    """Returns (types module, client, kind)."""
    if cfg.AIRSIM_CLIENT in ("auto", "official"):
        try:
            import airsim as A
            return A, A.MultirotorClient(ip=cfg.AIRSIM_IP, port=int(cfg.AIRSIM_PORT),
                                         timeout_value=float(cfg.RPC_TIMEOUT)), "official"
        except Exception as e:  # e.g. msgpack-rpc-python and tornado 4 do not import on Python 3.10+
            if cfg.AIRSIM_CLIENT == "official":
                raise
            print(f"[airsim] the official client does not import ({type(e).__name__}: {e}); using the built-in one",
                  flush=True)
    try:
        return LITE_TYPES, LiteAirSimClient(cfg.AIRSIM_IP, int(cfg.AIRSIM_PORT), float(cfg.RPC_TIMEOUT)), "lite"
    except ImportError:
        raise SystemExit("the built-in AirSim client needs msgpack: pip install msgpack") from None
    except OSError as e:
        raise SystemExit(f"cannot reach AirSim at {cfg.AIRSIM_IP}:{cfg.AIRSIM_PORT} ({e}). "
                         "Is the simulator running (press Play in the editor or start the binary)?") from None


def _kin_state(A, x, y, z):
    ks = A.KinematicsState()  # set every field: the official class sends only the attributes set on the instance
    ks.position = A.Vector3r(x, y, z)
    ks.orientation = A.Quaternionr(0.0, 0.0, 0.0, 1.0)
    ks.linear_velocity, ks.angular_velocity = A.Vector3r(0.0, 0.0, 0.0), A.Vector3r(0.0, 0.0, 0.0)
    ks.linear_acceleration, ks.angular_acceleration = A.Vector3r(0.0, 0.0, 0.0), A.Vector3r(0.0, 0.0, 0.0)
    return ks


# ======================================================================================================================
# Physics backends: AirSim, and a fast surrogate of a velocity-controlled multirotor
# ======================================================================================================================
class SurrogateBackend:
    """First-order velocity response (TAU) with an acceleration limit (ACC_MAX), 4 substeps per slot.
    Altitude is held in 2-D, modeled like the other axes in 3-D. No obstacles."""

    def __init__(self, cfg, n, seeds):
        self.cfg, self.n, self.D = cfg, int(n), int(cfg.DIM)
        self.pos, self.vel = np.zeros((self.n, self.D)), np.zeros((self.n, self.D))
        self.wind = np.zeros((self.n, self.D))
        self.rng = np.random.default_rng([int(seeds[0]), 99, self.n])
        self.offsets = np.zeros((self.n, 3))
        self.z_alt = -float(cfg.ALTITUDE)

    def reset(self, start):
        self.pos[:] = start
        self.vel[:] = 0.0
        return self.pos.copy(), self.vel.copy()

    def step(self, cmd, held=None):
        cfg = self.cfg
        dt = float(cfg.DT)
        drift = 0.0
        if cfg.WIND_STD > 0:
            a = math.exp(-dt / float(cfg.WIND_TAU))
            self.wind[:, :2] = a * self.wind[:, :2] + float(cfg.WIND_STD) * math.sqrt(1 - a * a) * \
                self.rng.standard_normal((self.n, 2))
            drift = 0.5 * self.wind
        h = dt / 4.0
        a = math.exp(-h / float(cfg.TAU))
        amax = float(cfg.ACC_MAX) * h
        target = cmd + drift
        for _ in range(4):
            dv = (target - self.vel) * (1.0 - a)
            nrm = np.linalg.norm(dv, axis=1, keepdims=True)
            dv *= np.minimum(1.0, amax / np.maximum(nrm, 1e-12))
            v_new = self.vel + dv
            self.pos += 0.5 * (self.vel + v_new) * h
            self.vel = v_new
        return self.pos.copy(), self.vel.copy(), np.zeros(self.n, bool)

    def teleport(self, i, p):
        self.pos[i] = p
        self.vel[i] = 0.0

    def pause(self, flag):
        pass

    def close(self):
        pass


class AirSimBackend:
    """SimpleFlight multirotors driven by velocity commands. In lockstep mode the simulator stays paused; each slot
    sends one command per drone (lasting DT - CMD_MARGIN, so that every command ends inside its slot and the server
    never holds more than one thread per drone) and then advances the simulator by DT with simContinueForTime."""

    def __init__(self, cfg, n=None):
        self.cfg = cfg
        self.n = int(n or cfg.NUM_DRONES)
        self.A, self.c, self.client_kind = connect_airsim(cfg)
        self.D = int(cfg.DIM)
        self.z_alt = -float(cfg.ALTITUDE)
        self.lockstep = cfg.STEP_MODE == "lockstep"
        self.yaw = self.A.YawMode(False, 0.0)
        self.wind = np.zeros(2)
        self.rng_wind = np.random.default_rng([int(cfg.SEED), 77])
        self._setkin = bool(cfg.USE_SET_KINEMATICS)
        self._col_ts = [None] * self.n
        self._warned_pause = False
        self._paused_ok = 0  # consecutive slots in which the simulator was already paused on return
        self.names = [f"{cfg.VEHICLE_PREFIX}{i + 1}" for i in range(self.n)]
        try:
            self.c.confirmConnection()
        except Exception as e:
            raise SystemExit(f"cannot reach AirSim at {cfg.AIRSIM_IP}:{cfg.AIRSIM_PORT} ({type(e).__name__}: {e}). "
                             "Is the simulator running?") from None
        self._check_vehicles()
        self.offsets = self._spawn_offsets()
        self.cont_scale = self._continue_scale()
        if self.lockstep and self.cont_scale != 1.0:
            print(f"[airsim] lockstep: each slot calls simContinueForTime(DT x {self.cont_scale:g}) to advance DT of "
                  "sim time", flush=True)
        self._ts, self.slot_dt = None, [0, float("inf"), 0.0, 0]  # slots, shortest, longest, off by more than 10%
        self._takeoff()

    # ---- setup ---------------------------------------------------------------------------------------------------
    def _check_vehicles(self):
        try:
            have = list(self.c.listVehicles())
        except Exception:
            return
        missing = [nm for nm in self.names if nm not in have]
        if not missing:
            return
        if self.n == 1 and len(have) == 1:
            print(f"[airsim] using the vehicle {have[0]!r}", flush=True)
            self.names = [have[0]]
            return
        raise SystemExit(f"AirSim has the vehicles {have}, but {missing} are needed (NUM_DRONES={self.n}). "
                         f"Write a matching settings.json (python {os.path.basename(_SCRIPT)} settings --write) "
                         "and restart AirSim.")

    def _spawn_offsets(self):
        """World position of each drone's start point (AirSim reports a vehicle's position relative to its own start
        point, and draws markers in the world frame)."""
        off = np.zeros((self.n, 3))
        for i in range(self.n):
            off[i] = (0.0, i * float(self.cfg.ARENA_SPACING), 0.0)
        try:
            s = json.loads(self.c.getSettingsString())
            veh = s.get("Vehicles", {})
            for i, nm in enumerate(self.names):
                if nm in veh:
                    v = veh[nm]
                    off[i] = (float(v.get("X", 0.0)), float(v.get("Y", 0.0)), float(v.get("Z", 0.0)))
        except Exception:
            pass
        return off

    def _continue_scale(self):
        """The factor on DT in simContinueForTime. AirSim 1.8.1 with the SteppableClock runs the call for that many
        seconds of wall time, so the sim clock moves ClockSpeed times as far (measured 4.98 to 5.00 at ClockSpeed 5)."""
        v = self.cfg.CONTINUE_SCALE
        if v != "auto":
            return float(v)
        try:
            s = json.loads(self.c.getSettingsString())
            k = float(s.get("ClockSpeed", 1.0))
            if s.get("ClockType", "") == "SteppableClock" and k > 0:
                return 1.0 / k
        except Exception:
            pass
        return 1.0

    def _takeoff(self):
        """Unpause (a crashed run can leave the simulator paused), take off the drones that are on the ground, fly
        to START_POS at the task altitude, then pause again in lockstep mode."""
        c, cfg = self.c, self.cfg
        c.simPause(False)
        for nm in self.names:
            c.enableApiControl(True, nm)
            c.armDisarm(True, nm)
        landed = []
        for nm in self.names:
            try:
                landed.append(int(c.getMultirotorState(nm).landed_state) == 0)
            except Exception:
                landed.append(True)
        futs = [c.takeoffAsync(timeout_sec=float(cfg.TAKEOFF_TIMEOUT), vehicle_name=nm)
                for nm, lnd in zip(self.names, landed) if lnd]
        for f in futs:
            try:
                f.join()
            except Exception as e:  # e.g. "cannot take off because the vehicle is moving"; moveToPosition follows
                print(f"[airsim] takeoff: {e}", flush=True)
        x0, y0 = float(cfg.start[0]), float(cfg.start[1])
        z0 = float(cfg.start[2]) if self.D == 3 else self.z_alt
        futs = [c.moveToPositionAsync(x0, y0, z0, 3.0, timeout_sec=30.0, yaw_mode=self.yaw, vehicle_name=nm)
                for nm in self.names]
        for f in futs:
            f.join()
        if self.lockstep:
            c.simPause(True)
            self._ts = c.getMultirotorState(self.names[0]).timestamp
        for i, nm in enumerate(self.names):
            self._col_ts[i] = c.simGetCollisionInfo(nm).time_stamp

    # ---- stepping ------------------------------------------------------------------------------------------------
    def reset(self, start):
        for i in range(self.n):
            self.teleport(i, start)
        pos, vel, _ = self.read()
        return pos, vel

    def step(self, cmd, held=None):
        cfg, c, A = self.cfg, self.c, self.A
        dur = cfg.cmd_duration if self.lockstep else float(cfg.DT)
        drive = A.DrivetrainType.MaxDegreeOfFreedom
        futs = []
        for i, nm in enumerate(self.names):
            v = cmd[i]
            if self.D == 2:
                futs.append(c.moveByVelocityZAsync(float(v[0]), float(v[1]), self.z_alt, dur, drive, self.yaw,
                                                   vehicle_name=nm))
            else:
                futs.append(c.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), dur, drive, self.yaw,
                                                  vehicle_name=nm))
        if cfg.WIND_STD > 0:
            a = math.exp(-float(cfg.DT) / float(cfg.WIND_TAU))
            self.wind = a * self.wind + float(cfg.WIND_STD) * math.sqrt(1 - a * a) * self.rng_wind.standard_normal(2)
            c.simSetWind(A.Vector3r(float(self.wind[0]), float(self.wind[1]), 0.0))
        if self.lockstep:
            if float(cfg.CMD_SETTLE) > 0:  # let the server start the commands before the clock moves
                time.sleep(float(cfg.CMD_SETTLE))
            c.simContinueForTime(float(cfg.DT) * self.cont_scale)
            self._wait_paused()
            self._track_slot()
        else:
            for f in futs:
                f.join()
        del futs  # unjoined futures: their replies are discarded
        return self.read()

    def _track_slot(self):
        """The sim time of the slot, from drone 1's clock. The simulator counts the continue in wall time, so a
        simulator that falls behind (a busy CPU) makes the slots shorter than DT; this warns when that happens."""
        try:
            ts = self.c.getMultirotorState(self.names[0]).timestamp
        except Exception:
            return
        if self._ts is not None:
            d, s, dt = (ts - self._ts) / 1e9, self.slot_dt, float(self.cfg.DT)
            s[0] += 1
            s[1], s[2] = min(s[1], d), max(s[2], d)
            if abs(d - dt) > 0.1 * dt:
                s[3] += 1
                if s[3] <= 3 or s[3] % 100 == 0:
                    print(f"[airsim] a slot took {d:.4f} s of sim time instead of DT = {dt:g} ({s[3]} such slots of "
                          f"{s[0]}); is the CPU busy?", flush=True)
        self._ts = ts

    def _wait_paused(self):
        """Current AirSim returns from simContinueForTime when the time has run; older versions return at once and
        pause later. Polls simIsPause until 50 slots in a row show the first kind, then stops polling."""
        if self._paused_ok >= 50:
            return
        t0 = time.time()
        limit = max(2.0, 20.0 * float(self.cfg.DT) / max(float(self.cfg.CLOCK_SPEED), 0.1))
        first = True
        while True:
            try:
                if self.c.simIsPause():
                    self._paused_ok = self._paused_ok + 1 if first else -10 ** 9
                    return
            except Exception:
                return
            first = False
            if time.time() - t0 > limit:
                if not self._warned_pause:
                    print("[airsim] the simulator did not pause after simContinueForTime; check STEP_MODE", flush=True)
                    self._warned_pause = True
                return
            time.sleep(0.0005)

    def read(self):
        n, D, c = self.n, self.D, self.c
        pos, vel, coll = np.zeros((n, D)), np.zeros((n, D)), np.zeros(n, bool)
        for i, nm in enumerate(self.names):
            k = c.simGetGroundTruthKinematics(nm)
            p, v = k.position, k.linear_velocity
            pos[i] = (p.x_val, p.y_val, p.z_val)[:D]
            vel[i] = (v.x_val, v.y_val, v.z_val)[:D]
            ci = c.simGetCollisionInfo(nm)
            if ci.has_collided and ci.time_stamp != self._col_ts[i]:
                coll[i] = True
                self._col_ts[i] = ci.time_stamp
        return pos, vel, coll

    def teleport(self, i, p):
        c, A, nm = self.c, self.A, self.names[i]
        x, y = float(p[0]), float(p[1])
        z = float(p[2]) if self.D == 3 else self.z_alt
        try:
            c.enableApiControl(True, nm)
            c.armDisarm(True, nm)
        except Exception:
            pass
        done = False
        if self._setkin:
            try:
                c.simSetKinematics(_kin_state(A, x, y, z), True, nm)
                done = True
            except Exception as e:
                self._setkin = False
                print(f"[airsim] simSetKinematics is not available ({e}); resets use simSetVehiclePose", flush=True)
        if not done:
            c.simSetVehiclePose(A.Pose(A.Vector3r(x, y, z), A.Quaternionr(0.0, 0.0, 0.0, 1.0)), True, nm)
        try:
            self._col_ts[i] = c.simGetCollisionInfo(nm).time_stamp
        except Exception:
            pass

    def pause(self, flag):
        if not self.lockstep:
            self.c.simPause(bool(flag))

    def close(self):
        """Leave the drones hovering: the hover commands go out before the simulator runs again."""
        s = getattr(self, "slot_dt", [0])
        if s[0]:
            print(f"[airsim] sim time per slot over {s[0]} slots: {s[1]:.4f} to {s[2]:.4f} s (DT {self.cfg.DT:g}); "
                  f"{s[3]} off by more than 10%", flush=True)
        try:
            if self.cfg.WIND_STD > 0:
                self.c.simSetWind(self.A.Vector3r(0.0, 0.0, 0.0))
            futs = [self.c.hoverAsync(vehicle_name=nm) for nm in self.names]
            if self.lockstep:
                time.sleep(max(float(self.cfg.CMD_SETTLE), 0.001))
            self.c.simPause(False)
            del futs
        except Exception:
            pass


def make_backend(cfg, n, seeds):
    if cfg.BACKEND == "airsim":
        return AirSimBackend(cfg, n)
    return SurrogateBackend(cfg, n, seeds)


# ======================================================================================================================
# AirSim drawing: red current target, green hit targets, safe box, trail, the controller's estimate, on-screen numbers
# ======================================================================================================================
class AirSimViz:
    RED, GREEN, GREY = [1.0, 0.1, 0.1, 1.0], [0.1, 1.0, 0.2, 1.0], [0.7, 0.7, 0.7, 1.0]
    BLUE, CYAN, ORANGE, YELLOW = [0.2, 0.6, 1.0, 0.9], [0.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]

    def __init__(self, backend, cfg):
        self.b, self.cfg, self.c, self.A = backend, cfg, backend.c, backend.A
        self.n = backend.n
        self.which = list(range(self.n)) if cfg.DRAW_ALL_DRONES else [0]
        self.targets = [None] * self.n
        self.hits = [deque(maxlen=max(1, int(cfg.SHOW_HITS))) for _ in range(self.n)]
        self.dirty, self.ok = True, True
        self._t_stats = self._t_cam = 0.0

    def _w(self, i, p):
        o = self.b.offsets[i]
        z = float(p[2]) if len(p) >= 3 else self.b.z_alt
        return self.A.Vector3r(float(p[0]) + o[0], float(p[1]) + o[1], z + o[2])

    def _ring(self, i, g, r, k=32):
        z = float(g[2]) if len(g) >= 3 else self.b.z_alt
        return [self._w(i, (g[0] + r * math.cos(a), g[1] + r * math.sin(a), z))
                for a in np.linspace(0.0, 2.0 * math.pi, k + 1)]

    def set_target(self, i, g):
        self.targets[i] = np.asarray(g, float).copy()
        self.dirty = True

    def hit(self, i, g):
        if int(self.cfg.SHOW_HITS) > 0:
            self.hits[i].append(np.asarray(g, float).copy())
        self.dirty = True

    def redraw(self):
        c, cfg, R = self.c, self.cfg, float(self.cfg.HIT_RADIUS)
        c.simFlushPersistentMarkers()
        lo, hi = cfg.safe_lo, cfg.safe_hi
        for i in self.which:
            z = (lo[2] + hi[2]) / 2 if len(lo) == 3 else self.b.z_alt
            box = [(lo[0], lo[1], z), (lo[0], hi[1], z), (hi[0], hi[1], z), (hi[0], lo[1], z), (lo[0], lo[1], z)]
            c.simPlotLineStrip([self._w(i, p) for p in box], self.GREY, 3.0, -1.0, True)
            if self.hits[i]:
                c.simPlotPoints([self._w(i, h) for h in self.hits[i]], self.GREEN, 22.0, -1.0, True)
                for h in self.hits[i]:
                    c.simPlotLineStrip(self._ring(i, h, R, 16), self.GREEN, 2.0, -1.0, True)
            g = self.targets[i]
            if g is not None:
                c.simPlotPoints([self._w(i, g)], self.RED, 32.0, -1.0, True)
                c.simPlotLineStrip(self._ring(i, g, R), self.RED, 6.0, -1.0, True)
        self.dirty = False

    def slot(self, env):
        if not self.ok:
            return
        cfg, c = self.cfg, self.c
        try:
            if self.dirty:
                self.redraw()
            trail = float(cfg.TRAIL_SECONDS)
            if trail > 0:
                c.simPlotPoints([self._w(0, env.pos[0])], self.BLUE, 7.0, trail, False)
                if env.outcome[0] == _DELIV:
                    c.simPlotPoints([self._w(0, env.pos[0])], self.CYAN, 14.0, trail, False)
                elif env.outcome[0] == _LOST:
                    c.simPlotPoints([self._w(0, env.pos[0])], self.ORANGE, 14.0, trail, False)
            if cfg.SHOW_ESTIMATE and env.mode[0] == _TASK:
                c.simPlotPoints([self._w(0, env.xh[0])], self.YELLOW, 11.0, 2.0 * float(cfg.DT), False)
        except Exception as e:
            self.ok = False
            print(f"[viz] drawing in AirSim stopped: {type(e).__name__}: {e}", flush=True)

    def stats(self, lines):
        if not (self.ok and self.cfg.ONSCREEN_STATS):
            return
        now = time.time()
        if now - self._t_stats < float(self.cfg.STATS_PERIOD):
            return
        self._t_stats = now
        try:
            for k, v in lines:
                self.c.simPrintLogMessage(k, v, 0)
        except Exception:
            pass

    def camera(self, path):
        if not self.cfg.CAMERA_STREAM:
            return
        now = time.time()
        if now - self._t_cam < float(self.cfg.CAMERA_PERIOD):
            return
        self._t_cam = now
        try:
            r = self.c.simGetImages([self.A.ImageRequest(self.cfg.CAMERA_NAME, 0, False, True)],
                                    vehicle_name=self.b.names[0])
            data = r[0].image_data_uint8 if r else None
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                write_bytes(path, bytes(data))
        except Exception as e:
            print(f"[viz] camera image failed ({e}); CAMERA_STREAM is turned off", flush=True)
            self.cfg.CAMERA_STREAM = False


# ======================================================================================================================
# The task, the channel and the controller's record (vectorized over drones or surrogate environments)
# ======================================================================================================================
def _onehot3(codes):
    out = np.zeros((len(codes), 3))
    m = codes < 3
    out[np.flatnonzero(m), codes[m]] = 1.0
    return out


class DroneTaskEnv:
    """Slot order: (1) the scheduler decides, (2) transmit(): the channel outcome and the controller's record,
    (3) the controller decides, (4) advance(): physics, success / failure, hold and reset.

    The controller's record (also known to the drone, which sees the acknowledgments and the commands): the last
    delivered state, propagated through the commands sent since (x_hat, v_hat) with the first-order model PRED_TAU,
    its age, the losses and silences since, the last outcomes, the last command, the slots in the task. After a
    success the controller knows the drone is in the window, so x_hat is projected onto it; after a failure the drone
    is at START_POS. Channel and noise random numbers are drawn every slot whatever is decided, so policies compared
    with the same seeds see the same random numbers."""

    def __init__(self, cfg, backend, seeds):
        self.cfg, self.b = cfg, backend
        self.n, self.D = backend.n, int(cfg.DIM)
        n, D = self.n, self.D
        self.seeds = [int(s) for s in seeds]
        self.rng_t = [np.random.default_rng([s, 11]) for s in self.seeds]
        self.rng_s = [np.random.default_rng([s, 22]) for s in self.seeds]
        self._B, self._k = 1024, 1024
        self.start = np.asarray(cfg.start, float).copy()
        self.R = float(cfg.HIT_RADIUS)
        tau, dt = float(cfg.PRED_TAU), float(cfg.DT)
        self._pa = math.exp(-dt / tau) if tau > 0 else 0.0
        self._pb = tau * (1.0 - self._pa) if tau > 0 else 0.0
        if cfg.TARGET_MODE == "cycle":
            tl = []
            for e in cfg.TARGET_LIST:
                g = list(self.start[:2] + np.asarray(e[:2], float))
                if D == 3:
                    g.append(-float(e[2]) if len(e) > 2 else float(self.start[2]))
                tl.append(g)
            self._tlist = np.array(tl, float)
        self.pos, self.vel = backend.reset(self.start)
        self.mode, self.left = np.full(n, _TASK), np.zeros(n, int)
        self.tidx = np.full(n, -1)
        self.target = np.zeros((n, D))
        for i in range(n):
            self.target[i] = self._new_target(i, after_fail=True)
        self.xh, self.vh = self.pos.copy(), np.zeros((n, D))
        self.age, self.n_lost, self.n_sil = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.since_try = np.full(n, 10 ** 6)
        self.hist, self.seen, self.outcome = np.full((n, _HIST), _NONE), np.full(n, _NONE), np.full(n, _NONE)
        self.last_u = np.zeros((n, D))
        self.sent, self.deliv = np.zeros(n, bool), np.zeros(n, bool)
        self.task_slot, self.task_id = np.zeros(n, int), np.zeros(n, int)
        self.task_start, self.task_start_pos = np.zeros(n, int), self.pos.copy()
        self.task_sends, self.task_deliv = np.zeros(n, int), np.zeros(n, int)
        self.clock = np.zeros(n, int)
        self.events = [("target", i, self.target[i].copy()) for i in range(n)]
        self.phi = self.potential()

    # ---- random numbers and targets ------------------------------------------------------------------------------
    def _ensure(self):
        if self._k >= self._B:
            self._ch = np.stack([r.random(self._B) for r in self.rng_s])
            self._nz = np.stack([r.standard_normal((self._B, self.D)) for r in self.rng_s])
            self._k = 0

    def _new_target(self, i, after_fail):
        cfg, r, D = self.cfg, self.rng_t[i], self.D
        if cfg.TARGET_MODE == "cycle":
            m = len(self._tlist)
            j = int(r.integers(m)) if (after_fail or self.tidx[i] < 0) else (int(self.tidx[i]) + 1) % m
            self.tidx[i] = j
            return self._tlist[j].copy()
        ref = self.start if (after_fail or self.tidx[i] < 0) else self.target[i]
        self.tidx[i] = 0
        th, dmin, dmax = float(cfg.TARGET_HALF), float(cfg.TARGET_MIN_DIST), float(cfg.TARGET_MAX_DIST)
        g = self.start.copy()
        for _ in range(1000):
            g = self.start.copy()
            g[:2] += r.uniform(-th, th, 2)
            if D == 3:
                g[2] = -r.uniform(float(cfg.Z_RANGE[0]), float(cfg.Z_RANGE[1]))
            if dmin <= np.linalg.norm(g - ref) <= dmax:
                break
        return g

    # ---- what each side sees -------------------------------------------------------------------------------------
    def _record(self, target_rel=True):
        """The common record; target_rel=False leaves out the target and the task time (a sender that does not know
        the tasks, SENDER_KNOWS_TARGET=False)."""
        cfg, g = self.cfg, self.target
        parts = []
        if target_rel:
            parts += [(self.xh - g) / float(cfg.TARGET_HALF), np.clip((self.xh - g) / self.R, -6, 6) / 6]
        cnt = np.stack([np.minimum(self.age, _AGE_CAP), np.minimum(self.n_lost, _AGE_CAP),
                        np.minimum(self.n_sil, _AGE_CAP)], 1) / _AGE_CAP
        parts += [self.vh / float(cfg.U_MAX), (self.xh - self.start) / float(cfg.SAFE_HALF),
                  self.last_u / float(cfg.U_MAX), cnt]
        if target_rel:
            parts.append(np.clip(self.task_slot / max(int(cfg.MAX_TASK_SLOTS), 1), 0, 1)[:, None])
        parts += [(self.hist == _LOST).astype(float), (self.hist == _DELIV).astype(float)]
        return np.concatenate(parts, 1)

    def _private(self, target_rel=True):
        cfg, g = self.cfg, self.target
        if target_rel:
            parts = [(self.pos - g) / float(cfg.TARGET_HALF), np.clip((self.pos - g) / self.R, -6, 6) / 6]
        else:
            parts = [(self.pos - self.start) / float(cfg.SAFE_HALF)]
        parts += [self.vel / float(cfg.U_MAX), np.clip((self.pos - self.xh) / self.R, -6, 6) / 6,
                  (self.vel - self.vh) / float(cfg.U_MAX)]
        return np.concatenate(parts, 1)

    def sched_obs(self, kind):
        """push: the drone's own state and the common record; pull (and always): the controller's record."""
        if kind == "push":
            t = bool(self.cfg.SENDER_KNOWS_TARGET)
            return np.concatenate([self._record(t), self._private(t)], 1)
        return self._record(True)

    def ctrl_obs(self):
        """The controller's record after this slot's outcome."""
        return np.concatenate([self._record(True), _onehot3(self.seen)], 1)

    def critic_obs(self):
        m = np.zeros((self.n, 3))
        m[np.arange(self.n), self.mode] = 1.0
        left = (self.left / max(int(self.cfg.RESET_SLOTS), int(self.cfg.HOLD_SLOTS), 1))[:, None]
        return np.concatenate([self._record(True), self._private(True), m, left], 1)

    def dims(self, kind):
        return self.ctrl_obs().shape[1], self.sched_obs(kind).shape[1], self.critic_obs().shape[1]

    def potential(self):
        dist = np.linalg.norm(self.pos - self.target, axis=1) / (float(self.cfg.U_MAX) * float(self.cfg.DT))
        return np.where(self.mode == _TASK, -dist, 0.0)

    # ---- the slot -------------------------------------------------------------------------------------------------
    def transmit(self, d):
        self._ensure()
        cfg = self.cfg
        task = self.mode == _TASK
        d = (np.asarray(d, float) > 0.5) & task
        deliv = d & (self._ch[:, self._k] < float(cfg.P_S))
        lost = d & ~deliv
        self.outcome = np.where(deliv, _DELIV, np.where(lost, _LOST, np.where(task, _SILENT, _NONE)))
        seen = self.outcome.copy()
        if not cfg.LOSS_VISIBLE:
            seen[seen == _LOST] = _SILENT
        self.seen = seen
        if deliv.any():
            self.xh[deliv], self.vh[deliv] = self.pos[deliv], self.vel[deliv]
            self.age[deliv] = self.n_lost[deliv] = self.n_sil[deliv] = 0
        self.n_lost[seen == _LOST] += 1
        self.n_sil[seen == _SILENT] += 1
        self.since_try[d] = 0
        self.sent, self.deliv = d, deliv
        self.task_sends[d] += 1
        self.task_deliv[deliv] += 1
        return self.ctrl_obs()

    def advance(self, u_norm):
        self._ensure()
        cfg, n = self.cfg, self.n
        self.events = []
        task = self.mode == _TASK
        held = ~task
        u = np.clip(np.asarray(u_norm, float).reshape(n, self.D), -1.0, 1.0) * float(cfg.U_MAX)
        if cfg.U_NORM_CAP:
            nrm = np.linalg.norm(u, axis=1, keepdims=True)
            u *= np.minimum(1.0, float(cfg.U_MAX) / np.maximum(nrm, 1e-9))
        u[held] = 0.0
        w = self._nz[:, self._k] * float(cfg.NOISE_STD)
        w[held] = 0.0
        self._k += 1
        pos, vel, coll = self.b.step(u + w, held)
        self.pos, self.vel = np.asarray(pos, float).copy(), np.asarray(vel, float).copy()
        self.last_u = u
        dt = float(cfg.DT)  # the controller predicts with the command it sent; it does not know the noise
        if self._pa > 0:
            self.xh = self.xh + u * dt + (self.vh - u) * self._pb
            self.vh = u + (self.vh - u) * self._pa
        else:
            self.xh, self.vh = self.xh + u * dt, u.copy()
        self.age += 1
        self.since_try += 1
        self.task_slot[task] += 1
        self.hist = np.roll(self.hist, 1, axis=1)
        self.hist[:, 0] = self.seen
        dg = np.linalg.norm(self.pos - self.target, axis=1)
        hit = task & (dg <= self.R)
        if float(cfg.HIT_MAX_SPEED) > 0:
            hit &= np.linalg.norm(self.vel, axis=1) <= float(cfg.HIT_MAX_SPEED)
        out = task & np.any((self.pos < cfg.safe_lo) | (self.pos > cfg.safe_hi), axis=1)
        col = task & np.asarray(coll, bool) & bool(cfg.COLLISION_IS_FAILURE)
        hit &= ~(out | col)
        tout = task & (int(cfg.MAX_TASK_SLOTS) > 0) & (self.task_slot >= int(cfg.MAX_TASK_SLOTS)) & ~hit
        code = np.zeros(n, int)
        code[tout], code[out], code[col] = 2, 1, 3
        fail = code > 0
        k = np.ones(n, int)
        for i in np.flatnonzero(held):  # held and reset drones count down
            self.left[i] -= 1
            if self.left[i] <= 0:
                self._start_task(i)
        for i in np.flatnonzero(hit):
            g = self.target[i].copy()
            self._end_task(i, "success", dg[i])
            e = self.xh[i] - g
            ne = float(np.linalg.norm(e))
            if ne > self.R:  # the window cut: the controller now knows |x - g| <= R
                self.xh[i] = g + e * (self.R / ne)
            self.target[i] = self._new_target(i, after_fail=False)
            self.events.append(("target", i, self.target[i].copy()))
            if cfg.PHYSICAL_PENALTY and int(cfg.HOLD_SLOTS) > 0:
                self.mode[i], self.left[i] = _HOLD, int(cfg.HOLD_SLOTS)
            else:
                k[i] += 0 if cfg.PHYSICAL_PENALTY else int(cfg.HOLD_SLOTS)
                self._start_task(i)
        for i in np.flatnonzero(fail):
            self._end_task(i, _FAIL_NAMES[int(code[i])], dg[i])
            self.b.teleport(i, self.start)
            self.pos[i], self.vel[i] = self.start, 0.0
            self.xh[i], self.vh[i] = self.start, 0.0
            self.age[i] = self.n_lost[i] = self.n_sil[i] = 0
            self.hist[i] = _NONE
            self.since_try[i] = 10 ** 6
            self.target[i] = self._new_target(i, after_fail=True)
            self.events.append(("target", i, self.target[i].copy()))
            if cfg.PHYSICAL_PENALTY and int(cfg.RESET_SLOTS) > 0:
                self.mode[i], self.left[i] = _RESET, int(cfg.RESET_SLOTS)
            else:
                k[i] += 0 if cfg.PHYSICAL_PENALTY else int(cfg.RESET_SLOTS)
                self._start_task(i)
        self.clock += k
        sent, deliv = self.sent.astype(float), self.deliv.astype(float)
        self.sent, self.deliv = np.zeros(n, bool), np.zeros(n, bool)
        self.phi = self.potential()
        return dict(succ=hit.astype(float), sent=sent, deliv=deliv, k=k, fail=fail, code=code, task=task)

    def _start_task(self, i):
        self.mode[i], self.left[i] = _TASK, 0
        self.task_slot[i] = 0
        self.task_id[i] += 1
        self.task_start[i] = self.clock[i] + 1
        self.task_start_pos[i] = self.pos[i]
        self.task_sends[i] = self.task_deliv[i] = 0

    def _end_task(self, i, outcome, dist):
        self.events.append(("task_end", i, dict(
            env=int(i), task=int(self.task_id[i]), outcome=outcome, start_slot=int(self.task_start[i]),
            end_slot=int(self.clock[i] + 1), slots=int(self.task_slot[i]), sends=int(self.task_sends[i]),
            deliveries=int(self.task_deliv[i]), final_dist=float(dist), target=self.target[i].tolist(),
            start_pos=self.task_start_pos[i].tolist())))

    # ---- rule-based schedulers and the PD controller ------------------------------------------------------------------
    def rule(self, name, param=None):
        task = self.mode == _TASK
        if name == "always":
            d = task.copy()
        elif name == "never":
            d = np.zeros(self.n, bool)
        elif name == "periodic":  # an attempt every `param` slots
            d = task & (self.since_try >= int(param))
        elif name == "age":  # pull: ask when the last delivered state is `param` slots old
            d = task & (self.age >= int(param))
        elif name == "err":  # push: estimation-error threshold
            d = task & (np.linalg.norm(self.pos - self.xh, axis=1) > float(param))
        elif name == "near":  # push: within `param` m of the target
            d = task & (np.linalg.norm(self.pos - self.target, axis=1) <= float(param))
        elif name == "near_est":  # pull: the estimate is within `param` m of the target
            d = task & (np.linalg.norm(self.xh - self.target, axis=1) <= float(param)) & (self.age >= 1)
        else:
            raise ValueError(f"unknown scheduler {name!r}; use one of {_RULES} or learned")
        return d

    def pd(self, kp, kd):
        u = float(kp) * (self.target - self.xh) - float(kd) * self.vh
        return np.clip(u / float(self.cfg.U_MAX), -1.0, 1.0)


# ======================================================================================================================
# Learning: running normalization, the agent (scheduler head, controller head, critic), PPO
# ======================================================================================================================
class RunningNorm:
    def __init__(self, dim, enabled=True):
        self.mean, self.var, self.count, self.enabled = np.zeros(dim), np.ones(dim), 1e-4, bool(enabled)

    def update(self, x):
        x = np.asarray(x, float).reshape(-1, len(self.mean))
        if not self.enabled or len(x) == 0:
            return
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.mean, self.var, self.count = self.mean + delta * bc / tot, m2 / tot, tot

    def __call__(self, x):
        x = np.asarray(x, float)
        if not self.enabled:
            return x.astype(np.float32)
        return np.clip((x - self.mean) / np.sqrt(np.maximum(self.var, 1e-2)), -5.0, 5.0).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), count=float(self.count), enabled=self.enabled)

    def load(self, s):
        self.mean, self.var = np.asarray(s["mean"], float), np.asarray(s["var"], float)
        self.count, self.enabled = float(s["count"]), bool(s.get("enabled", True))


if torch is not None:
    def _mlp(inp, out, hidden, out_gain):
        layers, d = [], inp
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.orthogonal_(lin.weight, math.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = h
        last = nn.Linear(d, out)
        nn.init.orthogonal_(last.weight, out_gain)
        nn.init.zeros_(last.bias)
        return nn.Sequential(*(layers + [last]))

    class Agent(nn.Module):
        """Controller: Gaussian over the normalized command, mean tanh(MLP(record after the outcome)).
        Scheduler (push or pull): Bernoulli, logit MLP(its inputs). Critic: MLP(everything), training only."""

        def __init__(self, dims, D, sched_kind, hidden=(128, 128), init_log_std=-0.7, p0=0.9, obs_norm=True):
            super().__init__()
            dc, ds, dv = (int(x) for x in dims)
            self.dims, self.D, self.sched_kind = (dc, ds, dv), int(D), sched_kind
            self.hidden = tuple(int(h) for h in hidden)
            self.ctrl = _mlp(dc, self.D, self.hidden, 0.01)
            self.log_std = nn.Parameter(torch.full((self.D,), float(init_log_std)))
            self.sched = _mlp(ds, 1, self.hidden, 0.01) if sched_kind in ("push", "pull") else None
            if self.sched is not None:
                p0 = min(max(float(p0), 1e-3), 1 - 1e-3)
                nn.init.constant_(self.sched[-1].bias, math.log(p0 / (1 - p0)))
            self.critic = _mlp(dv, 1, self.hidden, 1.0)
            self.nc, self.ns, self.nv = RunningNorm(dc, obs_norm), RunningNorm(ds, obs_norm), RunningNorm(dv, obs_norm)
            self.dev = torch.device("cpu")

        def to_device(self, dev):
            self.dev = torch.device(dev)
            return self.to(self.dev)

        def t(self, x):
            return torch.as_tensor(np.asarray(x, np.float32), device=self.dev)

        def dist(self, c):
            mu = torch.tanh(self.ctrl(c))
            return torch.distributions.Normal(mu, torch.exp(self.log_std.clamp(-5.0, 1.0)).expand_as(mu))

        @torch.no_grad()
        def act_ctrl(self, c, det=False):
            dist = self.dist(self.t(c))
            u = dist.mean if det else dist.sample()
            return u.cpu().numpy(), dist.log_prob(u).sum(-1).cpu().numpy()

        @torch.no_grad()
        def act_sched(self, s, det=False):
            logit = self.sched(self.t(s)).squeeze(-1)
            p = torch.sigmoid(logit)
            d = (p > 0.5).float() if det else torch.bernoulli(p)
            lp = -F.binary_cross_entropy_with_logits(logit, d, reduction="none")
            return d.cpu().numpy(), lp.cpu().numpy(), p.cpu().numpy()

        @torch.no_grad()
        def value(self, v):
            return self.critic(self.t(v)).squeeze(-1).cpu().numpy()

    def _bernoulli_entropy(logit):
        p = torch.sigmoid(logit)
        return p * F.softplus(-logit) + (1 - p) * F.softplus(logit)

    class PPO:
        def __init__(self, agent, cfg):
            self.a, self.cfg = agent, cfg
            self.lr = float(cfg.LR)
            self.opt = torch.optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)

        def update(self, B, adv, ret, progress):
            cfg, a = self.cfg, self.a
            T, N = B["rew"].shape
            M = T * N

            def fl(x):
                return x.reshape(M, *x.shape[2:])

            c, s, v, u, d = a.t(fl(B["c"])), a.t(fl(B["s"])), a.t(fl(B["v"])), a.t(fl(B["u"])), a.t(fl(B["d"]))
            lpc0, lps0, mc, ms, R = a.t(fl(B["lpc"])), a.t(fl(B["lps"])), a.t(fl(B["mc"])), a.t(fl(B["ms"])), \
                a.t(fl(ret))
            A_np = fl(adv).astype(np.float64)
            sel = fl(B["mc"]) > 0
            if sel.sum() > 1:
                A_np = (A_np - A_np[sel].mean()) / (A_np[sel].std() + 1e-8)
            A = a.t(A_np)
            if cfg.LR_SCHEDULE == "linear":
                self.lr = float(cfg.LR) * max(1.0 - progress, 0.05)
            elif cfg.LR_SCHEDULE == "constant":
                self.lr = float(cfg.LR)
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            mb = max(1, M // max(1, int(cfg.MINIBATCHES)))
            idx = np.arange(M)
            log = defaultdict(list)
            epochs, kl_epoch = 0, 0.0
            for _ in range(int(cfg.EPOCHS)):
                np.random.shuffle(idx)
                kls = []
                for st in range(0, M, mb):
                    j = torch.as_tensor(idx[st:st + mb], device=a.dev)
                    dist = a.dist(c[j])
                    lpc = dist.log_prob(u[j]).sum(-1)
                    entc = dist.entropy().sum(-1)
                    if a.sched is not None:
                        logit = a.sched(s[j]).squeeze(-1)
                        lps = -F.binary_cross_entropy_with_logits(logit, d[j], reduction="none")
                        ents = _bernoulli_entropy(logit)
                    else:
                        lps = ents = torch.zeros_like(lpc)
                    mcj, msj = mc[j], ms[j]
                    logr = (lpc - lpc0[j]) * mcj + (lps - lps0[j]) * msj  # joint ratio of the two heads
                    ratio = torch.exp(logr)
                    Aj = A[j]
                    pg = -torch.min(ratio * Aj, torch.clamp(ratio, 1 - float(cfg.CLIP), 1 + float(cfg.CLIP)) * Aj)
                    nc_, ns_ = mcj.sum().clamp(min=1.0), msj.sum().clamp(min=1.0)
                    pg_loss = (pg * mcj).sum() / nc_
                    v_loss = 0.5 * ((a.critic(v[j]).squeeze(-1) - R[j]) ** 2).mean()
                    ent = float(cfg.ENT_CTRL) * (entc * mcj).sum() / nc_ + float(cfg.ENT_SCHED) * (ents * msj).sum() / ns_
                    loss = pg_loss + float(cfg.VF_COEF) * v_loss - ent
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(a.parameters(), float(cfg.MAX_GRAD_NORM))
                    self.opt.step()
                    with torch.no_grad():
                        kl = (((ratio - 1) - logr) * mcj).sum() / nc_
                        cf = ((((ratio - 1).abs() > float(cfg.CLIP)).float()) * mcj).sum() / nc_
                    kls.append(float(kl))
                    log["pg"].append(float(pg_loss.detach()))
                    log["v"].append(float(v_loss.detach()))
                    log["cf"].append(float(cf))
                    log["entc"].append(float((entc.detach() * mcj).sum() / nc_))
                    log["ents"].append(float((ents.detach() * msj).sum() / ns_) if a.sched is not None else 0.0)
                epochs += 1
                kl_epoch = float(np.mean(kls)) if kls else 0.0
                if float(cfg.TARGET_KL) > 0 and kl_epoch > 2.5 * float(cfg.TARGET_KL):
                    break
            if cfg.LR_SCHEDULE == "kl_adaptive" and float(cfg.TARGET_KL) > 0:
                if kl_epoch < float(cfg.TARGET_KL) / 2:
                    self.lr = min(self.lr * 1.5, float(cfg.LR_MAX))
                elif kl_epoch > 2 * float(cfg.TARGET_KL):
                    self.lr = max(self.lr / 1.5, float(cfg.LR_MIN))
            vr = fl(ret) - fl(B["val"])
            ev = 1.0 - float(np.var(vr)) / max(float(np.var(fl(ret))), 1e-12)
            return dict(kl=kl_epoch, clipfrac=float(np.mean(log["cf"])), policy_loss=float(np.mean(log["pg"])),
                        value_loss=float(np.mean(log["v"])), ent_ctrl=float(np.mean(log["entc"])),
                        ent_sched=float(np.mean(log["ents"])), expl_var=ev, epochs=epochs, lr=self.lr,
                        log_std=float(a.log_std.detach().mean()))


def gae(rew, disc, val, last_val, lam):
    """Generalized advantage estimation for a continuing task; disc[t] = gamma^(slots of transition t)."""
    T = rew.shape[0]
    adv = np.zeros_like(rew, dtype=np.float64)
    last = np.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + disc[t] * nv - val[t]
        last = delta + disc[t] * lam * last
        adv[t] = last
    return adv, adv + val


def save_ckpt(path, agent, cfg, **extra):
    obj = dict(agent=agent.state_dict(), norms={k: getattr(agent, k).state() for k in ("nc", "ns", "nv")},
               dims=list(agent.dims), D=agent.D, sched_kind=agent.sched_kind, hidden=list(agent.hidden),
               cfg=cfg_dump(cfg), version=_VERSION)
    obj.update(extra)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path):
    _need_torch()
    if not os.path.isfile(path):
        raise SystemExit(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def agent_from_ckpt(path, cfg):
    ck = load_ckpt(path)
    a = Agent(ck["dims"], ck["D"], ck["sched_kind"], hidden=ck["hidden"], obs_norm=True)
    a.load_state_dict(ck["agent"])
    for k in ("nc", "ns", "nv"):
        getattr(a, k).load(ck["norms"][k])
    a.to_device(_device(cfg))
    a.eval()
    return a, ck


def init_from(agent, path):
    """Start a phase from a checkpoint: the controller, its log std, the critic and their input normalization; also
    the scheduler when the checkpoint has one of the same kind (push or pull), e.g. to continue in AirSim a policy
    trained on the surrogate. Returns the checkpoint's price."""
    ck = load_ckpt(path)
    if int(ck["D"]) != agent.D or tuple(ck["dims"])[0] != agent.dims[0] or tuple(ck["dims"])[2] != agent.dims[2]:
        raise SystemExit(f"{path} does not match the current inputs (DIM or the observation layout changed)")
    if tuple(ck["hidden"]) != agent.hidden:
        raise SystemExit(f"{path} has HIDDEN={tuple(ck['hidden'])}, the current setting is {agent.hidden}")
    take_sched = (agent.sched is not None and ck["sched_kind"] == agent.sched_kind
                  and tuple(ck["dims"])[1] == agent.dims[1])
    own = agent.state_dict()
    for k, v in ck["agent"].items():
        if k.startswith(("ctrl.", "critic.")) or k == "log_std" or (take_sched and k.startswith("sched.")):
            own[k] = v
    agent.load_state_dict(own)
    agent.nc.load(ck["norms"]["nc"])
    agent.nv.load(ck["norms"]["nv"])
    if take_sched:
        agent.ns.load(ck["norms"]["ns"])
    return ck.get("lam"), ("controller, critic and scheduler" if take_sched else "controller and critic")


def _ckpt_of(run_dir, which="latest"):
    for name in ((f"ckpt_{which}.pt", "ckpt_latest.pt") if which == "best" else ("ckpt_latest.pt",)):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _find_latest_run(root, phase):
    best, best_t = None, -1.0
    for cf in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = read_json(cf)
        if not c or c.get("phase") != phase:
            continue
        p = os.path.join(os.path.dirname(cf), "ckpt_latest.pt")
        if os.path.isfile(p) and os.path.getmtime(p) > best_t:
            best, best_t = p, os.path.getmtime(p)
    return best


# ======================================================================================================================
# Policies, statistics, the slot loop
# ======================================================================================================================
class Policy:
    """A scheduler and a controller. sched: 'learned' (the agent's head) or a rule in _RULES;
    controller: 'agent' (the learned head) or 'pd'."""

    def __init__(self, cfg, agent=None, sched="learned", param=None, controller="agent", deterministic=False):
        self.cfg, self.agent, self.sched, self.param = cfg, agent, sched, param
        self.controller, self.det = controller, bool(deterministic)
        self.learned_sched = sched == "learned"
        if self.learned_sched and (agent is None or agent.sched is None):
            raise SystemExit("a learned scheduler needs a checkpoint from joint_push or joint_pull")
        if not self.learned_sched and sched not in _RULES:
            raise SystemExit(f"unknown scheduler {sched!r}; use learned or one of {_RULES}")
        if controller == "agent" and agent is None:
            raise SystemExit("the learned controller needs a checkpoint (or use the pd controller)")
        self.kind = agent.sched_kind if (self.learned_sched and agent is not None) else sched

    def describe(self):
        s = f"learned {self.kind}" if self.learned_sched else (self.sched + (f"({self.param:g})" if
                                                                            self.param is not None else ""))
        return f"scheduler {s}, controller {self.controller}"

    def schedule(self, env):
        if self.learned_sched:
            s_raw = env.sched_obs(self.agent.sched_kind)
            s_n = self.agent.ns(s_raw)
            d, lp, p = self.agent.act_sched(s_n, self.det)
            return d, lp, p, s_n, s_raw
        d = env.rule(self.sched, self.param).astype(float)
        return d, np.zeros(env.n), d, None, None

    def control(self, env, c_raw):
        if self.controller == "pd":
            return env.pd(self.cfg.PD_KP, self.cfg.PD_KD), np.zeros(env.n), None
        c_n = self.agent.nc(c_raw)
        u, lp = self.agent.act_ctrl(c_n, self.det)
        return u, lp, c_n


class Stats:
    """Per-environment sums over a stretch of slots."""

    def __init__(self, n, C):
        self.n, self.C = n, float(C)
        z = lambda: np.zeros(n)  # noqa: E731
        self.succ, self.sends, self.deliv, self.slots, self.tslots = z(), z(), z(), z(), z()
        self.fail = {c: z() for c in _FAIL_NAMES}
        self.p_sum, self.p_cnt = z(), z()

    def add(self, out, p, events):
        self.succ += out["succ"]
        self.sends += out["sent"]
        self.deliv += out["deliv"]
        self.slots += out["k"]
        for c in _FAIL_NAMES:
            self.fail[c] += out["code"] == c
        t = out["task"]
        self.p_sum += np.where(t, p, 0.0)
        self.p_cnt += t
        for kind, i, info in events:
            if kind == "task_end" and info["outcome"] == "success":
                self.tslots[i] += info["slots"]

    def merge(self, other):
        for key in ("succ", "sends", "deliv", "slots", "tslots", "p_sum", "p_cnt"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for c in _FAIL_NAMES:
            self.fail[c] = self.fail[c] + other.fail[c]

    def summary(self):
        S, M, T = self.succ.sum(), self.sends.sum(), max(self.slots.sum(), 1.0)
        nf = sum(f.sum() for f in self.fail.values())
        tasks = S + nf
        return dict(J=float((S - self.C * M) / T), succ=float(S), fail=float(nf), tasks=float(tasks),
                    fail_out=float(self.fail[1].sum()), fail_timeout=float(self.fail[2].sum()),
                    fail_collision=float(self.fail[3].sum()), slots=float(T),
                    succ_rate=float(S / tasks) if tasks else float("nan"),
                    sends_per_task=float(M / tasks) if tasks else float("nan"),
                    deliv_per_task=float(self.deliv.sum() / tasks) if tasks else float("nan"),
                    sends_per_slot=float(M / T), tasks_per_slot=float(S / T),
                    task_slots=float(self.tslots.sum() / S) if S else float("nan"),
                    send_prob=float(self.p_sum.sum() / max(self.p_cnt.sum(), 1.0)))

    def per_env(self):
        out = []
        for i in range(self.n):
            T = max(self.slots[i], 1.0)
            nf = sum(f[i] for f in self.fail.values())
            tasks = self.succ[i] + nf
            out.append(dict(J=(self.succ[i] - self.C * self.sends[i]) / T, tasks_per_slot=self.succ[i] / T,
                            sends_per_slot=self.sends[i] / T, succ=self.succ[i], fail=nf, tasks=tasks,
                            sends=self.sends[i], deliv=self.deliv[i], slots=T, tslots=self.tslots[i]))
        return out


def run_slots(env, pol, T, cfg, lam=0.0, mon=None, store=False, probe=None):
    """Runs T slots. With store=True it also returns the PPO buffer (normalized inputs at collection time, raw inputs
    for the normalizer update, actions, log-probabilities, masks, rewards, discounts, values). With a list as probe,
    it appends, for every task slot, the state seen at decision time and the decision (for the scheduler maps)."""
    n, a = env.n, pol.agent
    st = Stats(n, cfg.COST_C)
    B = None
    if store:
        dc, ds, dv = a.dims
        f32 = np.float32
        B = dict(c=np.zeros((T, n, dc), f32), s=np.zeros((T, n, ds), f32), v=np.zeros((T, n, dv), f32),
                 c_raw=np.zeros((T, n, dc), f32), s_raw=np.zeros((T, n, ds), f32), v_raw=np.zeros((T, n, dv), f32),
                 u=np.zeros((T, n, env.D), f32), d=np.zeros((T, n), f32), lpc=np.zeros((T, n), f32),
                 lps=np.zeros((T, n), f32), mc=np.zeros((T, n), f32), ms=np.zeros((T, n), f32),
                 rew=np.zeros((T, n)), disc=np.zeros((T, n)), val=np.zeros((T, n)))
    gamma, C, shp = float(cfg.GAMMA), float(cfg.COST_C), float(cfg.SHAPING)
    for t in range(T):
        task = env.mode == _TASK
        if store:
            v_raw = env.critic_obs()
            v_n = a.nv(v_raw)
            B["v_raw"][t], B["v"][t], B["val"][t] = v_raw, v_n, a.value(v_n)
        if probe is not None and task.any():
            probe.append(np.stack([np.linalg.norm(env.pos - env.target, axis=1),
                                   np.linalg.norm(env.pos - env.xh, axis=1),
                                   np.linalg.norm(env.xh - env.target, axis=1), env.age.astype(float),
                                   np.linalg.norm(env.vel, axis=1), env.task_slot.astype(float)], 1)[task])
        d, lps, p, s_n, s_raw = pol.schedule(env)
        if probe is not None and task.any():
            probe[-1] = np.concatenate([probe[-1], (np.asarray(d, float)[task] > 0.5)[:, None],
                                        np.asarray(p, float)[task][:, None]], 1)
        c_raw = env.transmit(d)
        u, lpc, c_n = pol.control(env, c_raw)
        phi0 = env.phi
        out = env.advance(u)
        disc = gamma ** out["k"]
        st.add(out, p, env.events)
        if store:
            B["c"][t], B["c_raw"][t], B["u"][t], B["lpc"][t], B["mc"][t] = c_n, c_raw, u, lpc, task
            if pol.learned_sched:
                B["s"][t], B["s_raw"][t], B["d"][t], B["lps"][t], B["ms"][t] = s_n, s_raw, d, lps, task
            B["rew"][t] = out["succ"] - C * out["sent"] - lam * out["k"] + shp * (disc * env.phi - phi0)
            B["disc"][t] = disc
        if mon is not None:
            mon.on_slot(env, out)
    return st, B


# ======================================================================================================================
# Tracking: CSV logs, trace, live file for the dashboard, TensorBoard, AirSim drawing
# ======================================================================================================================
class Monitor:
    LOG_COLS = ["update", "slots", "wall_s", "slots_per_s", "J", "J_ema", "succ", "fail", "fail_out", "fail_timeout",
                "fail_collision", "tasks", "succ_rate", "sends_per_task", "deliv_per_task", "sends_per_slot",
                "tasks_per_slot", "task_slots", "send_prob", "lambda", "lr", "kl", "clipfrac", "ent_ctrl", "ent_sched",
                "value_loss", "policy_loss", "expl_var", "log_std", "epochs"]
    HIST_KEYS = ["slots", "J", "J_ema", "succ_rate", "sends_per_task", "deliv_per_task", "task_slots", "send_prob",
                 "lambda", "kl", "lr", "ent_ctrl", "fail_out", "fail_timeout", "fail_collision", "tasks"]

    def __init__(self, run_dir, cfg, env, label, viz=None, trace=True):
        self.dir, self.cfg, self.label, self.viz, self.env = run_dir, cfg, label, viz, env
        os.makedirs(run_dir, exist_ok=True)
        self.t0, self.gslot, self.upd, self._t_live = time.time(), 0, 0, 0.0
        self.f_tasks, self.w_tasks = _csv_open(os.path.join(run_dir, "tasks.csv"), [
            "env", "task", "outcome", "start_slot", "end_slot", "slots", "sends", "deliveries", "final_dist",
            "target", "start_pos"])
        self.f_trace = self.w_trace = None
        if trace and cfg.TRACE:
            D = env.D
            ax = ["x", "y", "z"][:D]
            self.f_trace, self.w_trace = _csv_open(os.path.join(run_dir, "trace.csv"), (
                ["slot", "mode"] + ax + ["v" + a for a in ax] + [a + "_hat" for a in ax] + ["g" + a for a in ax]
                + ["send", "outcome"] + ["u" + a for a in ax] + ["age", "task", "event"]))
        self.trace_rows = 0
        self.f_log = self.w_log = None
        self.traj = deque(maxlen=int(cfg.LIVE_TRAJ))
        self.hits0 = deque(maxlen=max(1, int(cfg.SHOW_HITS)))
        self.hist = {k: [] for k in self.HIST_KEYS}
        self.last_row = {}
        self.tot = dict(succ=0.0, fail=0.0, sends=0.0, deliv=0.0, slots=0.0)
        self.tb = None
        if cfg.TENSORBOARD and torch is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
            except Exception:
                self.tb = None
        if viz is not None:
            for i in range(env.n):
                viz.set_target(i, env.target[i])

    def on_slot(self, env, out):
        self.gslot += 1
        t = self.tot
        t["succ"] += float(out["succ"].sum())
        t["fail"] += float(out["fail"].sum())
        t["sends"] += float(out["sent"].sum())
        t["deliv"] += float(out["deliv"].sum())
        t["slots"] += float(out["k"].sum())
        ev0 = ""
        for kind, i, info in env.events:
            if kind == "task_end":
                self.w_tasks.writerow([info["env"], info["task"], info["outcome"], info["start_slot"], info["end_slot"],
                                       info["slots"], info["sends"], info["deliveries"], f"{info['final_dist']:.3f}",
                                       " ".join(f"{x:.2f}" for x in info["target"]),
                                       " ".join(f"{x:.2f}" for x in info["start_pos"])])
                if info["outcome"] == "success":
                    if self.viz is not None:
                        self.viz.hit(i, info["target"])
                    if i == 0:
                        self.hits0.append(info["target"])
                if i == 0:
                    ev0 = info["outcome"]
            elif kind == "target" and self.viz is not None:
                self.viz.set_target(i, info)
        p, xh, g, v = env.pos[0], env.xh[0], env.target[0], env.vel[0]
        oc = int(env.outcome[0])
        self.traj.append([float(p[0]), float(p[1]), float(xh[0]), float(xh[1]), oc, int(env.mode[0])])
        if self.w_trace is not None and self.trace_rows < int(self.cfg.TRACE_MAX_ROWS):
            self.w_trace.writerow([int(env.clock[0]), _MODES[int(env.mode[0])]] + [f"{x:.4f}" for x in p]
                                  + [f"{x:.4f}" for x in v] + [f"{x:.4f}" for x in xh] + [f"{x:.3f}" for x in g]
                                  + [int(out["sent"][0]), ("silent", "lost", "delivered", "-")[oc]]
                                  + [f"{x:.4f}" for x in env.last_u[0]] + [int(env.age[0]), int(env.task_id[0]), ev0])
            self.trace_rows += 1
        if self.viz is not None:
            self.viz.slot(env)
            self.viz.stats(self._onscreen())
            self.viz.camera(os.path.join(self.dir, "camera.png"))
        now = time.time()
        if now - self._t_live >= float(self.cfg.LIVE_PERIOD):
            self._t_live = now
            self.write_live()

    def _onscreen(self):
        t = self.tot
        J = (t["succ"] - float(self.cfg.COST_C) * t["sends"]) / max(t["slots"], 1.0)
        tasks = t["succ"] + t["fail"]
        return [("drone_jcc ", self.label), ("slots ", f"{int(t['slots'])}"),
                ("throughput J ", f"{J:.4f}"), ("tasks done ", f"{int(t['succ'])} of {int(tasks)}"),
                ("sends per task ", f"{t['sends'] / tasks:.2f}" if tasks else "-"),
                ("channel ", f"p_s {self.cfg.P_S:g}, C {self.cfg.COST_C:g}")]

    def on_update(self, row):
        self.upd += 1
        self.last_row = row
        if self.w_log is None:
            self.f_log, self.w_log = _csv_open(os.path.join(self.dir, "train_log.csv"), self.LOG_COLS)
        self.w_log.writerow([row.get(k, "") if not isinstance(row.get(k), float) else f"{row[k]:.6g}"
                             for k in self.LOG_COLS])
        self.f_log.flush()
        self.f_tasks.flush()
        if self.f_trace is not None:
            self.f_trace.flush()
        for k in self.HIST_KEYS:
            self.hist[k].append(row.get(k))
        if self.tb is not None:
            for k in self.LOG_COLS[4:]:
                if isinstance(row.get(k), (int, float)) and math.isfinite(row[k]):
                    self.tb.add_scalar(k, row[k], int(row["slots"]))
        self.write_live()

    def write_live(self):
        cfg, env = self.cfg, self.env
        t = self.tot
        tasks = t["succ"] + t["fail"]
        data = dict(run=self.label, backend=cfg.BACKEND, time=time.strftime("%H:%M:%S"),
                    wall=time.time() - self.t0, slots=t["slots"], update=self.upd,
                    geometry=dict(dim=env.D, safe_lo=cfg.safe_lo, safe_hi=cfg.safe_hi,
                                  target_half=float(cfg.TARGET_HALF), start=cfg.start, radius=float(cfg.HIT_RADIUS)),
                    env0=dict(traj=list(self.traj), target=env.target[0], hits=list(self.hits0),
                              mode=_MODES[int(env.mode[0])], age=int(env.age[0]), pos=env.pos[0], est=env.xh[0]),
                    totals=dict(t, J=(t["succ"] - float(cfg.COST_C) * t["sends"]) / max(t["slots"], 1.0),
                                tasks=tasks, sends_per_task=t["sends"] / tasks if tasks else None),
                    params=dict(P_S=cfg.P_S, COST_C=cfg.COST_C, HIT_RADIUS=cfg.HIT_RADIUS, DT=cfg.DT,
                                NOISE_STD=cfg.NOISE_STD, U_MAX=cfg.U_MAX),
                    last=self.last_row, history=self.hist,
                    camera=os.path.isfile(os.path.join(self.dir, "camera.png")))
        write_json(os.path.join(self.dir, "live.json"), data)

    def close(self):
        self.write_live()
        for f in (self.f_tasks, self.f_trace, self.f_log):
            if f is not None:
                f.close()
        if self.tb is not None:
            self.tb.close()


def launch_dashboard(path, cfg):
    if cfg.DASHBOARD == "none":
        return None
    port = _free_port(cfg.DASHBOARD_HOST, cfg.DASHBOARD_PORT)
    cmd = [sys.executable, _SCRIPT, "dashboard", path, "--mode", cfg.DASHBOARD, "--port", str(port),
           "--host", cfg.DASHBOARD_HOST, "--refresh", str(cfg.DASHBOARD_REFRESH)]
    if cfg.OPEN_BROWSER:
        cmd.append("--open")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        tqdm.write(f"[dashboard] could not start ({e})")
        return None
    if cfg.DASHBOARD == "web":
        tqdm.write(f"dashboard: http://{cfg.DASHBOARD_HOST}:{port}/   "
                   f"(again later: python {os.path.basename(_SCRIPT)} dashboard {path})")
    if not cfg.DASHBOARD_KEEP:
        import atexit
        atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc


# ======================================================================================================================
# Training
# ======================================================================================================================
def _row(upd, slots, t0, st, J_ema, lam, info):
    """One line of train_log.csv: the rollout's statistics, the update's, and the cumulative counters."""
    wall = time.time() - t0
    r = dict(st)
    r.update(info or {})
    r.update(update=upd, slots=slots, wall_s=wall, slots_per_s=slots / max(wall, 1e-9), J_ema=J_ema)
    r["lambda"] = lam
    return r


def train_phase(cfg, phase, run_dir, init_path=None, dashboard=True, backend=None):
    """Trains one phase; returns the path of its latest checkpoint."""
    _need_torch()
    if phase not in _PHASE_SCHED:
        raise SystemExit(f"unknown phase {phase!r}; use one of {list(_PHASE_SCHED)}")
    kind = _PHASE_SCHED[phase]
    torch.set_num_threads(max(1, int(cfg.TORCH_THREADS)))
    seed_all(int(cfg.SEED) + 7919 * list(_PHASE_SCHED).index(phase))
    os.makedirs(run_dir, exist_ok=True)
    N = cfg.n_envs
    seeds = [int(cfg.SEED) * 100003 + 31 * list(_PHASE_SCHED).index(phase) + i for i in range(N)]
    own_backend = backend is None
    if own_backend:
        backend = make_backend(cfg, N, seeds)
    mon = None
    try:
        env = DroneTaskEnv(cfg, backend, seeds)
        agent = Agent(env.dims(kind), cfg.DIM, kind, hidden=cfg.HIDDEN, init_log_std=cfg.INIT_LOG_STD,
                      p0=cfg.SCHED_INIT_SEND_PROB, obs_norm=cfg.OBS_NORM).to_device(_device(cfg))
        lam = float(cfg.PRICE_INIT) if cfg.PRICE_MODE != "none" else 0.0
        if init_path:
            lam0, what = init_from(agent, init_path)
            if cfg.PRICE_MODE == "adaptive" and lam0 is not None:
                lam = float(lam0)
            tqdm.write(f"[{phase}] {what} start from {init_path} (price {lam:.4f})")
        ppo = PPO(agent, cfg)
        pol = Policy(cfg, agent, sched="learned" if kind != "always" else "always")
        viz = AirSimViz(backend, cfg) if (cfg.BACKEND == "airsim" and cfg.DRAW) else None
        mon = Monitor(run_dir, cfg, env, phase, viz)
        write_json(os.path.join(run_dir, "config.json"), dict(phase=phase, init=init_path, version=_VERSION,
                                                              argv=sys.argv, started=_now(), params=cfg_dump(cfg)),
                   indent=1)
        if dashboard:
            launch_dashboard(run_dir, cfg)
        T = cfg.rollout
        per = T * N
        n_upd = max(1, math.ceil(cfg.total / per))
        tqdm.write(f"[{phase}] {cfg.BACKEND}, {N} env(s), {n_upd} updates of {per} slots; {pol.describe()}; "
                   f"p_s {cfg.P_S:g}, C {cfg.COST_C:g}; run folder {run_dir}")
        pbar = tqdm(total=n_upd * per, unit="slot", desc=phase, dynamic_ncols=True)
        t0, slots, J_ema, best = time.time(), 0, None, -float("inf")
        ck_latest = os.path.join(run_dir, "ckpt_latest.pt")
        upd = 0
        try:
            for upd in range(1, n_upd + 1):
                st, B = run_slots(env, pol, T, cfg, lam=lam, mon=mon, store=True)
                backend.pause(True)
                last_v = agent.value(agent.nv(env.critic_obs()))
                adv, ret = gae(B["rew"], B["disc"], B["val"], last_v, float(cfg.GAE_LAMBDA))
                info = ppo.update(B, adv, ret, progress=upd / n_upd)
                backend.pause(False)
                m = B["mc"] > 0
                agent.nc.update(B["c_raw"][m])
                agent.nv.update(B["v_raw"].reshape(-1, B["v_raw"].shape[-1]))
                if pol.learned_sched:
                    agent.ns.update(B["s_raw"][B["ms"] > 0])
                s = st.summary()
                slots += per
                J_ema = s["J"] if J_ema is None else 0.9 * J_ema + 0.1 * s["J"]
                if cfg.PRICE_MODE == "adaptive":
                    lam += float(cfg.PRICE_RATE) * (s["J"] - lam)
                row = _row(upd, slots, t0, s, J_ema, lam, info)
                mon.on_update(row)
                extra = dict(lam=lam, update=upd, slots=slots, J_ema=J_ema, phase=phase)
                save_ckpt(ck_latest, agent, cfg, **extra)
                if upd >= max(3, n_upd // 10) and J_ema > best:
                    best = J_ema
                    save_ckpt(os.path.join(run_dir, "ckpt_best.pt"), agent, cfg, **extra)
                if int(cfg.CKPT_EVERY) > 0 and upd % int(cfg.CKPT_EVERY) == 0:
                    save_ckpt(os.path.join(run_dir, f"ckpt_{upd:05d}.pt"), agent, cfg, **extra)
                pbar.update(per)
                pbar.set_postfix(J=f"{s['J']:.4f}", succ=f"{s['succ_rate']:.2f}", send=f"{s['sends_per_task']:.1f}",
                                 p=f"{s['send_prob']:.2f}", lam=f"{lam:.3f}", kl=f"{info['kl']:.3f}")
                if upd == 1 or upd % int(cfg.LOG_EVERY) == 0 or upd == n_upd:
                    sd = getattr(backend, "slot_dt", None)  # AirSim: slots whose sim time was more than 10% off DT
                    tqdm.write(f"[{phase}] update {upd}/{n_upd}  slots {slots}  J {s['J']:.4f} (ema {J_ema:.4f})  "
                               f"success {s['succ_rate']:.3f}  slots/task {s['task_slots']:.1f}  "
                               f"sends/task {s['sends_per_task']:.2f}  send prob {s['send_prob']:.2f}  "
                               f"fails out/time/coll {int(s['fail_out'])}/{int(s['fail_timeout'])}/"
                               f"{int(s['fail_collision'])}  price {lam:.4f}  kl {info['kl']:.4f}  lr {info['lr']:.1e}  "
                               f"std {math.exp(info['log_std']):.2f}  [{time.time() - t0:.0f}s]"
                               + (f"  short slots {sd[3]}/{sd[0]}" if sd and sd[0] else ""))
        except KeyboardInterrupt:
            tqdm.write(f"[{phase}] interrupted at update {upd}; the latest checkpoint is {ck_latest}")
            raise
        finally:
            pbar.close()
        tqdm.write(f"[{phase}] done in {time.time() - t0:.0f}s; final J_ema {J_ema:.4f}; checkpoint {ck_latest}")
        return ck_latest
    finally:
        if mon is not None:
            mon.close()
        if own_backend:
            backend.close()


def _phase_init(cfg, phase, init):
    if init:
        return None if init == "none" else init
    if phase == "always_send":
        return None
    if cfg.INIT_FROM == "none":
        return None
    if cfg.INIT_FROM != "auto":
        return cfg.INIT_FROM
    p = _find_latest_run(cfg.RUN_ROOT, "always_send")
    if p is None:
        tqdm.write(f"[{phase}] no always_send run found under {cfg.RUN_ROOT}; training from scratch")
    elif cfg.CKPT_FOR_INIT == "best":
        p = _ckpt_of(os.path.dirname(p), "best")
    return p


def run_all(cfg, session_dir, dashboard=True, skip_eval=False, init_session=None):
    """always_send, then joint_push and joint_pull from it, then the evaluation and the plots. With init_session
    (e.g. a surrogate session), every phase starts from that session's checkpoint of the same phase."""
    os.makedirs(session_dir, exist_ok=True)
    tqdm.write(f"session folder: {session_dir}")
    if dashboard:
        launch_dashboard(session_dir, cfg)
    backend = None
    if cfg.BACKEND == "airsim":  # one connection and one takeoff for the whole session
        backend = AirSimBackend(cfg, cfg.n_envs)
    try:
        dirs = {p: os.path.join(session_dir, p) for p in _PHASE_SCHED}
        pre = {}
        if init_session:
            for p in _PHASE_SCHED:
                pre[p] = _ckpt_of(os.path.join(init_session, p), cfg.CKPT_FOR_INIT)
                if pre[p] is None:
                    raise SystemExit(f"{init_session} has no {p} checkpoint")
        ck_a = train_phase(cfg, "always_send", dirs["always_send"], init_path=pre.get("always_send"),
                           dashboard=False, backend=backend)
        ck_init = _ckpt_of(dirs["always_send"], cfg.CKPT_FOR_INIT) or ck_a
        ck_push = train_phase(cfg, "joint_push", dirs["joint_push"], init_path=pre.get("joint_push", ck_init),
                              dashboard=False, backend=backend)
        ck_pull = train_phase(cfg, "joint_pull", dirs["joint_pull"], init_path=pre.get("joint_pull", ck_init),
                              dashboard=False, backend=backend)
        if not skip_eval:
            evaluate(cfg, default_entries(cfg, ck_init, ck_push, ck_pull), os.path.join(session_dir, "eval"),
                     backend=backend)
    finally:
        if backend is not None:
            backend.close()
    cmd_plot([session_dir], os.path.join(session_dir, "plots"))
    tqdm.write(f"session done: {session_dir}")


# ======================================================================================================================
# Evaluation
# ======================================================================================================================
def default_entries(cfg, ck_a, ck_push, ck_pull):
    E = []
    if ck_a:
        E.append(dict(label="always send (learned controller)", ckpt=ck_a, sched="always"))
    for ck, name in ((ck_push, "push"), (ck_pull, "pull")):
        if ck:
            E.append(dict(label=f"joint {name} (learned)", ckpt=ck, sched="learned"))
            E.append(dict(label=f"joint {name} (learned, sampled actions)", ckpt=ck, sched="learned",
                          deterministic=False))
    if ck_a:
        for K in cfg.PERIODS:
            E.append(dict(label=f"periodic, every {K} slots", ckpt=ck_a, sched="periodic", param=K))
        for th in cfg.ERR_THRESHOLDS:
            E.append(dict(label=f"push rule: error > {th:g} m", ckpt=ck_a, sched="err", param=th))
        for r in cfg.NEAR_RADII:
            E.append(dict(label=f"push rule: within {r:g} m of target", ckpt=ck_a, sched="near", param=r))
        for a in cfg.AGE_THRESHOLDS:
            E.append(dict(label=f"pull rule: age >= {a}", ckpt=ck_a, sched="age", param=a))
        E.append(dict(label="never send", ckpt=ck_a, sched="never"))
        E.append(dict(label="reference: free state every slot (C = 0, p_s = 1)", ckpt=ck_a, sched="always",
                      overrides=dict(P_S=1.0, COST_C=0.0), reference=True))
    E.append(dict(label="PD controller, always send", sched="always", controller="pd"))
    return E


def evaluate(cfg, entries, out_dir, backend=None, dashboard=False):
    """Every policy runs on the same seeds (common random numbers); one replication = one environment (surrogate)
    or one drone in one round (AirSim). Prints each policy as it finishes."""
    os.makedirs(out_dir, exist_ok=True)
    R, T = cfg.eval_reps, cfg.eval_slots
    air = cfg.BACKEND == "airsim"
    own = air and backend is None
    if own:
        backend = AirSimBackend(cfg, cfg.n_envs)
    if dashboard:
        launch_dashboard(out_dir, cfg)
    results = []
    tqdm.write(f"[eval] {len(entries)} policies, {R} replications x {T} slots each, "
               f"{'AirSim' if air else 'surrogate'}; folder {out_dir}")
    try:
        for e in tqdm(entries, desc="eval", unit="policy", dynamic_ncols=True):
            ce = make_cfg(base=cfg, **e.get("overrides", {}))
            agent = agent_from_ckpt(e["ckpt"], ce)[0] if e.get("ckpt") else None
            pol = Policy(ce, agent, sched=e.get("sched", "learned"), param=e.get("param"),
                         controller=e.get("controller", "agent"),
                         deterministic=e.get("deterministic", ce.EVAL_DETERMINISTIC))
            per, t1, probe = [], time.time(), []
            if air:
                viz = AirSimViz(backend, ce) if ce.DRAW else None
                for r in range(R):
                    seeds = [int(ce.EVAL_SEED) + 1000 * r + i for i in range(backend.n)]
                    env = DroneTaskEnv(ce, backend, seeds)
                    mon = Monitor(os.path.join(out_dir, "live"), ce, env, f"eval: {e['label']}", viz, trace=False)
                    st, _ = run_slots(env, pol, T, ce, mon=mon, probe=probe)
                    mon.close()
                    per += st.per_env()
            else:
                seeds = [int(ce.EVAL_SEED) + 1000 * r for r in range(R)]
                env = DroneTaskEnv(ce, SurrogateBackend(ce, R, seeds), seeds)
                st, _ = run_slots(env, pol, T, ce, probe=probe)
                per = st.per_env()
            if probe:
                pr = np.concatenate(probe, 0)
                if len(pr) > 300_000:
                    pr = pr[np.random.default_rng(0).choice(len(pr), 300_000, replace=False)]
                np.savez_compressed(os.path.join(out_dir, f"probe_{len(results):02d}.npz"), probe=pr.astype(np.float32),
                                    label=e["label"], cols=np.array(_PROBE_COLS))
            J = [p["J"] for p in per]
            m, h = mean_ci(J)
            S = sum(p["succ"] for p in per)
            tasks = sum(p["tasks"] for p in per)
            res = dict(label=e["label"], sched=e.get("sched"), param=e.get("param"),
                       controller=e.get("controller", "agent"), ckpt=e.get("ckpt"), overrides=e.get("overrides", {}),
                       reference=bool(e.get("reference", False)), J=m, J_ci=h, J_reps=J,
                       tasks_per_slot=mean_ci([p["tasks_per_slot"] for p in per])[0],
                       sends_per_slot=mean_ci([p["sends_per_slot"] for p in per])[0],
                       sends_per_task=sum(p["sends"] for p in per) / max(tasks, 1),
                       deliv_per_task=sum(p["deliv"] for p in per) / max(tasks, 1),
                       succ_rate=S / max(tasks, 1), task_slots=sum(p["tslots"] for p in per) / max(S, 1),
                       reps=len(per), slots_per_rep=T, seconds=time.time() - t1)
            results.append(res)
            tqdm.write(f"  {res['label']:<52s} J {m:.4f} +- {h:.4f}   success {res['succ_rate']:.3f}   "
                       f"sends/task {res['sends_per_task']:.2f}   slots/task {res['task_slots']:.1f}")
            write_json(os.path.join(out_dir, "eval.json"), dict(results=results, params=cfg_dump(cfg)), indent=1)
    finally:
        if own:
            backend.close()
    ref = results[0] if results else None
    if ref is not None:
        for r in results:
            if len(r["J_reps"]) == len(ref["J_reps"]):
                r["diff_vs_first"], r["diff_ci"] = mean_ci(np.asarray(r["J_reps"]) - np.asarray(ref["J_reps"]))
    write_json(os.path.join(out_dir, "eval.json"), dict(results=results, params=cfg_dump(cfg)), indent=1)
    with open(os.path.join(out_dir, "eval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["label", "J", "J_ci", "diff_vs_first", "diff_ci", "succ_rate", "tasks_per_slot", "sends_per_task",
                "deliv_per_task", "sends_per_slot", "task_slots", "reps"]
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])
    if ref is not None:
        tqdm.write(f"\n[eval] throughput J with 95% CI; difference to '{ref['label']}' on common random numbers:")
        for r in sorted(results, key=lambda x: -x["J"]):
            dv = r.get("diff_vs_first")
            ds = f"{dv:+.4f} +- {r['diff_ci']:.4f}" if dv is not None and math.isfinite(r.get("diff_ci", 0) or 0) \
                else "-"
            tqdm.write(f"  {r['label']:<52s} {r['J']:.4f} +- {r['J_ci']:.4f}   diff {ds}")
        tqdm.write(f"[eval] written to {out_dir}")
    return results


# ======================================================================================================================
# Watching
# ======================================================================================================================
def cmd_watch(cfg, ckpt=None, sched=None, param=None, controller=None, slots=0, stochastic=False, dashboard=True,
              name=None):
    agent = None
    if ckpt:
        agent, _ = agent_from_ckpt(ckpt, cfg)
    controller = controller or ("agent" if agent is not None else "pd")
    sched = sched or ("learned" if (agent is not None and agent.sched is not None) else "always")
    n = cfg.n_envs if cfg.BACKEND == "airsim" else 1
    seeds = [int(cfg.SEED) * 7 + 3 + i for i in range(n)]
    backend = make_backend(cfg, n, seeds)
    run_dir = os.path.join(cfg.RUN_ROOT, name or cfg.RUN_NAME or f"watch_{_now()}")
    try:
        env = DroneTaskEnv(cfg, backend, seeds)
        pol = Policy(cfg, agent, sched=sched, param=param, controller=controller, deterministic=not stochastic)
        viz = AirSimViz(backend, cfg) if (cfg.BACKEND == "airsim" and cfg.DRAW) else None
        mon = Monitor(run_dir, cfg, env, f"watch: {pol.describe()}", viz)
        write_json(os.path.join(run_dir, "config.json"), dict(phase="watch", ckpt=ckpt, policy=pol.describe(),
                                                              params=cfg_dump(cfg)), indent=1)
        if dashboard:
            launch_dashboard(run_dir, cfg)
        tqdm.write(f"[watch] {pol.describe()}; {cfg.BACKEND}; Ctrl+C to stop; folder {run_dir}")
        chunk, done, t0 = 20, 0, time.time()
        total = Stats(n, cfg.COST_C)
        pbar = tqdm(total=slots or None, unit="slot", desc="watch", dynamic_ncols=True)
        try:
            while slots <= 0 or done < slots:
                k = chunk if slots <= 0 else min(chunk, slots - done)
                t1 = time.time()
                if cfg.WATCH_REALTIME:
                    st = Stats(n, cfg.COST_C)
                    for _ in range(k):
                        ts = time.time()
                        st.merge(run_slots(env, pol, 1, cfg, mon=mon)[0])
                        time.sleep(max(0.0, float(cfg.DT) - (time.time() - ts)))
                else:
                    st, _ = run_slots(env, pol, k, cfg, mon=mon)
                total.merge(st)
                done += k
                s, tot = st.summary(), total.summary()
                row = dict(s)
                row.update(update=mon.upd + 1, slots=done * n, wall_s=time.time() - t0,
                           slots_per_s=k * n / max(time.time() - t1, 1e-9), J_ema=tot["J"])
                mon.on_update(row)
                pbar.update(k)
                pbar.set_postfix(J=f"{tot['J']:.4f}", tasks=int(tot["succ"]), fails=int(tot["fail"]),
                                 send=f"{tot['sends_per_task']:.2f}" if tot["tasks"] else "-")
        except KeyboardInterrupt:
            pass
        finally:
            pbar.close()
            mon.close()
        s = total.summary()
        tqdm.write(f"[watch] {done} slots: J {s['J']:.4f}, {int(s['succ'])} successes, {int(s['fail'])} failures, "
                   f"sends/task {s['sends_per_task']:.2f}; logs in {run_dir}")
    finally:
        backend.close()


# ======================================================================================================================
# Dashboard
# ======================================================================================================================
def _dash_axes(fig):
    gs = fig.add_gridspec(3, 4, width_ratios=[1.25, 1.25, 1, 1])
    return {"map": fig.add_subplot(gs[:, :2]), "J": fig.add_subplot(gs[0, 2]), "send": fig.add_subplot(gs[0, 3]),
            "succ": fig.add_subplot(gs[1, 2]), "len": fig.add_subplot(gs[1, 3]), "pol": fig.add_subplot(gs[2, 2]),
            "txt": fig.add_subplot(gs[2, 3])}


def _dash_draw(fig, axs, data):
    from matplotlib.patches import Circle, Rectangle
    for ax in axs.values():
        ax.clear()
    if not data:
        axs["map"].text(0.5, 0.5, "waiting for live.json ...", ha="center", va="center", transform=axs["map"].transAxes)
        return
    g, e = data["geometry"], data["env0"]
    R = g["radius"]
    lo, hi, st, th = g["safe_lo"], g["safe_hi"], g["start"], g["target_half"]
    ax = axs["map"]  # top view: east (y) to the right, north (x) up
    ax.plot([lo[1], hi[1], hi[1], lo[1], lo[1]], [lo[0], lo[0], hi[0], hi[0], lo[0]], color="0.4", ls="--", lw=1,
            label="safe box")
    ax.add_patch(Rectangle((st[1] - th, st[0] - th), 2 * th, 2 * th, fill=False, ec="0.85", lw=1))
    for h in e["hits"]:
        ax.add_patch(Circle((h[1], h[0]), R, color="tab:green", alpha=0.45))
    tg = e["target"]
    ax.add_patch(Circle((tg[1], tg[0]), R, color="red", alpha=0.65))
    ax.plot([tg[1]], [tg[0]], "r+", ms=10)
    tr = np.asarray(e["traj"], float) if e["traj"] else None
    if tr is not None and len(tr):
        ax.plot(tr[:, 1], tr[:, 0], color="tab:blue", lw=1.2, label="drone")
        ax.plot(tr[:, 3], tr[:, 2], ".", color="orange", ms=2.5, label="controller's estimate")
        dl, ls = tr[:, 4] == _DELIV, tr[:, 4] == _LOST
        ax.plot(tr[dl, 1], tr[dl, 0], "o", mfc="none", mec="c", ms=5, label="state delivered")
        ax.plot(tr[ls, 1], tr[ls, 0], "x", color="crimson", ms=5, label="lost")
        ax.plot([tr[-1, 1]], [tr[-1, 0]], "k^", ms=9)
        ax.plot([tr[-1, 3]], [tr[-1, 2]], "*", color="orange", ms=11, mec="k")
    ax.set_aspect("equal")
    ax.set_xlim(lo[1] - 1, hi[1] + 1)
    ax.set_ylim(lo[0] - 1, hi[0] + 1)
    ax.set_xlabel("east, y (m)")
    ax.set_ylabel("north, x (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax.set_title(f"{data['run']}   drone 1: {e['mode']}, age {e['age']}", fontsize=10)
    H = data.get("history") or {}
    xs = H.get("slots") or []

    def series(ax_, key, label, **kw):
        ys = H.get(key) or []
        pts = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float)) and y is not None and math.isfinite(y)]
        if pts:
            ax_.plot([p[0] for p in pts], [p[1] for p in pts], label=label, **kw)

    series(axs["J"], "J", "J per update", alpha=0.35)
    series(axs["J"], "J_ema", "J (EMA)")
    series(axs["J"], "lambda", "price", ls="--")
    axs["J"].set_title("throughput (tasks - C sends) / slot", fontsize=9)
    series(axs["send"], "sends_per_task", "sends")
    series(axs["send"], "deliv_per_task", "deliveries")
    axs["send"].set_title("per task", fontsize=9)
    series(axs["succ"], "succ_rate", "success rate")
    axs["succ"].set_ylim(-0.02, 1.02)
    axs["succ"].set_title("success rate of tasks", fontsize=9)
    series(axs["len"], "task_slots", "slots per successful task")
    axs["len"].set_title("slots per successful task", fontsize=9)
    series(axs["pol"], "send_prob", "send probability")
    axs["pol"].set_title("scheduler: send probability in task slots", fontsize=9)
    axs["pol"].set_ylim(-0.02, 1.02)
    for k in ("J", "send", "pol"):
        if axs[k].lines:
            axs[k].legend(fontsize=7)
    for k in ("J", "send", "succ", "len", "pol"):
        axs[k].tick_params(labelsize=7)
        axs[k].set_xlabel("slots", fontsize=7)
    tot, last, prm = data.get("totals", {}), data.get("last") or {}, data.get("params", {})

    def f_(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) and math.isfinite(v) else "-"

    lines = [f"run      {data['run']}", f"backend  {data['backend']}   {data['time']}",
             f"slots    {int(tot.get('slots') or 0)}   update {data.get('update', 0)}",
             f"J (all)  {f_(tot.get('J'), '.4f')}",
             f"tasks    {int(tot.get('succ') or 0)} done of {int(tot.get('tasks') or 0)}",
             f"sends    {f_(tot.get('sends_per_task'), '.2f')} per task",
             f"p_s {prm.get('P_S')}   C {prm.get('COST_C')}   R {prm.get('HIT_RADIUS')} m"]
    if last:
        lines += [f"last update: J {f_(last.get('J'), '.4f')}, success {f_(last.get('succ_rate'), '.3f')}",
                  f"  price {f_(last.get('lambda'), '.4f')}  kl {f_(last.get('kl'), '.4f')}  "
                  f"lr {f_(last.get('lr'), '.1e')}"]
    axs["txt"].axis("off")
    axs["txt"].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8,
                    transform=axs["txt"].transAxes)


def _find_live(path):
    if os.path.isfile(path):
        return path
    p = os.path.join(path, "live.json")
    if os.path.isfile(p):
        return p
    cands = glob.glob(os.path.join(path, "**", "live.json"), recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None


_DASH_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>drone_jcc dashboard</title>
<style>body{margin:0;background:#16181c;color:#ddd;font:13px sans-serif}#w{display:flex;flex-wrap:wrap;gap:8px;
padding:8px}img{max-width:100%;height:auto;background:#fff}#c{max-width:480px}</style></head><body><div id="w">
<img id="d" src="dash.png"><img id="c" src="camera.png" onerror="this.style.display='none'"></div><script>
setInterval(function(){var t=Date.now();document.getElementById('d').src='dash.png?'+t;
var c=document.getElementById('c');c.style.display='';c.src='camera.png?'+t;},REFRESH);</script></body></html>"""


def cmd_dashboard(path, mode="web", port=8765, host="127.0.0.1", open_browser=False, refresh=1.0, once=None):
    import matplotlib
    if mode == "web" or once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 8.6))
    axs = _dash_axes(fig)

    def render():
        f = _find_live(path)
        data = read_json(f) if f else None
        _dash_draw(fig, axs, data)
        fig.tight_layout()
        return f

    if once:  # render a single image (used by selftest)
        render()
        fig.savefig(once, dpi=72)
        return once
    if mode == "window":
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, lambda _i: render(), interval=int(1000 * refresh), cache_frame_data=False)
        plt.show()
        return anim
    state = {"png": b""}

    def loop():
        while True:
            try:
                render()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=72)
                state["png"] = buf.getvalue()
            except Exception as ex:  # keep serving
                print(f"[dashboard] {type(ex).__name__}: {ex}", flush=True)
            time.sleep(refresh)

    threading.Thread(target=loop, daemon=True).start()
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/dash.png":
                self._send(state["png"], "image/png")
            elif p in ("/camera.png", "/live.json"):
                f = _find_live(path)
                fp = os.path.join(os.path.dirname(f), p[1:]) if f else None
                try:
                    with open(fp, "rb") as fh:
                        self._send(fh.read(), "image/png" if p.endswith("png") else "application/json")
                except (OSError, TypeError):
                    self._send(b"", "text/plain", 404)
            else:
                self._send(_DASH_HTML.replace("REFRESH", str(int(1000 * refresh))).encode(), "text/html")

    srv = None
    for pp in range(int(port), int(port) + 20):
        try:
            srv = ThreadingHTTPServer((host, pp), Handler)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port from {port}")
    url = f"http://{host}:{srv.server_address[1]}/"
    print(f"dashboard for {path}: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ======================================================================================================================
# Plots
# ======================================================================================================================
def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                out.append(float("nan"))
        return np.array(out)
    return col, len(rows)


def cmd_plot(paths, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = out_dir or os.path.join(paths[0] if os.path.isdir(paths[0]) else os.path.dirname(paths[0]), "plots")
    os.makedirs(out_dir, exist_ok=True)
    logs, evals = [], []
    for p in paths:
        if os.path.isfile(p):
            (logs if p.endswith(".csv") else evals).append(p)
        else:
            logs += glob.glob(os.path.join(p, "**", "train_log.csv"), recursive=True)
            evals += glob.glob(os.path.join(p, "**", "eval.json"), recursive=True)
    made = []
    logs = [f for f in sorted(set(logs)) if "watch" not in os.path.basename(os.path.dirname(f))]
    if logs:
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        tops = {(0, 1): [], (1, 1): []}
        for f in logs:
            col, n = _read_csv(f)
            if n == 0:
                continue
            name = os.path.basename(os.path.dirname(f))
            x = col("slots")
            ax[0, 0].plot(x, col("J_ema"), label=name)
            ax[0, 1].plot(x, col("sends_per_task"), label=name)
            ax[1, 0].plot(x, col("succ_rate"), label=name)
            ax[1, 1].plot(x, col("task_slots"), label=name)
            for key, c in (((0, 1), "sends_per_task"), ((1, 1), "task_slots")):
                v = col(c)
                v = v[np.isfinite(v)]
                if len(v):  # the first updates of a run from scratch would squash the rest of the curve
                    tops[key].append(np.percentile(v[len(v) // 10:], 99) if len(v) > 10 else v.max())
        for key, t in tops.items():
            if t:
                ax[key].set_ylim(0, 1.15 * max(t))
        for a, t in zip(ax.flat, ["throughput J (EMA)", "sends per task", "success rate", "slots per successful task"]):
            a.set_title(t)
            a.set_xlabel("training slots")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        f = os.path.join(out_dir, "learning_curves.png")
        fig.savefig(f, dpi=120)
        plt.close(fig)
        made.append(f)
    for ef in sorted(set(evals)):
        d = read_json(ef)
        res = [r for r in (d or {}).get("results", []) if r.get("J") is not None]
        if not res:
            continue
        res = sorted(res, key=lambda r: r["J"])
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(res) + 1.5))
        y = np.arange(len(res))
        col_ = ["tab:red" if "joint" in r["label"] else ("0.6" if r.get("reference") else "tab:blue") for r in res]
        ax.barh(y, [r["J"] for r in res], xerr=[r["J_ci"] if r["J_ci"] == r["J_ci"] else 0 for r in res],
                color=col_, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in res], fontsize=8)
        ax.set_xlabel("throughput J = (tasks - C sends) / slot, 95% CI")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        f = os.path.join(out_dir, "eval_" + os.path.basename(os.path.dirname(ef)) + ".png")
        fig.savefig(f, dpi=120)
        plt.close(fig)
        made.append(f)
    for ef in sorted(set(evals)):
        made += _plot_sched_maps(os.path.dirname(ef), out_dir, plt)
    for f in made:
        tqdm.write(f"[plot] {f}")
    return made


def _plot_sched_maps(eval_dir, out_dir, plt):
    """Where each scheduler sends. Top: the send frequency against the true distance to the target and the estimation
    error |x - x_hat| (what the drone sees). Bottom: against the estimate's distance to the target and the age of the
    last delivered state (what the controller sees). Learned schedulers, and the best rule of each kind."""
    d = read_json(os.path.join(eval_dir, "eval.json")) or {}
    J = {r["label"]: r.get("J") for r in d.get("results", [])}
    probes = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "probe_*.npz"))):
        z = np.load(f, allow_pickle=False)
        probes[str(z["label"])] = z["probe"]
    pick = [lab for lab in probes if lab.startswith("joint") and "sampled" not in lab]
    for prefix in ("push rule: error", "push rule: within", "pull rule: age", "periodic"):
        cands = [lab for lab in probes if lab.startswith(prefix) and J.get(lab) is not None]
        if cands:
            pick.append(max(cands, key=lambda lab: J[lab]))
    if not pick:
        return []
    fig, axs = plt.subplots(2, len(pick), figsize=(3.3 * len(pick) + 1.0, 6.4), squeeze=False)
    im = None
    for j, lab in enumerate(pick):
        pr = probes[lab]
        dist, err, dest, age, send = pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3], pr[:, 6]
        rows = [(dist, err, np.linspace(0, 12, 25), np.linspace(0, 2, 21), "distance to target (m)",
                 "estimation error |x - x_hat| (m)"),
                (dest, age, np.linspace(0, 12, 25), np.arange(-0.5, 11.0, 1.0), "estimate's distance to target (m)",
                 "age of the last delivered state (slots)")]
        for row, (xv, yv, xb, yb, xl, yl) in enumerate(rows):
            ax = axs[row, j]
            num, _, _ = np.histogram2d(xv, yv, bins=[xb, yb], weights=send)
            den, _, _ = np.histogram2d(xv, yv, bins=[xb, yb])
            frac = np.where(den >= 20, num / np.maximum(den, 1), np.nan)
            im = ax.imshow(frac.T, origin="lower", extent=[xb[0], xb[-1], yb[0], yb[-1]], aspect="auto", vmin=0,
                           vmax=1, cmap="viridis", interpolation="nearest")
            ax.set_xlabel(xl, fontsize=8)
            if j == 0:
                ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                jv = J.get(lab)
                ax.set_title(lab + (f"\nJ = {jv:.4f}" if jv is not None else ""), fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.7, label="fraction of task slots with a transmission")
    f = os.path.join(out_dir, "scheduler_maps_" + os.path.basename(eval_dir) + ".png")
    fig.savefig(f, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [f]


# ======================================================================================================================
# AirSim settings and preflight check
# ======================================================================================================================
def make_settings(cfg):
    veh = {}
    for i in range(int(cfg.NUM_DRONES)):
        veh[f"{cfg.VEHICLE_PREFIX}{i + 1}"] = {"VehicleType": "SimpleFlight", "X": 0.0,
                                               "Y": i * float(cfg.ARENA_SPACING), "Z": 0.0, "Yaw": 0.0,
                                               "AllowAPIAlways": True}
    return {"SettingsVersion": 1.2, "SimMode": "Multirotor", "ClockSpeed": float(cfg.CLOCK_SPEED),
            "ClockType": "SteppableClock", "ViewMode": cfg.VIEW_MODE, "Vehicles": veh}


def cmd_settings(cfg, write=None):
    s = make_settings(cfg)
    text = json.dumps(s, indent=2)
    print(text)
    if write:
        path = os.path.join(os.path.expanduser("~"), "Documents", "AirSim", "settings.json") if write == "auto" \
            else write
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.isfile(path):
            bak = path + f".bak_{_now()}"
            os.replace(path, bak)
            print(f"previous settings saved as {bak}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"written to {path}; restart AirSim to use it")


def cmd_check(cfg):
    """Connection, vehicles, lockstep timing, velocity response, markers."""
    print(f"connecting to AirSim at {cfg.AIRSIM_IP}:{cfg.AIRSIM_PORT} ...", flush=True)
    b = AirSimBackend(cfg, cfg.n_envs)
    c, nm = b.c, b.names[0]
    print(f"client: {b.client_kind}; vehicles: {b.names}; start offsets: {b.offsets.tolist()}")
    try:
        s = json.loads(c.getSettingsString())
        print(f"settings: ClockSpeed {s.get('ClockSpeed', 1)}, ClockType {s.get('ClockType', '(default)')}, "
              f"ViewMode {s.get('ViewMode', '(default)')}")
    except Exception:
        pass
    t0 = time.time()
    for _ in range(50):
        c.simIsPause()
    print(f"RPC round trip {1000 * (time.time() - t0) / 50:.2f} ms ({b.client_kind} client; a slot makes "
          f"{3 * b.n + 1} calls plus the drawing)")
    try:
        n = b.n
        env = DroneTaskEnv(cfg, b, [1 + i for i in range(n)])
        print("lockstep timing (20 slots, alternating commands) ...", flush=True)
        ts, wall = [], []
        for k in range(20):
            t0 = time.time()
            s0 = c.getMultirotorState(nm).timestamp
            sign = 1.0 if (k // 5) % 2 == 0 else -1.0
            b.step(np.tile([sign * 1.0] + [0.0] * (b.D - 1), (n, 1)))
            ts.append((c.getMultirotorState(nm).timestamp - s0) / 1e9)
            wall.append(time.time() - t0)
        print(f"  sim time per slot {np.mean(ts):.4f} s (DT {cfg.DT}), wall time per slot {np.mean(wall):.4f} s, "
              f"{float(cfg.DT) / max(np.mean(wall), 1e-9):.1f} slots of sim time per wall slot")
        print(f"  sim time per slot: shortest {min(ts):.4f} s, longest {max(ts):.4f} s")
        if abs(np.mean(ts) - float(cfg.DT)) > 0.25 * float(cfg.DT):
            print("  [warning] the sim time per slot differs from DT: try STEP_MODE=realtime or check ClockType")
        print("velocity response (2 m/s step along x) ...", flush=True)
        b.teleport(0, cfg.start)
        for _ in range(8):
            b.step(np.zeros((n, b.D)))
        vs, cmd = [], np.zeros((n, b.D))
        cmd[0, 0] = 2.0
        for _ in range(16):
            vs.append(b.step(cmd)[1][0, 0])
        vs = np.array(vs)
        taus = np.linspace(0.05, 2.0, 196)
        errs = [np.sum((vs - 2.0 * (1 - np.exp(-float(cfg.DT) * (np.arange(16) + 1) / tt))) ** 2) for tt in taus]
        tau = float(taus[int(np.argmin(errs))])
        print(f"  velocity after 1, 2, 4, 8 slots: {vs[0]:.2f}, {vs[1]:.2f}, {vs[3]:.2f}, {vs[7]:.2f} m/s")
        print("  velocity after each of the 16 slots: " + ", ".join(f"{v:.2f}" for v in vs) + " m/s")
        print(f"  first-order fit: time constant {tau:.2f} s  ->  use --set PRED_TAU={tau:.2f} (and TAU for the "
              "surrogate)")
        res = vs - 2.0 * (1 - np.exp(-float(cfg.DT) * (np.arange(16) + 1) / tau))
        print(f"  fit error: RMS {np.sqrt(np.mean(res ** 2)):.3f} m/s, largest {np.max(np.abs(res)):.3f} m/s")
        for _ in range(12):
            b.step(np.zeros((n, b.D)))
        print("markers: a red target ring 4 m north of drone 1 and a green one 4 m south ...", flush=True)
        viz = AirSimViz(b, cfg)
        g = cfg.start.copy()
        g[0] += 4.0
        viz.set_target(0, g)
        h = cfg.start.copy()
        h[0] -= 4.0
        viz.hit(0, h)
        viz.redraw()
        c.simPrintLogMessage("drone_jcc check ", "red = current target, green = hit target", 0)
        print("  look at the AirSim window. The check is done; the drone hovers.")
        del env
    finally:
        b.close()


# ======================================================================================================================
# Self-test and command line
# ======================================================================================================================
def cmd_selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="drone_jcc_selftest_")
    sets = ["BACKEND='surrogate'", "TOTAL_SLOTS=24000", "N_ENVS=16", "ROLLOUT_SLOTS=128", "EVAL_SLOTS=2000",
            "EVAL_REPS=4", "DASHBOARD='none'", f"RUN_ROOT={root!r}", "TENSORBOARD=False", "PERIODS=(2,)",
            "ERR_THRESHOLDS=(0.5,)", "NEAR_RADII=(2.0,)", "AGE_THRESHOLDS=(2,)", "LOG_EVERY=4"]
    cfg = make_cfg(sets)
    t0 = time.time()
    session = os.path.join(root, "session")
    run_all(cfg, session, dashboard=False)
    png = cmd_dashboard(session, once=os.path.join(root, "dashboard_frame.png"))
    need = [os.path.join(session, p, f) for p in _PHASE_SCHED for f in ("ckpt_latest.pt", "train_log.csv",
                                                                          "tasks.csv", "trace.csv", "live.json")]
    need += [os.path.join(session, "eval", "eval.json"), os.path.join(session, "plots", "learning_curves.png"), png]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"selftest failed: missing {missing}")
    print(f"selftest OK in {time.time() - t0:.0f}s; files in {root}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a parameter")
        p.add_argument("--backend", choices=["airsim", "surrogate"], help="shortcut for --set BACKEND=...")
        p.add_argument("--run", help="run or session name (default: date and time)")

    p = sub.add_parser("train", help="train one phase")
    common(p)
    p.add_argument("--phase", choices=list(_PHASE_SCHED))
    p.add_argument("--init", help="checkpoint to start from, or none (joint phases default to INIT_FROM)")
    p.add_argument("--no-dashboard", action="store_true")
    p = sub.add_parser("all", help="always_send, joint_push, joint_pull, evaluation and plots")
    common(p)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--init-session", help="start every phase from this session's checkpoints (e.g. a surrogate "
                                          "session, to continue in AirSim)")
    p = sub.add_parser("eval", help="compare policies on common random numbers")
    common(p)
    p.add_argument("--session", help="session folder from `all`: evaluates its three phases and the baselines")
    p.add_argument("--ckpt", action="append", default=[], help="checkpoint(s) to evaluate")
    p.add_argument("--sched", default="learned", help=f"scheduler for --ckpt: learned or one of {_RULES}")
    p.add_argument("--param", type=float, help="parameter of a rule scheduler")
    p.add_argument("--out", help="output folder")
    p.add_argument("--dashboard", action="store_true")
    p = sub.add_parser("watch", help="fly a policy and watch it")
    common(p)
    p.add_argument("--ckpt", help="checkpoint (none: the PD controller)")
    p.add_argument("--sched", help=f"learned or one of {_RULES}")
    p.add_argument("--param", type=float)
    p.add_argument("--controller", choices=["agent", "pd"])
    p.add_argument("--slots", type=int, default=0, help="0 = until Ctrl+C")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--no-dashboard", action="store_true")
    p = sub.add_parser("dashboard", help="live dashboard of a run or session folder")
    p.add_argument("path")
    p.add_argument("--mode", choices=["web", "window"], default=_PARAMS["DASHBOARD"]
                   if _PARAMS["DASHBOARD"] != "none" else "web")
    p.add_argument("--port", type=int, default=_PARAMS["DASHBOARD_PORT"])
    p.add_argument("--host", default=_PARAMS["DASHBOARD_HOST"])
    p.add_argument("--refresh", type=float, default=_PARAMS["DASHBOARD_REFRESH"])
    p.add_argument("--open", action="store_true", help="open a browser")
    p = sub.add_parser("settings", help="print (or --write) the AirSim settings.json")
    common(p)
    p.add_argument("--write", nargs="?", const="auto", help="path, or nothing for ~/Documents/AirSim/settings.json")
    p = sub.add_parser("check", help="AirSim preflight check")
    common(p)
    p = sub.add_parser("plot", help="learning curves and evaluation chart")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out")
    sub.add_parser("selftest", help="quick run of everything on the surrogate")
    args = ap.parse_args(argv)

    if args.cmd == "dashboard":
        return cmd_dashboard(args.path, args.mode, args.port, args.host, args.open, args.refresh)
    if args.cmd == "plot":
        return cmd_plot(args.paths, args.out)
    if args.cmd == "selftest":
        return cmd_selftest()
    sets = list(args.set)
    if args.backend:
        sets.append(f"BACKEND={args.backend!r}")
    cfg = make_cfg(sets)
    if args.cmd == "settings":
        return cmd_settings(cfg, args.write)
    if args.cmd == "check":
        cfg.BACKEND = "airsim"
        return cmd_check(derive(cfg))
    name = args.run or cfg.RUN_NAME
    if args.cmd == "train":
        phase = args.phase or cfg.PHASE
        run_dir = os.path.join(cfg.RUN_ROOT, name or f"{phase}_{cfg.BACKEND}_{_now()}")
        return train_phase(cfg, phase, run_dir, init_path=_phase_init(cfg, phase, args.init),
                           dashboard=not args.no_dashboard)
    if args.cmd == "all":
        return run_all(cfg, os.path.join(cfg.RUN_ROOT, name or f"session_{cfg.BACKEND}_{_now()}"),
                       dashboard=not args.no_dashboard, skip_eval=args.skip_eval, init_session=args.init_session)
    if args.cmd == "eval":
        if args.session:
            dirs = {p: os.path.join(args.session, p) for p in _PHASE_SCHED}
            ck_a = _ckpt_of(dirs["always_send"], cfg.CKPT_FOR_INIT)
            entries = default_entries(cfg, ck_a, _ckpt_of(dirs["joint_push"]), _ckpt_of(dirs["joint_pull"]))
            out = args.out or os.path.join(args.session, f"eval_{cfg.BACKEND}_{_now()}")
        elif args.ckpt:
            entries = [dict(label=f"{os.path.basename(os.path.dirname(os.path.abspath(c)))}: {args.sched}"
                            + (f"({args.param:g})" if args.param is not None else ""), ckpt=c, sched=args.sched,
                            param=args.param) for c in args.ckpt]
            out = args.out or os.path.join(cfg.RUN_ROOT, f"eval_{_now()}")
        else:
            raise SystemExit("eval needs --session or --ckpt")
        res = evaluate(cfg, entries, out, dashboard=args.dashboard)
        cmd_plot([out], os.path.join(out, "plots"))
        return res
    if args.cmd == "watch":
        return cmd_watch(cfg, args.ckpt, args.sched, args.param, args.controller, args.slots, args.stochastic,
                         dashboard=not args.no_dashboard, name=args.run)
    return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C); checkpoints and logs written so far are kept", flush=True)
        sys.exit(130)

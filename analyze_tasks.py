#!/usr/bin/env python3
"""
analyze_tasks.py: where the slots of a task go, from a run's tasks.csv and trace.csv.

For each run folder written by drone_jcc.py (train, all or watch) it reports, per task and averaged:
  * the target distance d, from the task's start position to its target;
  * the slots used, and the straight-line bound ceil((d - HIT_RADIUS) / (U_MAX x DT));
  * the slots spent accelerating, cruising and landing, and the slots lost in each against flying straight at U_MAX;
  * the sends and the held slots;
  * lever b: the first slot whose path passes within HIT_RADIUS of the target;
  * J per slot and per second.

tasks.csv has every drone (or surrogate environment). trace.csv has drone 1 only, so the phases and lever b come from
drone 1's tasks.

Phases. The speed is |v| at the end of each task slot. Accelerating: the slots up to and including the first one
that ends at CRUISE_FRAC x U_MAX or faster. Landing: the slots after the last one that does. Cruising: the slots in
between. When no slot reaches that speed, the slots up to the fastest one count as accelerating and the rest as
landing. The slots lost in a phase are its slots minus its progress (the drop in the distance to the target) divided
by U_MAX x DT. Phases are computed for successful tasks only.

Lever b. The path in a slot is taken as the straight segment between the drone's positions at the ends of
consecutive slots; the first segment starts at the task's start position. Under the segment rule the task would end
at the first slot whose segment passes within HIT_RADIUS of the target (the minimum point-to-segment distance).
"Slots saved" is the task's slots minus that slot, on the same trajectory; a policy trained for the rule could save
more. For a failed task the last segment is left out, because the logged position after a failure is the reset point.

J. Each completed task is a cycle: its slots plus the slots held after it (HOLD_SLOTS after a success, RESET_SLOTS
after a failure). J = (successes - COST_C x sends) / cycle slots, summed over the completed tasks. The task that is
still running when the log ends is left out, so J can differ slightly from the J that drone_jcc.py prints.

Usage:
  python analyze_tasks.py RUN [RUN ...] [--cruise-frac 0.8] [--tail 1.0] [--md FILE]
RUN is a run folder (with tasks.csv) or a session folder from `all` (its phase folders are analyzed in turn). Each
run gets RUN/task_analysis.csv (one row per task) and RUN/task_analysis.json (the averages).
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from tqdm import tqdm

FAILS = ("out_of_bounds", "timeout", "collision")
PHASE_ORDER = ("always_send", "joint_push", "joint_pull")


# ---- reading ---------------------------------------------------------------------------------------------------------
def find_runs(paths):
    runs = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "tasks.csv")):
            runs.append(p)
            continue
        subs = sorted(d for d in os.listdir(p) if os.path.isfile(os.path.join(p, d, "tasks.csv"))) \
            if os.path.isdir(p) else []
        if not subs:
            raise SystemExit(f"{p}: no tasks.csv here or in its subfolders")
        subs.sort(key=lambda d: (PHASE_ORDER.index(d) if d in PHASE_ORDER else len(PHASE_ORDER), d))
        runs += [os.path.join(p, d) for d in subs]
    return runs


def vec(s):
    return np.array([float(x) for x in s.split()], float)


def read_tasks(run):
    with open(os.path.join(run, "tasks.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("env", "task", "start_slot", "end_slot", "slots", "sends", "deliveries"):
            r[k] = int(r[k])
        r["final_dist"] = float(r["final_dist"])
        r["target"], r["start_pos"] = vec(r["target"]), vec(r["start_pos"])
    return rows


def read_trace(run):
    path = os.path.join(run, "trace.csv")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        total = max(sum(1 for _ in f) - 1, 0)
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {k: i for i, k in enumerate(head)}
        ax = [a for a in ("x", "y", "z") if a in ix]
        pc, vc, gc = [ix[a] for a in ax], [ix["v" + a] for a in ax], [ix["g" + a] for a in ax]
        im, ie = ix["mode"], ix["event"]
        mode, event, pos, vel, goal = [], [], [], [], []
        for r in tqdm(rd, total=total, unit="row", desc=f"reading {run}/trace.csv", leave=False, mininterval=0.5):
            mode.append(r[im])
            event.append(r[ie])
            pos.append([float(r[i]) for i in pc])
            vel.append([float(r[i]) for i in vc])
            goal.append([float(r[i]) for i in gc])
    return dict(mode=mode, event=event, pos=np.array(pos, float).reshape(-1, len(ax)),
                vel=np.array(vel, float).reshape(-1, len(ax)), goal=np.array(goal, float).reshape(-1, len(ax)))


def trace_tasks(tr):
    """Drone 1's completed tasks as (row indices of the task slots, outcome). A row holds the state after its slot,
    and its mode is the mode after the slot, so a slot is a task slot when the previous row's mode is "task" (a run
    starts in a task). A row with an event ends a task. The task still running at the end of the log is dropped."""
    out, cur, prev = [], [], "task"
    for r, (m, ev) in enumerate(zip(tr["mode"], tr["event"])):
        if prev == "task":
            cur.append(r)
        if ev:
            out.append((cur, ev))
            cur = []
        prev = m
    return out


# ---- geometry --------------------------------------------------------------------------------------------------------
def seg_dist(a, b, g):
    """Minimum distance from the point g to each segment a[i] -> b[i]."""
    ab = b - a
    L2 = np.sum(ab * ab, axis=1)
    t = np.clip(np.sum((g - a) * ab, axis=1) / np.maximum(L2, 1e-12), 0.0, 1.0)
    t[L2 <= 1e-12] = 0.0
    return np.linalg.norm(a + t[:, None] * ab - g, axis=1)


def line_bound(d, R, step):
    return int(math.ceil(max(d - R, 0.0) / step - 1e-9))


def traced(rows, tr, trow, R, U, step, frac):
    """Phases and lever b of one of drone 1's tasks."""
    n, first, ok = len(rows), rows[0], trow["outcome"] == "success"
    if first > 0:  # the row before the first task slot: where the task starts, and its target
        p0, g = tr["pos"][first - 1], tr["goal"][first - 1]
    else:  # the first task of the run starts at the reset point
        p0, g = trow["start_pos"], (tr["goal"][first] if n > 1 else trow["target"])
    pts = np.vstack([p0[None, :], tr["pos"][rows]])
    out = {}
    m = n if ok else n - 1
    sd = seg_dist(pts[:m], pts[1:m + 1], g) if m > 0 else np.zeros(0)
    hit = np.flatnonzero(sd <= R)
    seg = int(hit[0]) + 1 if len(hit) else (n if ok else None)
    out["seg_slot"] = seg
    out["seg_saved"] = (n - seg) if (ok and seg is not None) else None
    if not ok:
        return out
    dist = np.linalg.norm(pts - g, axis=1)
    sp = np.linalg.norm(tr["vel"][rows], axis=1)
    fast = np.flatnonzero(sp >= frac * U)
    a, b = (fast[0] + 1, fast[-1] + 1) if len(fast) else (int(np.argmax(sp)) + 1,) * 2
    out.update(accel=int(a), cruise=int(b - a), land=int(n - b),
               lost_accel=float(a - (dist[0] - dist[a]) / step),
               lost_cruise=float((b - a) - (dist[a] - dist[b]) / step),
               lost_land=float((n - b) - (dist[b] - dist[n]) / step))
    return out


# ---- one run ---------------------------------------------------------------------------------------------------------
def analyze(run, frac, tail):
    with open(os.path.join(run, "config.json"), encoding="utf-8") as f:
        P = json.load(f)["params"]
    R, U, DT, C = float(P["HIT_RADIUS"]), float(P["U_MAX"]), float(P["DT"]), float(P["COST_C"])
    hold, reset = int(P["HOLD_SLOTS"]), int(P["RESET_SLOTS"])
    step = U * DT
    tasks = read_tasks(run)
    if not tasks:
        print(f"[{run}] no completed tasks", flush=True)
        return None
    last = {}
    for t in tasks:
        last[t["env"]] = max(last.get(t["env"], 0), t["end_slot"])
    for t in tasks:
        t["keep"] = t["start_slot"] >= (1.0 - tail) * last[t["env"]]
        t["d"] = float(np.linalg.norm(t["target"] - t["start_pos"]))
        t["bound"] = line_bound(t["d"], R, step)
        t["held"] = hold if t["outcome"] == "success" else reset
    use = [t for t in tasks if t["keep"]]
    if not use:
        print(f"[{run}] no completed tasks in the last {tail:g} of the slots", flush=True)
        return None
    succ = [t for t in use if t["outcome"] == "success"]
    sends = sum(t["sends"] for t in use)
    cycle = sum(t["slots"] + t["held"] for t in use)
    J = (len(succ) - C * sends) / max(cycle, 1)
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")  # noqa: E731
    S = {"tasks (all drones)": len(use), "success rate": len(succ) / len(use)}
    for k in FAILS:
        S[f"failures: {k}"] = sum(t["outcome"] == k for t in use)
    S.update({"target distance d (m)": mean([t["d"] for t in use]),
              "slots per successful task": mean([t["slots"] for t in succ]),
              "straight-line bound (slots)": mean([t["bound"] for t in succ]),
              "slots above the bound": mean([t["slots"] - t["bound"] for t in succ]),
              "sends per task": sends / len(use),
              "deliveries per task": sum(t["deliveries"] for t in use) / len(use),
              "held slots per task": mean([t["held"] for t in use]),
              "cycle slots per task": cycle / len(use),
              "J per slot": J, "J per second (J / DT)": J / DT})
    print(f"[{run}] {len(use)} tasks, success {S['success rate']:.3f}, {S['slots per successful task']:.2f} slots "
          f"per success (bound {S['straight-line bound (slots)']:.2f}), {S['sends per task']:.2f} sends per task, "
          f"J {J:.4f} per slot = {J / DT:.4f} per s", flush=True)

    tr = read_trace(run)
    mine = [t for t in tasks if t["env"] == 0]
    if tr is not None:
        tt = trace_tasks(tr)
        bad = next((j for j, ((rows, ev), t) in enumerate(zip(tt, mine))
                    if len(rows) != t["slots"] or ev != t["outcome"]), None)
        if len(tt) > len(mine) or bad is not None:
            print(f"[{run}] trace.csv does not match drone 1's rows in tasks.csv (task {bad}); phases and lever b "
                  "skipped", flush=True)
            tt = []
        for (rows, ev), t in zip(tt, mine):
            t.update(traced(rows, tr, t, R, U, step, frac))
    tok = [t for t in mine if t["keep"] and "seg_slot" in t]
    sok = [t for t in tok if t["outcome"] == "success"]
    if sok:
        S.update({"drone-1 successful tasks traced": len(sok),
                  "drone-1 slots per successful task": mean([t["slots"] for t in sok]),
                  "accelerating slots": mean([t["accel"] for t in sok]),
                  "cruising slots": mean([t["cruise"] for t in sok]),
                  "landing slots": mean([t["land"] for t in sok]),
                  "slots lost accelerating": mean([t["lost_accel"] for t in sok]),
                  "slots lost cruising": mean([t["lost_cruise"] for t in sok]),
                  "slots lost landing": mean([t["lost_land"] for t in sok]),
                  "lever b: share of tasks ending earlier": mean([t["seg_saved"] > 0 for t in sok]),
                  "lever b: slots saved per task": mean([t["seg_saved"] for t in sok]),
                  "lever b: failures that become successes": sum(t["outcome"] != "success" and
                                                                 t["seg_slot"] is not None for t in tok)})
        print(f"[{run}] drone 1, {len(sok)} successful tasks: accelerating {S['accelerating slots']:.2f}, cruising "
              f"{S['cruising slots']:.2f}, landing {S['landing slots']:.2f} slots; lever b saves "
              f"{S['lever b: slots saved per task']:.2f} slots per task", flush=True)
    elif tr is None:
        print(f"[{run}] no trace.csv; phases and lever b skipped", flush=True)

    cols = ["env", "task", "outcome", "d", "slots", "bound", "sends", "deliveries", "held", "final_dist", "accel",
            "cruise", "land", "lost_accel", "lost_cruise", "lost_land", "seg_slot", "seg_saved"]
    with open(os.path.join(run, "task_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in use:
            w.writerow(["" if t.get(c) is None else (f"{t[c]:.3f}" if isinstance(t[c], float) else t[c])
                        for c in cols])
    S["_params"] = dict(HIT_RADIUS=R, U_MAX=U, DT=DT, COST_C=C, HOLD_SLOTS=hold, RESET_SLOTS=reset,
                        cruise_frac=frac, tail=tail)
    with open(os.path.join(run, "task_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    return S


# ---- table -----------------------------------------------------------------------------------------------------------
def fmt(k, v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, int) or k.startswith(("tasks", "failures", "drone-1 successful", "lever b: failures")):
        return str(int(v))
    if k.startswith("J"):
        return f"{v:.4f}"
    if k in ("success rate", "lever b: share of tasks ending earlier"):
        return f"{v:.3f}"
    return f"{v:.2f}"


def label(run):
    """The run folder's name, with the session's name for a phase folder (session/joint_push)."""
    run = os.path.normpath(run)
    base = os.path.basename(run)
    return f"{os.path.basename(os.path.dirname(run))}/{base}" if base in PHASE_ORDER else base


def table(runs, sums):
    keys = []
    for s in sums:
        keys += [k for k in s if not k.startswith("_") and k not in keys]
    names = [label(r) for r in runs]
    lines = ["| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    lines += ["| " + k + " | " + " | ".join(fmt(k, s.get(k)) for s in sums) + " |" for k in keys]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders, or session folders from `all`")
    ap.add_argument("--cruise-frac", type=float, default=0.8, help="cruising speed threshold, as a fraction of U_MAX")
    ap.add_argument("--tail", type=float, default=1.0,
                    help="use only the tasks that start in the last TAIL fraction of each drone's slots (e.g. 0.2 "
                         "for the end of a training run)")
    ap.add_argument("--md", help="also write the table to this Markdown file")
    a = ap.parse_args(argv)
    runs = find_runs(a.runs)
    sums, done = [], []
    for run in runs:
        s = analyze(run, a.cruise_frac, a.tail)
        if s is not None:
            sums.append(s)
            done.append(run)
    if not sums:
        raise SystemExit("nothing to report")
    t = table(done, sums)
    p = sums[0]["_params"]
    note = (f"Phases: cruising is |v| >= {a.cruise_frac:g} x U_MAX = {a.cruise_frac * p['U_MAX']:g} m/s. "
            f"Lever b: the segment rule with HIT_RADIUS = {p['HIT_RADIUS']:g} m. Tail: {a.tail:g}.")
    print("\n" + t + "\n\n" + note, flush=True)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(t + "\n\n" + note + "\n")
    return sums


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

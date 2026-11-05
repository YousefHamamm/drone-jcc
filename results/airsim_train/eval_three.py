#!/usr/bin/env python3
"""eval_three.py: evaluate an AirSim session's always_send, joint_push and joint_pull policies in AirSim on common
random numbers (every policy flies the same seeds), with the paired differences: push and pull against always send,
and push against pull.

Usage: python results/airsim_train/eval_three.py SESSION [--set NAME=VALUE ...]
Writes SESSION/eval/ (eval.json, eval.csv, plots/) and SESSION/eval/summary.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import drone_jcc as dj  # noqa: E402

session, rest = sys.argv[1], sys.argv[2:]
sets = [rest[i + 1] for i in range(len(rest) - 1) if rest[i] == "--set"]
cfg = dj.make_cfg(["BACKEND='airsim'"] + sets)
ck = {ph: os.path.join(session, ph, "ckpt_latest.pt") for ph in ("always_send", "joint_push", "joint_pull")}
entries = [dict(label="always send (learned controller)", ckpt=ck["always_send"], sched="always"),
           dict(label="joint push (learned)", ckpt=ck["joint_push"], sched="learned"),
           dict(label="joint pull (learned)", ckpt=ck["joint_pull"], sched="learned")]
out = os.path.join(session, "eval")
res = dj.evaluate(cfg, entries, out)
dj.cmd_plot([out], os.path.join(out, "plots"))

dt = float(cfg.DT)
rows = ["| policy | J per slot (95% CI) | tasks per second | difference to always send (95% CI) | sends per task | "
        "success rate | slots per task |", "|---|---|---|---|---|---|---|"]
for r in res:
    d = r.get("diff_vs_first")
    ds = "0 (reference)" if r is res[0] else (f"{d:+.4f} +- {r['diff_ci']:.4f}" if d is not None else "-")
    rows.append(f"| {r['label']} | {r['J']:.4f} +- {r['J_ci']:.4f} | {r['J'] / dt:.4f} | {ds} | "
                f"{r['sends_per_task']:.2f} | {r['succ_rate']:.3f} | {r['task_slots']:.1f} |")
by = {r["label"]: r for r in res}
m, h = dj.mean_ci(np.asarray(by["joint push (learned)"]["J_reps"]) - np.asarray(by["joint pull (learned)"]["J_reps"]))
text = "\n".join(rows) + (
    f"\n\npush - pull, paired by seed: {m:+.4f} +- {h:.4f} per slot ({m / dt:+.4f} +- {h / dt:.4f} tasks per second)."
    f"\n{res[0]['reps']} replications of {res[0]['slots_per_rep']} slots per policy, EVAL_SEED {cfg.EVAL_SEED}; "
    f"checkpoints: {', '.join(ck.values())}.\n")
with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text, flush=True)

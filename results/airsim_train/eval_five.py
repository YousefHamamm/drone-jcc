#!/usr/bin/env python3
"""eval_five.py: evaluate five policies in AirSim on common random numbers (every policy flies the same seeds):
always send (the AirSim always_send checkpoint), push and pull from the surrogate session with no fine-tuning, and
push and pull fine-tuned in AirSim. Reports J per slot, tasks per second, sends per task, and paired 95% CIs against
always send, between push and pull, and between fine-tuned and not fine-tuned.

Usage: python results/airsim_train/eval_five.py AIRSIM_SESSION SURROGATE_SESSION [--set NAME=VALUE ...]
Writes AIRSIM_SESSION/eval/ (eval.json, eval.csv, plots/) and AIRSIM_SESSION/eval/summary.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import drone_jcc as dj  # noqa: E402

air, surr, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
sets = [rest[i + 1] for i in range(len(rest) - 1) if rest[i] == "--set"]
cfg = dj.make_cfg(["BACKEND='airsim'"] + sets)


def ck(session, phase):
    return os.path.join(session, phase, "ckpt_latest.pt")


A, SP, SL, FP, FL = ("always send (AirSim always_send)", "push, surrogate (no fine-tuning)",
                     "pull, surrogate (no fine-tuning)", "push, fine-tuned in AirSim", "pull, fine-tuned in AirSim")
entries = [dict(label=A, ckpt=ck(air, "always_send"), sched="always"),
           dict(label=SP, ckpt=ck(surr, "joint_push"), sched="learned"),
           dict(label=SL, ckpt=ck(surr, "joint_pull"), sched="learned"),
           dict(label=FP, ckpt=ck(air, "joint_push"), sched="learned"),
           dict(label=FL, ckpt=ck(air, "joint_pull"), sched="learned")]
out = os.path.join(air, "eval")
res = dj.evaluate(cfg, entries, out)
dj.cmd_plot([out], os.path.join(out, "plots"))

dt = float(cfg.DT)
by = {r["label"]: r for r in res}
rows = ["| policy | J per slot (95% CI) | tasks per second | difference to always send (95% CI) | sends per task | "
        "success rate | slots per task |", "|---|---|---|---|---|---|---|"]
for r in res:
    d = r.get("diff_vs_first")
    ds = "0 (reference)" if r["label"] == A else (f"{d:+.4f} +- {r['diff_ci']:.4f}" if d is not None else "-")
    rows.append(f"| {r['label']} | {r['J']:.4f} +- {r['J_ci']:.4f} | {r['J'] / dt:.4f} | {ds} | "
                f"{r['sends_per_task']:.2f} | {r['succ_rate']:.3f} | {r['task_slots']:.1f} |")
rows += ["", "| paired difference (same seeds) | J per slot (95% CI) | tasks per second (95% CI) |", "|---|---|---|"]
for a, b in ((SP, SL), (FP, FL), (FP, SP), (FL, SL)):
    if a in by and b in by:
        m, h = dj.mean_ci(np.asarray(by[a]["J_reps"]) - np.asarray(by[b]["J_reps"]))
        rows.append(f"| {a} minus {b} | {m:+.4f} +- {h:.4f} | {m / dt:+.4f} +- {h / dt:.4f} |")
text = "\n".join(rows) + (
    f"\n\n{res[0]['reps']} replications of {res[0]['slots_per_rep']} slots per policy, EVAL_SEED {cfg.EVAL_SEED}. "
    "Checkpoints: " + ", ".join(e["ckpt"] for e in entries) + ".\n")
with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text, flush=True)

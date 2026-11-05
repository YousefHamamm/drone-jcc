| policy | J per slot (95% CI) | tasks per second | difference to always send (95% CI) | sends per task | success rate | slots per task |
|---|---|---|---|---|---|---|
| always send (AirSim always_send) | 0.0400 +- 0.0013 | 0.1601 | 0 (reference) | 18.39 | 1.000 | 18.3 |
| push, surrogate (no fine-tuning) | 0.0464 +- 0.0017 | 0.1854 | +0.0063 +- 0.0006 | 4.42 | 1.000 | 18.5 |
| pull, surrogate (no fine-tuning) | 0.0464 +- 0.0012 | 0.1855 | +0.0063 +- 0.0005 | 4.70 | 1.000 | 18.5 |
| push, fine-tuned in AirSim | 0.0465 +- 0.0010 | 0.1859 | +0.0064 +- 0.0004 | 4.51 | 1.000 | 18.5 |
| pull, fine-tuned in AirSim | 0.0458 +- 0.0008 | 0.1832 | +0.0058 +- 0.0009 | 5.15 | 1.000 | 18.7 |

| paired difference (same seeds) | J per slot (95% CI) | tasks per second (95% CI) |
|---|---|---|
| push, surrogate (no fine-tuning) minus pull, surrogate (no fine-tuning) | -0.0000 +- 0.0008 | -0.0001 +- 0.0032 |
| push, fine-tuned in AirSim minus pull, fine-tuned in AirSim | +0.0007 +- 0.0007 | +0.0027 +- 0.0028 |
| push, fine-tuned in AirSim minus push, surrogate (no fine-tuning) | +0.0001 +- 0.0008 | +0.0005 +- 0.0032 |
| pull, fine-tuned in AirSim minus pull, surrogate (no fine-tuning) | -0.0006 +- 0.0006 | -0.0024 +- 0.0025 |

6 replications of 3000 slots per policy, EVAL_SEED 5000. Checkpoints: runs/air/always_send/ckpt_latest.pt, runs/surr_tau155/joint_push/ckpt_latest.pt, runs/surr_tau155/joint_pull/ckpt_latest.pt, runs/air/joint_push/ckpt_latest.pt, runs/air/joint_pull/ckpt_latest.pt.

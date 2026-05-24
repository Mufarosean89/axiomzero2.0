#!/usr/bin/env python3
"""
Visualize training stats from checkpoints/training_stats.json
Saves charts to checkpoints/training_charts.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS_PATH = "checkpoints/training_stats.json"
OUTPUT_PATH = "checkpoints/training_charts.png"

# Load data
with open(STATS_PATH) as f:
    stats = json.load(f)

# Separate pretrain and iteration data
pretrain = stats[0] if stats[0].get("phase") == "supervised_pretrain" else None
iterations = [s for s in stats if "iteration" in s]

iters = [s["iteration"] for s in iterations]
win_rates = [s["win_rate"] for s in iterations]
policy_losses = [s["policy_loss"] for s in iterations]
value_losses = [s["value_loss"] for s in iterations]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Axiom Zero - RL Agent Training", fontsize=16, fontweight="bold", y=0.98)

# Colors
blue = "#1f77b4"
green = "#2ca02c"
red = "#d62728"
orange = "#ff7f0e"

# ── 1) Win Rate ───────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(iters, win_rates, marker="o", linestyle="-", color=green, linewidth=2, markersize=8)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="100%")
ax.fill_between(iters, win_rates, alpha=0.15, color=green)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Win Rate", fontsize=11)
ax.set_title("Win Rate per Iteration", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid(True, alpha=0.3)
# Annotate values
for i, wr in zip(iters, win_rates):
    ax.annotate(f"{wr:.0%}", (i, wr), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color=green)

# ── 2) Policy Loss ────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(iters, policy_losses, marker="s", linestyle="-", color=blue, linewidth=2, markersize=7)
ax.fill_between(iters, policy_losses, alpha=0.15, color=blue)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Policy Loss", fontsize=11)
ax.set_title("Policy Loss per Iteration", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(iters, policy_losses, 1)
p = np.poly1d(z)
ax.plot(iters, p(iters), linestyle="--", color="gray", alpha=0.6, linewidth=1.5, label=f"Trend: {z[0]:.4f}/iter")
ax.legend(fontsize=9)

# ── 3) Value Loss ─────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(iters, value_losses, marker="^", linestyle="-", color=red, linewidth=2, markersize=7)
ax.fill_between(iters, value_losses, alpha=0.15, color=red)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Value Loss", fontsize=11)
ax.set_title("Value Loss per Iteration", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Add trend line
z2 = np.polyfit(iters, value_losses, 1)
p2 = np.poly1d(z2)
ax.plot(iters, p2(iters), linestyle="--", color="gray", alpha=0.6, linewidth=1.5, label=f"Trend: {z2[0]:.4f}/iter")
ax.legend(fontsize=9)

# ── 4) Summary Stats Table ────────────────────────────────────────────
ax = axes[1, 1]
ax.axis("off")
summary_data = [
    ("Metric", "Value"),
    ("", ""),
]

if pretrain:
    summary_data.append(("Pre-train Policy Loss", f"{pretrain['pretrain_policy_loss']:.4f}"))
    summary_data.append(("Pre-train Value Loss", f"{pretrain['pretrain_value_loss']:.4f}"))
    summary_data.append(("Buffer After Seed", f"{pretrain['buffer_size']}"))

avg_wr = np.mean(win_rates)
final_wr = win_rates[-1]
avg_pi = np.mean(policy_losses)
avg_v = np.mean(value_losses)

summary_data.append(("", ""))
summary_data.append(("Avg Win Rate", f"{avg_wr:.1%}"))
summary_data.append(("Final Win Rate", f"{final_wr:.0%}"))
summary_data.append(("Avg Policy Loss", f"{avg_pi:.4f}"))
summary_data.append(("Avg Value Loss", f"{avg_v:.4f}"))
summary_data.append(("Best Iteration", f"{iters[win_rates.index(max(win_rates))]}"))

# Create table
table = ax.table(cellText=summary_data, loc="center", cellLoc="left", colWidths=[0.5, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(2):
    cell = table[0, j]
    cell.set_text_props(fontweight="bold", fontsize=11)
    cell.set_facecolor("#404040")
    cell.set_text_props(color="white")

# Style data rows
for i in range(2, len(summary_data)):
    for j in range(2):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        else:
            cell.set_facecolor("white")

ax.set_title("Training Summary", fontsize=13, fontweight="bold", pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Chart saved to {OUTPUT_PATH}")
plt.close()

# ── Also print text summary ───────────────────────────────────────────
print()
print("=" * 60)
print("TRAINING STATS SUMMARY")
print("=" * 60)

if pretrain:
    print(f"\nPre-training:")
    print(f"  Policy Loss: {pretrain['pretrain_policy_loss']:.4f}")
    print(f"  Value Loss:  {pretrain['pretrain_value_loss']:.4f}")
    print(f"  Buffer Size: {pretrain['buffer_size']}")

print(f"\nSelf-Play ({len(iterations)} iterations):")
print(f"  Avg Win Rate:      {avg_wr:.1%}")
print(f"  Final Win Rate:    {final_wr:.0%}")
print(f"  Avg Policy Loss:   {avg_pi:.4f}")
print(f"  Avg Value Loss:    {avg_v:.4f}")

print(f"\nPer-Iteration Breakdown:")
print(f"  {'Iter':>5s}  {'Win Rate':>10s}  {'Policy Loss':>12s}  {'Value Loss':>12s}")
print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}")
for s in iterations:
    wr_str = f"{s['win_rate']:.0%}"
    pi_str = f"{s['policy_loss']:.4f}"
    v_str = f"{s['value_loss']:.4f}"
    print(f"  {s['iteration']:>5d}  {wr_str:>10s}  {pi_str:>12s}  {v_str:>12s}")

print()

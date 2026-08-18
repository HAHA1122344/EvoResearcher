#!/usr/bin/env python3
"""
Figure: compute cost (mean tokens/question) vs. accuracy for E8.
Pareto-style scatter: each condition is a point; arrows connect d1 -> d3
per benchmark to show the loop's cost/accuracy effect. Accuracy bars = 95% Wilson CI.

Reproduces figures/fig05_cost_accuracy.png. Data from results/crossbench_study.json
(deepseek-v4-flash, n=100, temperature 0.3).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "crossbench_study.json")

d = json.load(open(RESULTS))
gsm = d["gsm8k"]; math = d["math"]
tau_rows = {r["tau"]: r for r in d["bbh_tau"]}
tau0, tau07, d1 = tau_rows[0.0], tau_rows[0.7], tau_rows[1.0]

# (label, acc%, ci_low, ci_high, tokens, calls, benchmark, d1?)
points = [
    # BBH
    ("BBH d1",       d1["acc"]*100,  d1["ci"][0]*100,  d1["ci"][1]*100,  d1["avg_tokens"],  "BBH"),
    ("BBH d3",       tau0["acc"]*100, tau0["ci"][0]*100, tau0["ci"][1]*100, tau0["avg_tokens"], "BBH"),
    ("BBH τ=0.7", tau07["acc"]*100, tau07["ci"][0]*100, tau07["ci"][1]*100, tau07["avg_tokens"], "BBH"),
    # GSM8K
    ("GSM8K d1",     gsm["d1"]["acc"]*100, gsm["d1"]["ci"][0]*100, gsm["d1"]["ci"][1]*100, gsm["d1"]["avg_tokens"], "GSM8K"),
    ("GSM8K d3",     gsm["d3"]["acc"]*100, gsm["d3"]["ci"][0]*100, gsm["d3"]["ci"][1]*100, gsm["d3"]["avg_tokens"], "GSM8K"),
    # MATH
    ("MATH d1",      math["d1"]["acc"]*100, math["d1"]["ci"][0]*100, math["d1"]["ci"][1]*100, math["d1"]["avg_tokens"], "MATH"),
    ("MATH d3",      math["d3"]["acc"]*100, math["d3"]["ci"][0]*100, math["d3"]["ci"][1]*100, math["d3"]["avg_tokens"], "MATH"),
]

# (label, acc, tokens, marker, color, annotate_dx, annotate_dy)
markers = {
    "BBH":   ("o",  "#d62728"),
    "GSM8K": ("s",  "#1f77b4"),
    "MATH":  ("^",  "#2ca02c"),
}

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)

# arrows d1 -> d3 per benchmark
arrows = [("BBH", "BBH d1", "BBH d3"), ("GSM8K", "GSM8K d1", "GSM8K d3"), ("MATH", "MATH d1", "MATH d3")]
pt = {p[0]: p for p in points}
for bench, a, b in arrows:
    pa, pb = pt[a], pt[b]
    ax.annotate("", xy=(pb[4], pb[1]), xytext=(pa[4], pa[1]),
                arrowprops=dict(arrowstyle="->", color="0.6", lw=1.4, shrinkA=8, shrinkB=8))

for label, acc, lo, hi, tok, bench in points:
    m, c = markers[bench]
    ax.errorbar(tok, acc, yerr=[[acc - lo], [hi - acc]], fmt=m, color=c,
                markersize=7, ecolor=c, elinewidth=1.0, capsize=3, markeredgecolor="white", markeredgewidth=0.6)
    dx, dy = 12, 0.0
    if label == "MATH d1":   dx, dy = -8,  -1.8
    if label == "MATH d3":   dx, dy = 14,   2.6
    if label == "GSM8K d1":  dx, dy = -10,  2.0
    if label == "GSM8K d3":  dx, dy = 8,   -2.2
    if label == "BBH d3":    dx, dy = 12,  -1.5
    if label == "BBH τ=0.7": dx, dy = -6,  2.6
    ax.annotate(label, (tok, acc), textcoords="offset points", xytext=(dx, dy),
                fontsize=8, color=c)

ax.set_xlabel("Mean tokens per question (compute cost)", fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.set_title("Compute Cost vs. Accuracy (E8, deepseek-v4-flash, $n$=100)", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.4)
ax.tick_params(labelsize=8)
ax.set_xlim(360, 2500)
ax.set_ylim(15, 105)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# dashed line at 100 tokens x-axis log hint? keep linear.
fig.tight_layout()
out = os.path.join(HERE, "fig05_cost_accuracy.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("Saved", out)

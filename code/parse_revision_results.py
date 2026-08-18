#!/usr/bin/env python3
"""Parse results/revision_study.json into the paper's revision tables.

Prints:
  1. R1: n=500 core comparison (d1 vs d3 per benchmark) -> E7 replacement
  2. R2: named baselines (Self-Refine, Reflexion) vs EvoResearcher d3 -> new table
  3. R3: repeated-runs reproducibility (BBH n=100 x5 seeds, mean+std) -> new table
  4. R4: confidence calibration (ECE, mean conf, bins) -> E6 supplement
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "results", "revision_study.json")
if not os.path.exists(PATH):
    sys.exit("missing %s" % PATH)
d = json.load(open(PATH))

def pct(x): return "%.1f" % (x * 100)
def ci(c): return "[%.1f, %.1f]" % (c[0] * 100, c[1] * 100)

print("=" * 78)
print("R1: n=500 core comparison  (model %s)" % d.get("model", "?"))
print("=" * 78)
print("%-8s %-14s %-18s %-9s %-10s %-9s" % ("bench", "condition", "acc(CI)", "steps", "early%", "tok"))
for b, k in (("bbh", "bbh"), ("gsm8k", "gsm8k"), ("math", "math")):
    r = d["r1_core"][k]
    for cond in ("d1", "d3"):
        s = r[cond]
        print("%-8s %-14s %s %-9.2f %-10.0f %-9.0f"
              % (b, cond, "%s %s" % (pct(s["acc"]), ci(s["ci"])),
                 s["avg_steps"], s["early_stop_rate"] * 100, s["avg_tokens"]))
    s1, s3 = r["d1"], r["d3"]
    print("  -> gain %+.1f pp (d3-d1); overlapping CIs: %s"
          % ((s3["acc"] - s1["acc"]) * 100,
             bool(s3["ci"][0] > s1["ci"][1] or s1["ci"][0] > s3["ci"][1])))

if "r2_baselines" in d:
    print()
    print("=" * 78)
    print("R2: named baselines at n=500 (vs EvoResearcher d3, d1 single-shot)")
    print("=" * 78)
    print("%-8s %-14s %-18s %-9s %-9s" % ("bench", "method", "acc(CI)", "calls", "tok"))
    for b in ("bbh", "gsm8k", "math"):
        r = d["r2_baselines"][b]
        core = d["r1_core"][b]
        for name, s in (("d1", core["d1"]), ("EvoRes d3", core["d3"]),
                        ("Self-Refine", r["self_refine"]), ("Reflexion", r["reflexion"])):
            print("%-8s %-14s %s %-9.2f %-9.0f"
                  % (b, name, "%s %s" % (pct(s["acc"]), ci(s["ci"])),
                     s["avg_steps"], s["avg_tokens"]))

if "r3_repeats" in d:
    print()
    print("=" * 78)
    print("R3: reproducibility, BBH n=100 x 5 seeds (mean +/- std)")
    print("=" * 78)
    r3 = d["r3_repeats"]
    def mstd(vals):
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
        return m, sd
    for cond in ("d1", "d3"):
        m, sd = mstd(r3[cond]["accs"])
        ms, sds = mstd(r3[cond]["steps"])
        mt, sdt = mstd(r3[cond]["tokens"])
        if cond == "d3":
            me, sde = mstd(r3[cond]["early"])
            print("%-4s acc %.1f+/-%.1f%%  steps %.2f+/-%.2f  early %.0f+/-%.0f%%  tok %.0f+/-%.0f"
                  % (cond, m * 100, sd * 100, ms, sds, me * 100, sde * 100, mt, sdt))
        else:
            print("%-4s acc %.1f+/-%.1f%%  steps %.2f+/-%.2f  tok %.0f+/-%.0f"
                  % (cond, m * 100, sd * 100, ms, sds, mt, sdt))
    print("  d1 accs:", ["%.1f" % (a * 100) for a in r3["d1"]["accs"]])
    print("  d3 accs:", ["%.1f" % (a * 100) for a in r3["d3"]["accs"]])

if "r4_ece" in d:
    print()
    print("=" * 78)
    print("R4: confidence calibration (BBH n=%d single-shot)" % d["r4_ece"]["n"])
    print("=" * 78)
    e = d["r4_ece"]
    print("acc=%.1f%% mean_conf=%.1f ECE=%.4f (n_conf=%d)"
          % (e["acc"] * 100, e["mean_conf"], e["ece"], e["n_with_conf"]))
    for row in e["curve"]:
        print("  bin %.2f  n=%d  acc=%.2f" % (row["bin"], row["n"], row["acc"]))

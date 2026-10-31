#!/usr/bin/env python3
"""Recompute the paired statistics in e7_rerun_<dataset>.json from the saved
item-level records and rewrite the 'paired' section in place.

Fixes the delta/bootstrap bug in the first release of run_e7_rerun.py (the
delta and CI were computed on arm A's accuracy instead of the paired A-B
difference).  McNemar counts were always correct.
"""
import json, math, random, os, sys

SEED = int(os.environ.get("SEED", "42"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "results")

def paired_analysis(items_a, items_b, name_a, name_b):
    A = {l["id"]: l for l in items_a}
    B = {l["id"]: l for l in items_b}
    ids = sorted(set(A) & set(B))
    n = len(ids)
    b = sum(1 for i in ids if A[i]["correct"] and not B[i]["correct"])
    c = sum(1 for i in ids if B[i]["correct"] and not A[i]["correct"])
    n_pairs = b + c
    if n_pairs == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n_pairs, i) for i in range(k + 1)) / (2.0 ** n_pairs)
        p = min(1.0, 2 * tail)
    diffs = [(1 if A[i]["correct"] else 0) - (1 if B[i]["correct"] else 0) for i in ids]
    delta = sum(diffs) / n
    rng = random.Random(SEED)
    stats = []
    for _ in range(N_BOOT):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    return {"n_paired": n, "a_only": b, "b_only": c,
            "a_name": name_a, "b_name": name_b,
            "mcnemar_exact_p": p, "delta_a_minus_b": delta,
            "bootstrap_ci": [round(stats[int(0.025 * N_BOOT)], 4),
                             round(stats[int(0.975 * N_BOOT) - 1], 4)],
            "bootstrap_resamples": N_BOOT,
            "bootstrap_se": (sum((s - delta) ** 2 for s in stats) / N_BOOT) ** 0.5}

for ds in sys.argv[1:] or ("math", "gsm8k"):
    path = os.path.join(OUT_DIR, "e7_rerun_%s.json" % ds)
    if not os.path.exists(path):
        print("skip %s (no file)" % ds); continue
    d = json.load(open(path))
    d["paired"] = d.get("paired", {})
    for a, b in (("single", "loop_balanced"), ("single", "loop_plain"), ("loop_plain", "loop_balanced")):
        if a in d and b in d and "items" in d[a] and "items" in d[b]:
            key = "%s_minus_%s" % (a, b)
            pr = paired_analysis(d[a]["items"], d[b]["items"], a, b)
            d["paired"][key] = pr
            print("%s | %s - %s: delta=%+.1fpp CI[%.1f,%.1f] p=%.4f (%s-only=%d, %s-only=%d)"
                  % (ds, a, b, pr["delta_a_minus_b"] * 100,
                     pr["bootstrap_ci"][0] * 100, pr["bootstrap_ci"][1] * 100,
                     pr["mcnemar_exact_p"], a, pr["a_only"], b, pr["b_only"]))
    d["note"] = "paired section recomputed from item-level records (delta = A-B paired difference)"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print("rewrote %s\n" % path)

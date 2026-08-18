#!/usr/bin/env python3
"""
Build n=500 evaluation subsets for the revision round (reviewer: statistical power).
  BBH_500   : same 20 tasks as data/bbh_100.jsonl, 25 questions each = 500
  GSM8K_500 : stride 500 from openai/gsm8k 'test' (1319 total)
  MATH_500  : balanced across all 7 EleutherAI/hendrycks_math categories (~72 each)

Same golden-extraction rules and output format as data/fetch_gsm_math.py.
Deterministic stride sampling for reproducibility.
"""
import json, os, re, sys
from collections import Counter
from datasets import load_dataset, get_dataset_config_names

HERE = os.path.dirname(os.path.abspath(__file__))

def stride_rows(rows, target):
    n = len(rows)
    step = max(1, n // max(1, target))
    return [rows[i] for i in range(0, n, step)[:target]]

def save(name, rows):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved %s (%d rows)" % (path, len(rows)), flush=True)
    print("  per task:", dict(Counter(r["task"] for r in rows)), flush=True)

# ---------------- BBH ----------------
def fetch_bbh(per_task=25):
    tasks = []
    src = os.path.join(HERE, "bbh_100.jsonl")
    for l in open(src):
        tasks.append(json.loads(l)["task"])
    tasks = sorted(set(tasks))
    out = []
    for t in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", t, split="test")
        except Exception as e:
            print("  skip %s: %s" % (t, str(e)[:80]), flush=True)
            continue
        rows = list(ds)
        for j, r in enumerate(stride_rows(rows, per_task)):
            out.append({"id": "bbh-%s-%d" % (t, j), "task": t,
                        "question": r["input"], "golden": r["target"], "type": "bbh"})
        print("  bbh %-32s +%d" % (t, per_task), flush=True)
    return out

# ---------------- GSM8K ----------------
def fetch_gsm8k(target=500):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)
    out = []
    for i, r in enumerate(stride_rows(rows, target)):
        ans = r.get("answer", "")
        m = re.search(r"####\s*(-?[\d,]+(?:\.[\d]+)?)", ans)
        golden = m.group(1).replace(",", "") if m else ans.strip()
        out.append({"id": "gsm8k-%d" % i, "task": "gsm8k",
                    "question": r["question"], "golden": golden, "type": "gsm8k"})
    return out

# ---------------- MATH ----------------
def extract_boxed(sol):
    best = None
    for m in re.finditer(r"\\boxed\s*\{", sol):
        i, depth, start = m.end(), 1, m.end()
        while i < len(sol) and depth:
            if sol[i] == "{":
                depth += 1
            elif sol[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            best = sol[start:i - 1]
    return best

def fetch_math(target=500):
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    per = max(1, -(-target // max(1, len(configs))))  # ceil
    all_rows = []
    for cfg in configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        except Exception as e:
            print("  skip %s: %s" % (cfg, str(e)[:80]), flush=True)
            continue
        rows = list(ds)
        for j, r in enumerate(stride_rows(rows, per)):
            sol = r.get("solution", "")
            golden = extract_boxed(sol) or sol.strip()[-80:]
            all_rows.append({"id": "math-%s-%d" % (cfg, j), "task": cfg,
                             "question": r.get("problem", ""), "golden": golden,
                             "type": "math"})
        print("  math %-24s +%d" % (cfg, per), flush=True)
    return all_rows[:target]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "bbh"):
        save("bbh_500.jsonl", fetch_bbh())
    if mode in ("all", "gsm8k"):
        save("gsm8k_500.jsonl", fetch_gsm8k())
    if mode in ("all", "math"):
        save("math_500.jsonl", fetch_math())

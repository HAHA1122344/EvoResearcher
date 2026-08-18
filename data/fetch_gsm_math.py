#!/usr/bin/env python3
"""
Fetch GSM8K and MATH and build 100-question eval subsets, matching the
format of data/bbh_100.jsonl (id, task, question, golden, type).

GSM8K  : openai/gsm8k (main), split 'test'. golden = numeric answer after '####'.
MATH   : competition_math, balanced across all 7 categories. golden = last \boxed{...}.

Deterministic stride sampling for reproducibility. No auth required.
Output: data/gsm8k_100.jsonl, data/math_100.jsonl
"""
import json, os, re
from collections import Counter
from datasets import load_dataset, get_dataset_config_names

HERE = os.path.dirname(os.path.abspath(__file__))
PER_TASK = 100  # cap

def stride_rows(rows, target):
    n = len(rows)
    step = max(1, n // max(1, target))
    return [rows[i] for i in range(0, n, step)[:target]]

# ---------------- GSM8K ----------------
def fetch_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)
    picked = stride_rows(rows, PER_TASK)
    out = []
    for i, r in enumerate(picked):
        ans = r.get("answer", "")
        m = re.search(r"####\s*(-?[\d,]+(?:\.[\d]+)?)", ans)
        golden = m.group(1).replace(",", "") if m else ans.strip()
        out.append({"id": "gsm8k-%d" % i, "task": "gsm8k",
                    "question": r["question"], "golden": golden, "type": "gsm8k"})
    return out

# ---------------- MATH ----------------
def extract_boxed(sol):
    """Return the content of the last \\boxed{...} with balanced braces (handles
    nested {\\frac{...}{...}}), or None."""
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

def fetch_math():
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    all_rows = []
    for cfg in configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        except Exception as e:
            print("  skip %s: %s" % (cfg, str(e)[:80]), flush=True)
            continue
        rows = list(ds)
        per = max(1, -(-PER_TASK // max(1, len(configs))))  # ceil: 100/7 -> 15
        for j, r in enumerate(stride_rows(rows, per)):
            sol = r.get("solution", "")
            golden = extract_boxed(sol)
            if golden is None:
                # no boxed answer: keep a numeric-looking tail if present, else full tail
                tail = sol.strip()[-80:]
                m = re.search(r"\\boxed\s*\{", sol)
                golden = tail
            all_rows.append({"id": "math-%s-%d" % (cfg, j), "task": cfg,
                             "question": r.get("problem", ""), "golden": golden,
                             "type": "math"})
    return all_rows[:PER_TASK]

def save(name, rows):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved %s (%d rows)" % (path, len(rows)), flush=True)
    print("  per task:", dict(Counter(r["task"] for r in rows)), flush=True)

if __name__ == "__main__":
    save("gsm8k_100.jsonl", fetch_gsm8k())
    save("math_100.jsonl", fetch_math())

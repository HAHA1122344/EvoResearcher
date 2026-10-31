#!/usr/bin/env python3
"""
E7 confirmatory re-run with item-level records (closes the Data availability gap).

The August E7 runs (single-shot vs depth-3 loop on GSM8K and MATH, n=500) were
reduced to aggregates; the per-item transition counts were not retained, which
is why Tables 12/14 report conservative reconstructions instead of real paired
tests.  This script re-runs the E7 cells **in one session per benchmark, with
per-item logging**, so the paired statistics become exact and releasable:

  single        one call, no critique (the E7 single-shot anchor)
  loop_plain    depth-3 loop, plain system prompt   (the E3 framing control)
  loop_balanced depth-3 loop, balanced system prompt (the E7 loop arm / protocol)

Running single + loop_balanced in one session gives the exact McNemar test and
a paired bootstrap CI for the E7 gain; adding loop_plain on MATH gives the
missing MATH framing control in the same session.  Everything is checkpointed
per item (RESUME=1 resumes; FILL=1 completes a partial file).

Env: DATASET=gsm8k|math, ARMS=comma list (default: single,loop_balanced for
gsm8k; single,loop_plain,loop_balanced for math), WORKERS, MODEL_NAME,
OPENAI_BASE_URL, OPENAI_API_KEY, SMOKE, FILL, RESUME, SEED.
Output: ../results/e7_rerun_<dataset>.json
"""
import json, re, time, os, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise SystemExit("set OPENAI_API_KEY (this script ships no key)")
API_URL = os.environ.get("OPENAI_BASE_URL", "https://zhiyaoapi.top/v1")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4.1-flash")
WORKERS = int(os.environ.get("WORKERS", "5"))
RESUME = os.environ.get("RESUME", "1") == "1"
SMOKE = int(os.environ.get("SMOKE", "0"))
FILL = os.environ.get("FILL", "0") == "1"
DATASET = os.environ.get("DATASET", "gsm8k")
SEED = int(os.environ.get("SEED", "42"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))

_HERE = os.path.dirname(os.path.abspath(__file__))

def load_jsonl(paths, n=None, seed=SEED):
    for p in paths:
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if n is not None:
                random.seed(seed); random.shuffle(rows)
                rows = rows[:n]
            return rows
    raise SystemExit("data file not found in candidates: %s" % paths)

DATA_NAME = "math_500.jsonl" if DATASET == "math" else "gsm8k_500.jsonl"
ITEMS = load_jsonl([os.path.join(_HERE, "data", DATA_NAME),
                    os.path.join(_HERE, "..", "data", DATA_NAME)])
if SMOKE:
    ITEMS = ITEMS[:SMOKE]
print("Loaded %s n=%d (SMOKE=%d) model=%s" % (DATASET.upper(), len(ITEMS), SMOKE, MODEL), flush=True)

DEFAULT_ARMS = {"gsm8k": "single,loop_balanced",
                "math": "single,loop_plain,loop_balanced"}
ARMS = [a.strip() for a in os.environ.get("ARMS", DEFAULT_ARMS[DATASET]).split(",") if a.strip()]

# ---------------- prompts (verbatim from run_e3_gsm8k.py / E7) ---------------
PLAIN_SYS = "You are a deep research assistant. Answer accurately. Output final answer in <answer> tags."
BALANCED_SYS = ("You are EvoResearcher. Balance CORRECTNESS, EFFICIENCY, REFLECTION, "
                "DIVERSITY. Output final answer in <answer> tags.")
CRITIQUE = ("Now reflect: is your previous answer fully correct and complete? "
            "If it is correct, reply exactly CONFIRMED. If not, produce a "
            "corrected final answer in <answer> tags.")
MAX_DEPTH = 3
TEMP = 0.3

SYS_OF = {"single": PLAIN_SYS, "loop_plain": PLAIN_SYS, "loop_balanced": BALANCED_SYS}

# ---------------- scoring (verbatim from run_e3_gsm8k.py) --------------------
def extract_answer(t):
    m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
    return m.group(1).strip() if m else t[:300]

def numeric(x):
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", x.replace(",", ""))
    return float(m.group()) if m else None

def strip_articles(t):
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", t)).strip()

def normalize(a):
    return strip_articles(re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(a).lower()).strip()))

def norm_latex(x):
    s = x.strip()
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", r"(\1)/(\2)", s)
    s = re.sub(r"\\pi\b", "pi", s)
    s = re.sub(r"\\infty\b", "infinity", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}\s]", "", s)
    return s.lower()

def is_correct(p, g, fmt="gsm8k"):
    p = p.strip()
    if not p:
        return False
    if fmt == "math":
        pn, gn = norm_latex(p), norm_latex(g)
        if gn and gn == pn:
            return True
        if gn and len(pn) >= 6 and len(gn) >= 6 and gn in pn:
            return True
        if "/" not in pn and "/" not in gn:
            pn_, gn_ = numeric(p), numeric(g)
            if pn_ is not None and gn_ is not None and gn_ != 0:
                if abs(pn_ - gn_) / abs(gn_) < 0.01:
                    return True
        return False
    pn, gn = normalize(p), normalize(g)
    if pn == gn: return True
    if gn and gn in pn: return True
    if gn and pn in gn: return True
    pn_, gn_ = numeric(p), numeric(g)
    if pn_ is not None and gn_ is not None and gn_ != 0:
        if abs(pn_ - gn_) / abs(gn_) < 0.01: return True
    return False

# ---------------- LLM helpers (identical retry policy) -----------------------
def is_permanent(exc):
    msg = str(exc)
    if "402" in msg or "insufficient balance" in msg.lower():
        return True
    if "401" in msg or "authentication" in msg.lower():
        return True
    return False

def ask(client, messages, temp=0.3, max_tokens=4096, retries=8):
    delay = 1.0
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=temp, max_tokens=max_tokens)
            content = r.choices[0].message.content or ""
            tok = getattr(r, "usage", None)
            tok = tok.total_tokens if tok else 0
            if content.strip():
                return content, tok
        except Exception as exc:
            if is_permanent(exc):
                print("      [FATAL %s: %s]" % (type(exc).__name__, str(exc)[:200]), flush=True)
                raise
            if attempt == retries - 1:
                print("      [retried %d times: %s %s]" % (retries, type(exc).__name__, str(exc)[:100]), flush=True)
        time.sleep(delay)
        delay = min(delay * 1.7, 12.0)
    while True:
        time.sleep(20)
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=temp, max_tokens=max_tokens)
            content = r.choices[0].message.content or ""
            tok = getattr(r, "usage", None)
            tok = tok.total_tokens if tok else 0
            if content.strip():
                return content, tok
            print("      [empty generation; retrying]", flush=True)
        except Exception as exc:
            if is_permanent(exc):
                print("      [FATAL %s: %s]" % (type(exc).__name__, str(exc)[:200]), flush=True)
                raise
            print("      [retry %s: %s]" % (type(exc).__name__, str(exc)[:100]), flush=True)

def make_client():
    return OpenAI(api_key=API_KEY, base_url=API_URL, timeout=120.0, max_retries=1)

def wilson(acc, n, z=1.96):
    p = max(0.0, min(1.0, acc)); denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0, c - m), min(1, c + m))

# ---------------- runners -----------------------------------------------------
def single_run(question):
    """E7 single-shot anchor: one call, no critique."""
    client = make_client()
    cur, tok = ask(client, [{"role": "system", "content": PLAIN_SYS},
                            {"role": "user", "content": question}], TEMP)
    return extract_answer(cur), 1, False, tok

def loop_one(question, system):
    """Depth-3 loop, fused critique-and-revise call (verbatim logic)."""
    client = make_client()
    cur, tok = ask(client, [{"role": "system", "content": system},
                            {"role": "user", "content": question}], TEMP)
    total = tok; steps = 1
    cur_ext = extract_answer(cur) or cur[:300]
    early = False
    for _ in range(MAX_DEPTH - 1):
        c, tok = ask(client, [{"role": "system", "content": system + " " + CRITIQUE},
                              {"role": "user", "content": "Question: %s\nYour previous answer: %s"
                                                          % (question, cur_ext)}], TEMP, 16384)
        total += tok; steps += 1
        if "CONFIRMED" in c.upper():
            early = True
            break
        new_ext = extract_answer(c)
        if not new_ext or new_ext == cur_ext:
            break
        cur_ext = new_ext
    return cur_ext, steps, early, total

RUNNER = {"single": lambda q: single_run(q),
          "loop_plain": lambda q: loop_one(q, PLAIN_SYS),
          "loop_balanced": lambda q: loop_one(q, BALANCED_SYS)}

# ---------------- paired analysis --------------------------------------------
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
        tail = sum(math.comb(n_pairs, i) for i in range(0, k + 1)) / (2.0 ** n_pairs)
        p = min(1.0, 2 * tail)
    diffs = [(1 if A[i]["correct"] else 0) - (1 if B[i]["correct"] else 0) for i in ids]
    delta = sum(diffs) / n
    # real paired bootstrap on item-level outcome differences (10k resamples, percentile CI)
    rng = random.Random(SEED)
    stats = []
    for _ in range(N_BOOT):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo, hi = stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]
    return {"n_paired": n, "a_only": b, "b_only": c,
            "a_name": name_a, "b_name": name_b,
            "mcnemar_exact_p": p, "delta_a_minus_b": delta,
            "bootstrap_ci": [round(lo, 4), round(hi, 4)], "bootstrap_resamples": N_BOOT,
            "bootstrap_se": (sum((s - delta) ** 2 for s in stats) / N_BOOT) ** 0.5}

# ---------------- output bookkeeping ------------------------------------------
if os.path.isdir("/content") and os.access("/content", os.W_OK):
    OUT_DIR = "/content/results"
else:
    OUT_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "e7_rerun_%s.json" % DATASET)
out = {}
if RESUME and os.path.exists(OUT_PATH):
    out = json.load(open(OUT_PATH))
    print("RESUME: loaded %s" % OUT_PATH, flush=True)

def save(quiet=False):
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["model"] = MODEL
    out["config"] = {"temp": TEMP, "max_depth": MAX_DEPTH, "n": len(ITEMS),
                     "arms": ARMS, "critique": CRITIQUE, "workers": WORKERS,
                     "seed": SEED, "note": "E7 confirmatory re-run, item-level records retained"}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    if not quiet:
        print("  [saved %s]" % OUT_PATH, flush=True)

for arm in ARMS:
    if arm not in RUNNER:
        raise SystemExit("unknown arm: %s" % arm)
    if arm in out and out[arm]["summary"]["n"] == len(ITEMS):
        print("  %s already done (n=%d), skipping" % (arm, out[arm]["summary"]["n"]), flush=True)
        continue
    items = ITEMS
    if FILL and (arm + "_progress") in out:
        got = {l["id"] for l in out[arm + "_progress"]["items"]}
        items = [it for it in ITEMS if it["id"] not in got]
        print("  FILL mode: %s has %d done, %d missing" % (arm, len(got), len(items)), flush=True)
    print("\n=== E7 re-run arm: %s (n=%d) ===" % (arm, len(items)), flush=True)
    logs = list(out[arm + "_progress"]["items"]) if (FILL and (arm + "_progress") in out) else []
    done_ids = {l["id"] for l in logs}
    todo = [it for it in items if it["id"] not in done_ids]
    def one(it, _arm=arm):
        ans, steps, early, tokens = RUNNER[_arm](it["question"])
        return {"id": it["id"], "correct": is_correct(ans, it["golden"], DATASET),
                "steps": steps, "early_stopped": early, "tokens": tokens}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in todo]
        done_n = len(logs)
        target = len(logs) + len(todo)
        for fut in as_completed(futures):
            logs.append(fut.result())
            done_n += 1
            out[arm + "_progress"] = {"items": sorted(logs, key=lambda l: l["id"])}
            save(quiet=True)                 # checkpoint after every item
            if done_n % 5 == 0 or done_n == target:
                print("      [%s %d/%d]" % (arm, done_n, target), flush=True)
    out.pop(arm + "_progress", None)
    n = len(logs)
    acc = sum(l["correct"] for l in logs) / n
    s = {"n": n, "acc": acc, "ci": list(wilson(acc, n)),
         "avg_calls": sum(l["steps"] for l in logs) / n,
         "early_stop_rate": sum(1 for l in logs if l["early_stopped"]) / n,
         "avg_tokens": sum(l["tokens"] for l in logs) / n}
    out[arm] = {"summary": s, "items": sorted(logs, key=lambda l: l["id"])}
    save()
    print("  %s: acc=%.1f%% [%.1f,%.1f] calls=%.2f early=%.0f%% tok=%.0f"
          % (arm, s["acc"] * 100, s["ci"][0] * 100, s["ci"][1] * 100,
             s["avg_calls"], s["early_stop_rate"] * 100, s["avg_tokens"]), flush=True)

out["paired"] = out.get("paired", {})
for a, b in (("single", "loop_balanced"), ("single", "loop_plain"), ("loop_plain", "loop_balanced")):
    if a in out and b in out:
        key = "%s_minus_%s" % (a, b)
        out["paired"][key] = paired_analysis(out[a]["items"], out[b]["items"], a, b)
        pr = out["paired"][key]
        print("\n=== PAIRED (%s - %s) ===" % (a, b), flush=True)
        print("  %s acc = %.1f%% | %s acc = %.1f%%"
              % (a, out[a]["summary"]["acc"] * 100, b, out[b]["summary"]["acc"] * 100), flush=True)
        print("  delta = %+.1f pp, bootstrap 95%% CI [%.1f, %.1f] pp (10k resamples)"
              % (pr["delta_a_minus_b"] * 100, pr["bootstrap_ci"][0] * 100,
                 pr["bootstrap_ci"][1] * 100), flush=True)
        print("  discordant: %s-only=%d, %s-only=%d, exact McNemar p=%.4f"
              % (a, pr["a_only"], b, pr["b_only"], pr["mcnemar_exact_p"]), flush=True)
save()

print("\nALL DONE. %s" % OUT_PATH, flush=True)

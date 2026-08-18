#!/usr/bin/env python3
"""
EvoResearcher - Training-free inference-time self-reflection study.
====================================================================
Focus cells for the "pure-reasoning, cost-bounded" reframing:

  E6  Confidence-threshold (tau) sweep   BBH depth-3, tau in {0.6,0.7,0.8,0.9} + anchors
  E7  Cross-benchmark generalization     GSM8K (100) + MATH (100): d1 single-shot vs d3 loop
  E8  Compute-cost tracking              token usage logged for every cell

Every cell uses the same frozen backbone (deepseek-v4-flash), the balanced
meta-reward framing (META_SYS), and, for the loop cells, the baseline critique
+ CONFIRMED fast path of Algorithm 1. Output is saved incrementally so a crash
mid-run can be resumed with RESUME=1.
"""
import json, re, time, os, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "pk_BGpvcHKqb0WWjOFrAnnPqU2KrWbvIPOOOer_Mz_OihE")
API_URL = os.environ.get("OPENAI_BASE_URL", "https://zyapi.tuluo.top:8888/v1")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-flash")
WORKERS = int(os.environ.get("WORKERS", "4"))
RESUME = os.environ.get("RESUME", "0") == "1"
SMOKE = int(os.environ.get("SMOKE", "0"))
SEED = int(os.environ.get("SEED", "42"))
SKIP_TAU = os.environ.get("SKIP_TAU", "0") == "1"

# ---------------- data loading ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
def load_jsonl(paths):
    for p in paths:
        if os.path.exists(p):
            return [json.loads(l) for l in open(p) if l.strip()]
    raise SystemExit("data file not found in candidates: %s" % paths)

BBH = load_jsonl([os.path.join(_HERE, "data", "bbh_100.jsonl"),
                  os.path.join(_HERE, "..", "data", "bbh_100.jsonl"),
                  "/content/data/bbh_100.jsonl"])
GSM = load_jsonl([os.path.join(_HERE, "data", "gsm8k_100.jsonl"),
                  os.path.join(_HERE, "..", "data", "gsm8k_100.jsonl"),
                  "/content/data/gsm8k_100.jsonl"])
MATH = load_jsonl([os.path.join(_HERE, "data", "math_100.jsonl"),
                   os.path.join(_HERE, "..", "data", "math_100.jsonl"),
                   "/content/data/math_100.jsonl"])
random.seed(SEED)
for ds in (BBH, GSM, MATH):
    random.shuffle(ds)
if SMOKE:
    BBH = BBH[:SMOKE]; GSM = GSM[:SMOKE]; MATH = MATH[:SMOKE]
N = len(BBH)
print(f"Loaded BBH={len(BBH)} GSM8K={len(GSM)} MATH={len(MATH)} (SMOKE={SMOKE})", flush=True)

# ---------------- answer matching ----------------
def strip_articles(t):
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", t)).strip()

def normalize(a):
    return strip_articles(re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(a).lower()).strip()))

def extract_answer(t):
    m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
    return m.group(1).strip() if m else t[:300]

def numeric(x):
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", x.replace(",", ""))
    return float(m.group()) if m else None

def norm_latex(x):
    """Crude LaTeX canonicalisation for MATH: strip \\boxed, \\text, \\frac (keep frac as (a)/(b)), spacing.
    Plain 'a/b' fractions are normalised to '(a)/(b)' so they align with \\frac output."""
    s = x.strip()
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", r"(\1)/(\2)", s)
    # keep math constants that carry value (so \frac{3}{4}\pi != \frac{3}{4})
    s = re.sub(r"\\pi\b", "pi", s)
    s = re.sub(r"\\infty\b", "infinity", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}\s]", "", s)
    return s.lower()

def is_correct(p, g, fmt="bbh"):
    p = p.strip()
    if not p:
        return False
    if fmt == "math":
        # canonicalised-LaTeX match: exact first, then a length-guarded containment
        # (both sides >= 6 chars) so short-number traps (gold '10' vs. '100',
        # gold '1' vs. '1/16') never fire via substring.
        pn, gn = norm_latex(p), norm_latex(g)
        if gn and gn == pn:
            return True
        if gn and len(pn) >= 6 and len(gn) >= 6 and gn in pn:
            return True
        # numeric tolerance only when neither side is a fraction
        if "/" not in pn and "/" not in gn:
            pn_, gn_ = numeric(p), numeric(g)
            if pn_ is not None and gn_ is not None and gn_ != 0:
                if abs(pn_ - gn_) / abs(gn_) < 0.01:
                    return True
        return False
    # bbh / gsm8k
    pn, gn = normalize(p), normalize(g)
    if pn == gn: return True
    if gn and gn in pn: return True
    if gn and pn in gn: return True
    pm = re.search(r"\b([a-f])\b", pn)
    gm = re.search(r"\b([a-f])\b", gn)
    if pm and gm and pm.group(1) == gm.group(1): return True
    pn_, gn_ = numeric(p), numeric(g)
    if pn_ is not None and gn_ is not None and gn_ != 0:
        if abs(pn_ - gn_) / abs(gn_) < 0.01: return True
    return False

# ---------------- LLM helpers ----------------
COOLDOWN_ATTEMPTS = int(os.environ.get("COOLDOWN_ATTEMPTS", "60"))

def ask(client, messages, temp=0.3, max_tokens=16384, retries=8):
    """Return (content, total_tokens). Pauses (rather than scoring empties) during
    proxy degradation episodes, then counts one empty response."""
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
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 1.7, 12.0)
    for _ in range(COOLDOWN_ATTEMPTS):
        time.sleep(20)
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=temp, max_tokens=max_tokens)
            content = r.choices[0].message.content or ""
            tok = getattr(r, "usage", None)
            tok = tok.total_tokens if tok else 0
            if content.strip():
                return content, tok
        except Exception:
            pass
    EMPTY_TOTAL[0] += 1
    return "", 0

def make_client():
    return OpenAI(api_key=API_KEY, base_url=API_URL, timeout=300.0, max_retries=1)

def wilson(acc, n, z=1.96):
    p = max(0.0, min(1.0, acc)); denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0, c - m), min(1, c + m))

# ---------------- prompts ----------------
META_SYS = ("You are EvoResearcher. Balance CORRECTNESS, EFFICIENCY, REFLECTION, "
            "DIVERSITY. Output final answer in <answer> tags.")
CONF_INSTR = " Also output <confidence>N</confidence> (0-100) with your answer."
BASELINE_CRITIQUE = ("Now reflect: is your previous answer fully correct and complete? "
                     "If it is correct, reply exactly CONFIRMED. If not, produce a "
                     "corrected final answer in <answer> tags.")

def parse_conf(t):
    m = re.search(r"<confidence>\s*(\d{1,3})\s*</confidence>", t)
    return float(m.group(1)) if m else 50.0

# ---------------- runners ----------------
EMPTY_TOTAL = [0]

def single_one(client, system, text, temp):
    out, tok = ask(client, [{"role": "system", "content": system},
                            {"role": "user", "content": text}], temp)
    return extract_answer(out), tok

def single_stats(items, system, fmt, temp=0.3, progress=True, label=""):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        client = make_client()
        ans, tok = single_one(client, system, it["question"], temp)
        return {"id": it["id"], "task": it["task"], "correct": is_correct(ans, it["golden"], fmt),
                "steps": 1, "early_stopped": False, "first_conf": None, "tokens": tok}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 25 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% [%.1f,%.1f] tok=%.0f (empty=%d)"
              % (label, s["acc"] * 100, s["ci"][0] * 100, s["ci"][1] * 100, s["avg_tokens"],
                 EMPTY_TOTAL[0] - empty0), flush=True)
    return logs

def run_loop_v2(question, golden, system, fmt, temp=0.3, max_depth=3,
                critique_prompt=None, conf_threshold=None):
    client = make_client()
    crit = critique_prompt if critique_prompt is not None else BASELINE_CRITIQUE
    sys0 = system + (CONF_INSTR if conf_threshold is not None else "")
    steps = 1
    current, tok = ask(client, [{"role": "system", "content": sys0},
                                {"role": "user", "content": question}], temp)
    total_tok = tok
    cur_ext = extract_answer(current) or current[:300]
    confs = []
    if conf_threshold is not None:
        c0 = parse_conf(current)
        confs.append(c0)
        if c0 >= conf_threshold:
            return cur_ext, steps, True, confs, total_tok
    for _ in range(max_depth - 1):
        crit_instr = crit + (CONF_INSTR if conf_threshold is not None else "")
        c, tok = ask(client, [{"role": "system", "content": system + " " + crit_instr},
                              {"role": "user", "content": "Question: %s\nYour previous answer: %s"
                                                          % (question, cur_ext)}], temp, 16384)
        total_tok += tok
        steps += 1
        if conf_threshold is not None:
            confs.append(parse_conf(c))
        if "CONFIRMED" in c.upper():
            return cur_ext, steps, True, confs, total_tok
        new_ext = extract_answer(c)
        if not new_ext or new_ext == cur_ext:
            break
        cur_ext = new_ext
        if conf_threshold is not None and confs[-1] >= conf_threshold:
            return cur_ext, steps, True, confs, total_tok
    return cur_ext, steps, False, confs, total_tok

def loop_stats_v2(items, system, fmt, critique_prompt=None, max_depth=3, conf_threshold=None,
                  temp=0.3, progress=True, label=""):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        ans, steps, early, confs, tokens = run_loop_v2(it["question"], it["golden"], system, fmt,
                                                       temp, max_depth, critique_prompt, conf_threshold)
        if not ans.strip():
            EMPTY_TOTAL[0] += 1
        return {"id": it["id"], "task": it["task"], "correct": is_correct(ans, it["golden"], fmt),
                "steps": steps, "early_stopped": early, "first_conf": confs[0] if confs else None,
                "tokens": tokens}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 25 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% steps=%.2f loop=%.0f%% early=%.0f%% tok=%.0f (empty=%d)"
              % (label, s["acc"] * 100, s["avg_steps"], s["loop_rate"] * 100, s["early_stop_rate"] * 100,
                 s["avg_tokens"], EMPTY_TOTAL[0] - empty0), flush=True)
    return logs

def summarize(logs):
    n = len(logs)
    acc = sum(l["correct"] for l in logs) / n
    avg_steps = sum(l["steps"] for l in logs) / n
    loop_rate = sum(1 for l in logs if l["steps"] > 1) / n
    early_stop_rate = sum(1 for l in logs if l["early_stopped"]) / n
    return {"n": n, "acc": acc, "ci": list(wilson(acc, n)), "avg_steps": avg_steps,
            "loop_rate": loop_rate, "early_stop_rate": early_stop_rate,
            "calls": sum(l["steps"] for l in logs),
            "avg_tokens": sum(l["tokens"] for l in logs) / max(1, n),
            "tokens": sum(l["tokens"] for l in logs)}

# ---------------- output bookkeeping ----------------
if os.path.isdir("/content") and os.access("/content", os.W_OK):
    OUT_DIR = "/content/results"
else:
    OUT_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "crossbench_study.json")
out = {}
if RESUME and os.path.exists(OUT_PATH):
    out = json.load(open(OUT_PATH))
    print("RESUME mode: loaded existing %s (%d keys)" % (OUT_PATH, len(out)), flush=True)

def save():
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("  [saved %s]" % OUT_PATH, flush=True)

# ======================================================================
print("=" * 70, flush=True)
print("E7: Cross-benchmark generalization (GSM8K / MATH, d1 vs d3)", flush=True)
print("=" * 70, flush=True)
for bench, items, fmt in (("gsm8k", GSM, "gsm8k"), ("math", MATH, "math")):
    if bench in out:
        print("  [resume] skipping %s" % bench, flush=True)
        continue
    print("  Running %s d1 single-shot x%d..." % (bench, len(items)), flush=True)
    d1 = single_stats(items, META_SYS, fmt, temp=0.3, label="%s_d1" % bench)
    print("  Running %s d3 reflective loop x%d..." % (bench, len(items)), flush=True)
    d3 = loop_stats_v2(items, META_SYS, fmt, max_depth=3, temp=0.3, label="%s_d3" % bench)
    out[bench] = {"d1": summarize(d1), "d3": summarize(d3)}
    save()

# ======================================================================
print("\n" + "=" * 70, flush=True)
print("E6: Confidence-Threshold (tau) Sweep on BBH (depth-3)", flush=True)
print("=" * 70, flush=True)
if not SKIP_TAU and "bbh_tau" not in out:
    print("  Running BBH d1 (tau=1.0 anchor) x%d..." % len(BBH), flush=True)
    d1logs = single_stats(BBH, META_SYS, "bbh", temp=0.3, label="bbh_d1")
    print("  Running BBH d3 (tau=0.0 anchor) x%d..." % len(BBH), flush=True)
    d3logs = loop_stats_v2(BBH, META_SYS, "bbh", max_depth=3, temp=0.3, label="bbh_d3")
    tau0, tau1 = summarize(d3logs), summarize(d1logs)
    rows = [{"tau": 0.0, **tau0, "cell": "run"},
            {"tau": 1.0, **tau1, "cell": "run"}]
    for t in [0.6, 0.7, 0.8, 0.9]:
        print("  Running BBH tau=%.1f x%d..." % (t, len(BBH)), flush=True)
        logs = loop_stats_v2(BBH, META_SYS, "bbh", max_depth=3, conf_threshold=t, temp=0.3,
                             label="bbh_tau_%.1f" % t)
        rows.append({"tau": t, **summarize(logs), "cell": "run"})
    baseline_calls = tau0["calls"]
    for row in rows:
        row["calls_saved"] = max(0, baseline_calls - row["calls"])
        row["calls_saved_pp"] = 100.0 * (baseline_calls - row["calls"]) / max(1, baseline_calls)
    rows.sort(key=lambda r: r["tau"])
    out["bbh_tau"] = rows
    save()

# ---------------- final ----------------
out.update({"model": MODEL, "n_bbh": len(BBH), "n_gsm8k": len(GSM), "n_math": len(MATH),
            "config": {"temp": 0.3, "seed": SEED, "max_depth": 3, "meta_system": META_SYS}})
save()
print("\nALL DONE. Saved to %s" % OUT_PATH, flush=True)

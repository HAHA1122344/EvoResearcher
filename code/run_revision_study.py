#!/usr/bin/env python3
"""
Revision round study (response to the major-revision review).
Same frozen backbone (deepseek-v4-flash), temperature 0.3, n=500 subsets.

  R1  Core loop comparison at n=500: d1 single-shot vs d3 meta-reward loop
      (BBH, GSM8K, MATH), with per-item logs + token accounting.
  R2  Named baselines at n=500: Self-Refine (Madaan et al. 2023) and
      Reflexion (Shinn et al. 2023), same three benchmarks.
  R3  Session-drift quantification: BBH n=100, d1+d3, 5 seeds -> mean+std.
  R4  Confidence calibration: BBH n=500 single-shot with <confidence> tag,
      computing expected calibration error (ECE) and the calibration curve.

All results saved incrementally to results/revision_study.json; RESUME=1 resumes.
"""
import json, re, time, os, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "pk_FzzzKdTB4DBT3fiEvSK-bZT1R7oNUmGqMPFRUKNGDmM")
API_URL = os.environ.get("OPENAI_BASE_URL", "https://zyapi.tuluo.top:8888/v1")
MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-flash")
WORKERS = int(os.environ.get("WORKERS", "4"))
RESUME = os.environ.get("RESUME", "0") == "1"
SMOKE = int(os.environ.get("SMOKE", "0"))
SEED = int(os.environ.get("SEED", "42"))
SKIP = set(os.environ.get("SKIP", "").split(",")) - {""}

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

BBH = load_jsonl([os.path.join(_HERE, "data", "bbh_500.jsonl"),
                  os.path.join(_HERE, "..", "data", "bbh_500.jsonl")])
GSM = load_jsonl([os.path.join(_HERE, "data", "gsm8k_500.jsonl"),
                  os.path.join(_HERE, "..", "data", "gsm8k_500.jsonl")])
MATH = load_jsonl([os.path.join(_HERE, "data", "math_500.jsonl"),
                   os.path.join(_HERE, "..", "data", "math_500.jsonl")])
BBH100 = load_jsonl([os.path.join(_HERE, "data", "bbh_100.jsonl"),
                     os.path.join(_HERE, "..", "data", "bbh_100.jsonl")])
if SMOKE:
    BBH = BBH[:SMOKE]; GSM = GSM[:SMOKE]; MATH = MATH[:SMOKE]
    BBH100 = BBH100[:SMOKE]
print("Loaded BBH=%d GSM8K=%d MATH=%d BBH100=%d (SMOKE=%d)" %
      (len(BBH), len(GSM), len(MATH), len(BBH100), SMOKE), flush=True)

# ---------------- answer matching (same as crossbench study) ----------------
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

def is_correct(p, g, fmt="bbh"):
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
    # Persist: a transient upstream (DeepSeek key pool) outage must never turn a
    # generation into an empty answer (which would be scored wrong and silently
    # corrupt accuracy) -- wait until the proxy recovers instead of giving up.
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
        except Exception:
            pass

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
BASELINE_CRITIQUE = ("Now reflect: is your previous answer fully correct and complete? "
                     "If it is correct, reply exactly CONFIRMED. If not, produce a "
                     "corrected final answer in <answer> tags.")
CONF_INSTR = " Also output <confidence>N</confidence> (0-100) with your answer."

SR_SYS = ("You are a careful reasoning assistant. Output your final answer in "
          "<answer> tags.")
SR_FEEDBACK = ("Review your answer to the question. Identify any errors, omissions, "
               "or ways it could be improved. Give specific, actionable feedback only. "
               "Do NOT output a final answer.")
SR_IMPROVE = ("Using the feedback, produce a corrected and improved final answer "
              "in <answer> tags.")

RX_SYS = ("You are a reasoning agent that evaluates and improves its own answers. "
          "Output your final answer in <answer> tags.")
RX_EVAL = ("Evaluate the correctness of your answer to the question. "
           "Reply with exactly one score: <score>N</score> where N is 0-100 "
           "(100 = fully correct). Do not give a new answer.")
RX_REFLECT = ("Your answer was not good enough. Reflect on why it was wrong and "
              "what the correct reasoning would be. Write a concise reflection.")
RX_IMPROVE = ("Question: %s\nYour previous answer: %s\nYour reflection: %s\n"
              "Now produce a correct final answer in <answer> tags.")

def parse_conf(t):
    m = re.search(r"<confidence>\s*(\d{1,3})\s*</confidence>", t)
    return float(m.group(1)) if m else None

def parse_score(t):
    m = re.search(r"<score>\s*(\d{1,3})\s*</score>", t)
    return float(m.group(1)) if m else None

EMPTY_TOTAL = [0]

# ---------------- runners ----------------
def single_stats(items, system, fmt, temp=0.3, progress=True, label="", conf_instr=False):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        client = make_client()
        sysp = system + (CONF_INSTR if conf_instr else "")
        out, tok = ask(client, [{"role": "system", "content": sysp},
                                {"role": "user", "content": it["question"]}], temp)
        conf = parse_conf(out) if conf_instr else None
        return {"id": it["id"], "task": it["task"],
                "correct": is_correct(extract_answer(out), it["golden"], fmt),
                "steps": 1, "early_stopped": False, "tokens": tok, "conf": conf}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 50 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% [%.1f,%.1f] tok=%.0f (empty=%d)"
              % (label, s["acc"] * 100, s["ci"][0] * 100, s["ci"][1] * 100,
                 s["avg_tokens"], EMPTY_TOTAL[0] - empty0), flush=True)
    return logs

def meta_loop_one(question, golden, fmt, system, max_depth=3, temp=0.3):
    client = make_client()
    cur, tok = ask(client, [{"role": "system", "content": system},
                            {"role": "user", "content": question}], temp)
    total = tok; steps = 1
    cur_ext = extract_answer(cur) or cur[:300]
    for _ in range(max_depth - 1):
        c, tok = ask(client, [{"role": "system", "content": system + " " + BASELINE_CRITIQUE},
                              {"role": "user", "content": "Question: %s\nYour previous answer: %s"
                                                          % (question, cur_ext)}], temp, 16384)
        total += tok; steps += 1
        if "CONFIRMED" in c.upper():
            return cur_ext, steps, True, total
        new_ext = extract_answer(c)
        if not new_ext or new_ext == cur_ext:
            break
        cur_ext = new_ext
    return cur_ext, steps, False, total

def loop_stats(items, system, fmt, max_depth=3, temp=0.3, progress=True, label=""):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        ans, steps, early, tokens = meta_loop_one(it["question"], it["golden"], fmt,
                                                  system, max_depth, temp)
        if not ans.strip():
            EMPTY_TOTAL[0] += 1
        return {"id": it["id"], "task": it["task"],
                "correct": is_correct(ans, it["golden"], fmt),
                "steps": steps, "early_stopped": early, "tokens": tokens}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 50 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% steps=%.2f early=%.0f%% tok=%.0f (empty=%d)"
              % (label, s["acc"] * 100, s["avg_steps"], s["early_stop_rate"] * 100,
                 s["avg_tokens"], EMPTY_TOTAL[0] - empty0), flush=True)
    return logs

def self_refine_one(question, golden, fmt, rounds=1, temp=0.3):
    """Self-Refine (Madaan et al. 2023): generate -> feedback -> improve (x rounds)."""
    client = make_client()
    cur, tok = ask(client, [{"role": "system", "content": SR_SYS},
                            {"role": "user", "content": question}], temp)
    total = tok; steps = 1
    cur_ext = extract_answer(cur) or cur[:300]
    for _ in range(rounds):
        fb, tok = ask(client, [{"role": "system", "content": SR_SYS + " " + SR_FEEDBACK},
                               {"role": "user", "content": "Question: %s\nYour answer: %s"
                                                           % (question, cur_ext)}], temp, 16384)
        total += tok; steps += 1
        improved, tok = ask(client, [{"role": "system", "content": SR_SYS + " " + SR_IMPROVE},
                                     {"role": "user",
                                      "content": "Question: %s\nYour answer: %s\nFeedback: %s"
                                                 % (question, cur_ext, fb)}], temp, 16384)
        total += tok; steps += 1
        new_ext = extract_answer(improved)
        if new_ext:
            cur_ext = new_ext
    return cur_ext, steps, False, total

def self_refine_stats(items, fmt, rounds=1, temp=0.3, progress=True, label=""):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        ans, steps, early, tokens = self_refine_one(it["question"], it["golden"], fmt,
                                                    rounds, temp)
        if not ans.strip():
            EMPTY_TOTAL[0] += 1
        return {"id": it["id"], "task": it["task"],
                "correct": is_correct(ans, it["golden"], fmt),
                "steps": steps, "early_stopped": False, "tokens": tokens}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 50 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% calls=%.2f tok=%.0f"
              % (label, s["acc"] * 100, s["avg_steps"], s["avg_tokens"]), flush=True)
    return logs

def reflexion_one(question, golden, fmt, max_rounds=2, temp=0.3, eval_threshold=70.0):
    """Reflexion (Shinn et al. 2023): generate -> self-evaluate -> reflect -> regenerate."""
    client = make_client()
    cur, tok = ask(client, [{"role": "system", "content": RX_SYS},
                            {"role": "user", "content": question}], temp)
    total = tok; steps = 1
    cur_ext = extract_answer(cur) or cur[:300]
    reflection = ""
    for _ in range(max_rounds):
        ev, tok = ask(client, [{"role": "system", "content": RX_SYS + " " + RX_EVAL},
                               {"role": "user", "content": "Question: %s\nYour answer: %s"
                                                           % (question, cur_ext)}], temp, 16384)
        total += tok; steps += 1
        score = parse_score(ev)
        if score is not None and score >= eval_threshold:
            break
        refl, tok = ask(client, [{"role": "system", "content": RX_SYS + " " + RX_REFLECT},
                                 {"role": "user", "content": "Question: %s\nYour answer: %s"
                                                             % (question, cur_ext)}], temp, 16384)
        total += tok; steps += 1
        reflection = refl.strip()
        imp, tok = ask(client, [{"role": "system", "content": RX_SYS},
                                {"role": "user", "content": RX_IMPROVE
                                                           % (question, cur_ext, reflection)}], temp, 16384)
        total += tok; steps += 1
        new_ext = extract_answer(imp)
        if new_ext:
            cur_ext = new_ext
    return cur_ext, steps, False, total

def reflexion_stats(items, fmt, temp=0.3, progress=True, label=""):
    logs = []
    empty0 = EMPTY_TOTAL[0]
    def one(it):
        ans, steps, early, tokens = reflexion_one(it["question"], it["golden"], fmt,
                                                  max_rounds=2, temp=temp)
        if not ans.strip():
            EMPTY_TOTAL[0] += 1
        return {"id": it["id"], "task": it["task"],
                "correct": is_correct(ans, it["golden"], fmt),
                "steps": steps, "early_stopped": False, "tokens": tokens}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            logs.append(fut.result())
            done += 1
            if progress and done % 50 == 0:
                print("      [%d/%d]" % (done, len(items)), flush=True)
    if progress:
        s = summarize(logs)
        print("      %s acc=%.1f%% calls=%.2f tok=%.0f"
              % (label, s["acc"] * 100, s["avg_steps"], s["avg_tokens"]), flush=True)
    return logs

def summarize(logs):
    n = len(logs)
    acc = sum(l["correct"] for l in logs) / n
    avg_steps = sum(l["steps"] for l in logs) / n
    early_stop_rate = sum(1 for l in logs if l["early_stopped"]) / n
    return {"n": n, "acc": acc, "ci": list(wilson(acc, n)), "avg_steps": avg_steps,
            "early_stop_rate": early_stop_rate,
            "calls": sum(l["steps"] for l in logs),
            "avg_tokens": sum(l["tokens"] for l in logs) / max(1, n)}

def ece(logs, bins=10):
    """Expected calibration error over 10 equal-width confidence bins."""
    bins_n = [0] * bins; bins_acc = [0.0] * bins
    for l in logs:
        c = l["conf"] / 100.0 if l["conf"] is not None else None
        if c is None:
            continue
        b = min(bins - 1, int(c * bins))
        bins_n[b] += 1
        bins_acc[b] += int(l["correct"])
    tot = sum(bins_n); ece = 0.0; curve = []
    for b in range(bins):
        if bins_n[b]:
            acc = bins_acc[b] / bins_n[b]
            conf = (b + 0.5) / bins
            ece += (bins_n[b] / tot) * abs(acc - conf)
            curve.append({"bin": conf, "n": bins_n[b], "acc": acc})
    return ece, curve

# ---------------- output bookkeeping ----------------
if os.path.isdir("/content") and os.access("/content", os.W_OK):
    OUT_DIR = "/content/results"
else:
    OUT_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "revision_study.json")
out = {}
if RESUME and os.path.exists(OUT_PATH):
    out = json.load(open(OUT_PATH))
    print("RESUME: loaded %s (%d keys)" % (OUT_PATH, len(out)), flush=True)

def save():
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["model"] = MODEL
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("  [saved %s]" % OUT_PATH, flush=True)

# ======================================================================
# R1: core loop comparison at n=500
# ======================================================================
if "r1_core" not in out and "R1" not in SKIP:
    print("\n" + "=" * 70, flush=True)
    print("R1: core loop comparison at n=500", flush=True)
    print("=" * 70, flush=True)
    r1 = {}
    for bench, items, fmt in (("bbh", BBH, "bbh"), ("gsm8k", GSM, "gsm8k"), ("math", MATH, "math")):
        print("  %s: d1 single-shot x%d ..." % (bench, len(items)), flush=True)
        d1 = single_stats(items, META_SYS, fmt, label="%s_d1" % bench)
        print("  %s: d3 meta loop x%d ..." % (bench, len(items)), flush=True)
        d3 = loop_stats(items, META_SYS, fmt, max_depth=3, label="%s_d3" % bench)
        r1[bench] = {"d1": summarize(d1), "d3": summarize(d3),
                     "d1_items": d1, "d3_items": d3}
        save()
    out["r1_core"] = r1
    save()

# ======================================================================
# R2: named baselines at n=500 (Self-Refine, Reflexion)
# ======================================================================
if "r2_baselines" not in out and "R2" not in SKIP:
    print("\n" + "=" * 70, flush=True)
    print("R2: named baselines at n=500 (Self-Refine, Reflexion)", flush=True)
    print("=" * 70, flush=True)
    r2 = {}
    for bench, items, fmt in (("bbh", BBH, "bbh"), ("gsm8k", GSM, "gsm8k"), ("math", MATH, "math")):
        r2[bench] = {}
        print("  %s: Self-Refine (1 refine round) x%d ..." % (bench, len(items)), flush=True)
        sr = self_refine_stats(items, fmt, rounds=1, label="%s_selfrefine" % bench)
        r2[bench]["self_refine"] = summarize(sr)
        save()
        print("  %s: Reflexion (self-eval gate) x%d ..." % (bench, len(items)), flush=True)
        rx = reflexion_stats(items, fmt, label="%s_reflexion" % bench)
        r2[bench]["reflexion"] = summarize(rx)
        save()
    out["r2_baselines"] = r2
    save()

# ======================================================================
# R3: session-drift quantification (BBH n=100, 5 seeds, d1 + d3)
# ======================================================================
if "r3_repeats" not in out and "R3" not in SKIP:
    print("\n" + "=" * 70, flush=True)
    print("R3: repeated runs BBH n=100, seeds {42,123,2024,7,999}", flush=True)
    print("=" * 70, flush=True)
    seeds = [42, 123, 2024, 7, 999]
    r3 = {"d1": {"accs": [], "steps": [], "tokens": []},
          "d3": {"accs": [], "steps": [], "early": [], "tokens": []}}
    n_r3 = SMOKE if SMOKE else 100
    for sd in seeds:
        rows = load_jsonl([os.path.join(_HERE, "data", "bbh_100.jsonl"),
                           os.path.join(_HERE, "..", "data", "bbh_100.jsonl")], n=n_r3, seed=sd)
        random.seed(sd); random.shuffle(rows)
        d1 = single_stats(rows, META_SYS, "bbh", label="seed%d_d1" % sd)
        s1 = summarize(d1)
        d3 = loop_stats(rows, META_SYS, "bbh", max_depth=3, label="seed%d_d3" % sd)
        s3 = summarize(d3)
        r3["d1"]["accs"].append(s1["acc"]); r3["d1"]["steps"].append(s1["avg_steps"])
        r3["d1"]["tokens"].append(s1["avg_tokens"])
        r3["d3"]["accs"].append(s3["acc"]); r3["d3"]["steps"].append(s3["avg_steps"])
        r3["d3"]["early"].append(s3["early_stop_rate"]); r3["d3"]["tokens"].append(s3["avg_tokens"])
        print("  seed %d: d1 %.1f%%, d3 %.1f%%, early %.0f%%"
              % (sd, s1["acc"] * 100, s3["acc"] * 100, s3["early_stop_rate"] * 100), flush=True)
        save()
    out["r3_repeats"] = r3
    save()

# ======================================================================
# R4: confidence calibration (ECE) on BBH n=500 single-shot
# ======================================================================
if "r4_ece" not in out and "R4" not in SKIP:
    print("\n" + "=" * 70, flush=True)
    print("R4: confidence calibration (ECE) BBH n=500 single-shot", flush=True)
    print("=" * 70, flush=True)
    conf_logs = single_stats(BBH, META_SYS, "bbh", conf_instr=True, label="bbh_conf")
    n_conf = sum(1 for l in conf_logs if l["conf"] is not None)
    e, curve = ece(conf_logs)
    s = summarize(conf_logs)
    mean_conf = sum((l["conf"] or 0) for l in conf_logs) / max(1, n_conf)
    out["r4_ece"] = {"n": len(conf_logs), "n_with_conf": n_conf,
                     "acc": s["acc"], "ci": s["ci"],
                     "mean_conf": mean_conf, "ece": e, "curve": curve}
    print("  acc=%.1f%% mean_conf=%.1f ECE=%.4f (n_conf=%d)"
          % (s["acc"] * 100, mean_conf, e, n_conf), flush=True)
    for row in curve:
        print("    bin %.2f n=%d acc=%.2f" % (row["bin"], row["n"], row["acc"]), flush=True)
    save()

out.update({"config": {"temp": 0.3, "max_depth": 3, "model": MODEL}})
save()
print("\nALL DONE. %s" % OUT_PATH, flush=True)

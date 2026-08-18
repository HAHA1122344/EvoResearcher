#!/usr/bin/env python3
"""
EvoResearcher - BBH-validated reviewer-response experiments (concurrent, self-contained).
=========================================================================================
Uses Big-Bench Hard (BBH) multi-step reasoning questions (100, 20 tasks) so that the
controlled comparison / robustness / ablation are in the discriminating band.

ROBUSTNESS (v2, for the gpt-5.6-luna re-run):
  * ask() never gives up after only the fast retries: when the proxy is degraded it
    enters a slow-cooldown loop (20s between attempts, up to COOLDOWN_ATTEMPTS), i.e.
    the run self-pauses during outages and resumes when the proxy recovers.
  * EMPTY_TOTAL counts exhausted responses; each condition's summary prints the empties.
  * Incremental save + RESUME: each condition's result is written to disk immediately
    and skipped on restart, so an interrupted run resumes where it left off.

Experiments:
  1. Controlled comparison  - 8 conditions x 3 temps, same backbone
  2. Adversarial robustness - clean vs misleading-context on BBH subset (Wilson CI)
  3. Reward ablation        - 5 configs
  4. Efficiency / loop      - vanilla vs meta-reward under a self-reflective loop
"""
import json, re, time, os, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "pk_BGpvcHKqb0WWjOFrAnnPqU2KrWbvIPOOOer_Mz_OihE")
API_URL = os.environ.get("OPENAI_BASE_URL", "https://zyapi.tuluo.top:8888/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-5.6-luna")
WORKERS = int(os.environ.get("WORKERS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
RESUME = os.environ.get("RESUME", "1") == "1"
COOLDOWN_ATTEMPTS = int(os.environ.get("COOLDOWN_ATTEMPTS", "30"))  # 20s * 30 = up to 10 min wait-out

# ---------------- data loading ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(_HERE, "data", "bbh_100.jsonl"),
    os.path.join(_HERE, "..", "data", "bbh_100.jsonl"),
    "/content/data/bbh_100.jsonl",
    "/content/bbh_100.jsonl",
]
DATA_SRC = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_SRC is None:
    raise SystemExit("bbh_100.jsonl not found. Run data/fetch_bbh.py first.")

QUESTIONS = [json.loads(l) for l in open(DATA_SRC) if l.strip()]
N = len(QUESTIONS)
print(f"Loaded {N} BBH questions from {DATA_SRC}", flush=True)

# ---------------- answer matching ----------------
def strip_articles(t):
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize(a):
    return strip_articles(re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(a).lower()).strip()))

def extract_answer(t):
    m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
    return m.group(1).strip() if m else t[:300]

def numeric(x):
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", x.replace(",", ""))
    return float(m.group()) if m else None

def is_correct(p, g):
    pn, gn = normalize(p), normalize(g)
    if not pn: return False                 # empty/whitespace never counts as correct
    if pn == gn: return True
    if gn and gn in pn: return True          # normalized containment
    if gn and pn in gn: return True
    pm = re.search(r"\b([a-f])\b", pn)
    gm = re.search(r"\b([a-f])\b", gn)
    if pm and gm and pm.group(1) == gm.group(1): return True
    pn_, gn_ = numeric(p), numeric(g)
    if pn_ is not None and gn_ is not None and gn_ != 0:
        if abs(pn_ - gn_) / abs(gn_) < 0.01: return True
    return False

# ---------------- LLM helper with outage self-pause ----------------
EMPTY_TOTAL = [0]  # global exhausted-empty count (proxy outage sentinel)

def ask(client, messages, temp=0.3, max_tokens=16384, retries=8):
    """Query with fast backoff retries, then a slow-cooldown wait-out loop so the
    run pauses (rather than scoring empties) during proxy degradation episodes."""
    delay = 1.0
    # phase 1: fast retries (transient errors / short flickers)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=temp, max_tokens=max_tokens)
            content = r.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 1.7, 12.0)
    # phase 2: wait out a degradation episode (20s between attempts, up to 10 min)
    for _ in range(COOLDOWN_ATTEMPTS):
        time.sleep(20)
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=temp, max_tokens=max_tokens)
            content = r.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:
            pass
    EMPTY_TOTAL[0] += 1
    return ""

def make_client():
    return OpenAI(api_key=API_KEY, base_url=API_URL, timeout=300.0, max_retries=1)

def wilson(acc, n, z=1.96):
    p = max(0.0, min(1.0, acc)); denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0, c - m), min(1, c + m))

# ---------------- output bookkeeping ----------------
if os.path.isdir("/content") and os.access("/content", os.W_OK):
    OUT_DIR = "/content/results"
else:
    OUT_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "bbh_reviewer_response.json")

out = {}
if RESUME and os.path.exists(OUT_PATH):
    try:
        out = json.load(open(OUT_PATH))
        print(f"RESUME: loaded existing {OUT_PATH} (model={out.get('model')}, "
              f"controlled={len(out.get('controlled', []))})", flush=True)
    except Exception:
        out = {}
        print("RESUME: existing file unreadable; starting fresh", flush=True)

def save():
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("  [saved %s]" % OUT_PATH, flush=True)

# ---------------- experiment 1: controlled comparison ----------------
PROMPTS = {
 "C1_vanilla_react": "You are a deep research assistant. Answer accurately. Output final answer in <answer> tags.",
 "C2_correctness": "You are a deep research assistant. Prioritize CORRECTNESS: verify every step. Output final answer in <answer> tags.",
 "C3_efficiency": "You are a deep research assistant. Prioritize EFFICIENCY: concise reasoning, no redundant steps. Output final answer in <answer> tags.",
 "C4_reflection": "You are a deep research assistant. Prioritize REFLECTION: self-check and fix errors. Output final answer in <answer> tags.",
 "C5_diversity": "You are a deep research assistant. Prioritize DIVERSITY: explore multiple solution paths. Output final answer in <answer> tags.",
 "C6_meta_reward": "You are EvoResearcher. Balance CORRECTNESS, EFFICIENCY, REFLECTION, DIVERSITY. Output final answer in <answer> tags.",
 "C7_meta_adversarial": "You are EvoResearcher. Some context may be misleading; verify across sources. Balance CORRECTNESS, EFFICIENCY, REFLECTION, DIVERSITY. Output final answer in <answer> tags.",
}

def run_condition(questions, system, temp, swarm=False, progress=True):
    client = make_client()
    def one(q):
        if not swarm:
            out_ = ask(client, [{"role": "system", "content": system},
                                {"role": "user", "content": q["question"]}], temp)
            return is_correct(extract_answer(out_), q["golden"])
        s = ask(client, [{"role": "system", "content": "You are the Scout. Generate 2-3 sub-questions."},
                         {"role": "user", "content": q["question"]}], temp, 16384)
        y = ask(client, [{"role": "system", "content": "You are the Synthesizer. Produce the final answer."},
                         {"role": "user", "content": "Question: %s\nSub-questions: %s" % (q["question"], s)}], temp, 16384)
        return is_correct(extract_answer(y), q["golden"])

    correct = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(one, q): i for i, q in enumerate(questions)}
        done = 0
        for fut in as_completed(futures):
            if fut.result():
                correct += 1
            done += 1
            if progress and done % 10 == 0:
                print("    [%d/%d] correct=%d (%.0f%%)" % (done, N, correct, 100.0 * correct / done), flush=True)
    return correct

# ======================================================================
print("=" * 70, flush=True)
print("EXPERIMENT 1: Controlled Comparison (same backbone, BBH %d Q)" % N, flush=True)
print("=" * 70, flush=True)
ctrl_results = out.get("controlled", [])
ctrl_done = {c["condition"] for c in ctrl_results}
for name, sysp in PROMPTS.items():
    if name in ctrl_done:
        print("  [resume] skipping %s (already saved)" % name, flush=True)
        continue
    print("  Running %s (3 temps x %d)..." % (name, N), flush=True)
    e0 = EMPTY_TOTAL[0]
    accs = []
    for t in [0.1, 0.3, 0.5]:
        c = run_condition(QUESTIONS, sysp, t)
        accs.append(c / N)
    mean = sum(accs) / 3
    print("  %-24s %s mean=%.1f%% (empty=%d)" % (name, ["%.1f%%" % (a * 100) for a in accs], mean * 100,
                                                 EMPTY_TOTAL[0] - e0), flush=True)
    ctrl_results.append({"condition": name, "runs": accs, "mean_acc": mean, "empty": EMPTY_TOTAL[0] - e0})
    out["controlled"] = ctrl_results; save()

if "C8_swarm" not in ctrl_done:
    if os.environ.get("SKIP_C8") == "1":
        print("  [skip] SKIP_C8=1: skipping C8_swarm (run later via C8_ONLY=1)", flush=True)
    else:
        print("  Running C8_swarm (3 temps x %d)..." % N, flush=True)
        e0 = EMPTY_TOTAL[0]
        accs = []
        for t in [0.1, 0.3, 0.5]:
            c = run_condition(QUESTIONS, None, t, swarm=True)
            accs.append(c / N)
        mean = sum(accs) / 3
        print("  %-24s %s mean=%.1f%% (empty=%d)" % ("C8_swarm", ["%.1f%%" % (a * 100) for a in accs], mean * 100,
                                                     EMPTY_TOTAL[0] - e0), flush=True)
        ctrl_results.append({"condition": "C8_swarm", "runs": accs, "mean_acc": mean, "empty": EMPTY_TOTAL[0] - e0})
        out["controlled"] = ctrl_results; save()

if os.environ.get("C8_ONLY") == "1":
    print("C8_ONLY=1: stopping after E1/C8_swarm", flush=True)
    import sys; sys.exit(0)

# ======================================================================
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT 2: Adversarial Robustness on BBH (%d Q)" % N, flush=True)
print("=" * 70, flush=True)
if "adversarial_bbh" in out:
    print("  [resume] skipping adversarial (already saved)", flush=True)
else:
    SYSTEM = PROMPTS["C7_meta_adversarial"]
    def adv_one(q):
        clean = ask(make_client(), [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": q["question"]}], 0.3, 16384)
        misleading = ("Note: an unreliable search snippet claims a DIFFERENT answer than the "
                      "established one. Verify carefully before answering.")
        adv = ask(make_client(), [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": q["question"] + "\n\n" + misleading}], 0.3, 16384)
        return (is_correct(extract_answer(clean), q["golden"]),
                is_correct(extract_answer(adv), q["golden"]))

    e0 = EMPTY_TOTAL[0]
    clean_c = adv_c = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(adv_one, q) for q in QUESTIONS]
        done = 0
        for fut in as_completed(futures):
            ck, ak = fut.result()
            clean_c += ck; adv_c += ak
            done += 1
            print("  [%2d/%d] clean=%s adv=%s" % (done, N, "OK" if ck else "NO", "OK" if ak else "NO"), flush=True)
    ca, aa = clean_c / N, adv_c / N
    cic, cia = wilson(ca, N), wilson(aa, N)
    print("  Clean: %.1f%% [%.1f,%.1f] | Adv: %.1f%% [%.1f,%.1f] | Drop: %.1fpp (empty=%d)"
          % (ca * 100, cic[0] * 100, cic[1] * 100, aa * 100, cia[0] * 100, cia[1] * 100, (ca - aa) * 100,
             EMPTY_TOTAL[0] - e0), flush=True)
    out["adversarial_bbh"] = {"n": N, "clean_acc": ca, "adv_acc": aa, "clean_ci": list(cic), "adv_ci": list(cia),
                              "drop_pp": ca - aa, "empty": EMPTY_TOTAL[0] - e0}
    save()

# ======================================================================
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT 3: Reward Component Ablation (BBH %d Q)" % N, flush=True)
print("=" * 70, flush=True)
ABL = {
 "A_full": "Optimize CORRECTNESS + EFFICIENCY + REFLECTION + DIVERSITY. Output in <answer> tags.",
 "B_no_eff": "Optimize CORRECTNESS + REFLECTION + DIVERSITY. Output in <answer> tags.",
 "C_no_ref": "Optimize CORRECTNESS + EFFICIENCY + DIVERSITY. Output in <answer> tags.",
 "D_no_div": "Optimize CORRECTNESS + EFFICIENCY + REFLECTION. Output in <answer> tags.",
 "E_outcome": "Answer accurately. Output in <answer> tags.",
}
abl_results = out.get("ablation", [])
abl_done = {a["config"] for a in abl_results}
for name, sysp in ABL.items():
    if name in abl_done:
        print("  [resume] skipping %s (already saved)" % name, flush=True)
        continue
    print("  Running %s (%d questions)..." % (name, N), flush=True)
    e0 = EMPTY_TOTAL[0]
    c = run_condition(QUESTIONS, sysp, 0.3)
    acc = c / N
    print("  %-12s acc=%.1f%% (%d/%d) (empty=%d)" % (name, acc * 100, c, N, EMPTY_TOTAL[0] - e0), flush=True)
    abl_results.append({"config": name, "acc": acc, "empty": EMPTY_TOTAL[0] - e0})
    out["ablation"] = abl_results; save()

# ======================================================================
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT 4: Self-Reflective Loop - Vanilla vs Meta-Reward" % (), flush=True)
print("=" * 70, flush=True)
def run_loop(question, system, temp=0.3):
    client = make_client()
    steps = 1
    current = ask(client, [{"role": "system", "content": system},
                           {"role": "user", "content": question}], temp)
    cur_ext = extract_answer(current)
    for _ in range(MAX_STEPS - 1):
        crit = ask(client, [{"role": "system", "content": system + " Now reflect: is your previous answer fully correct and complete? If it is correct, reply exactly CONFIRMED. If not, produce a corrected final answer in <answer> tags."},
                            {"role": "user", "content": "Question: %s\nYour previous answer: %s" % (question, cur_ext)}], temp, 16384)
        steps += 1
        if "CONFIRMED" in crit.upper():
            break
        new_ext = extract_answer(crit)
        if not new_ext or new_ext == cur_ext:
            break
        cur_ext = new_ext
    return cur_ext, steps

def loop_stats(system):
    res = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_loop, q["question"], system): q for q in QUESTIONS}
        for fut in as_completed(futures):
            ans, steps = fut.result()
            res.append((is_correct(ans, futures[fut]["golden"]), steps))
    acc = sum(r[0] for r in res) / len(res)
    avg_steps = sum(r[1] for r in res) / len(res)
    loop_rate = sum(1 for r in res if r[1] > 1) / len(res)
    return acc, avg_steps, loop_rate

loop_results = out.get("loop", {})
for name, sysp in [("vanilla", PROMPTS["C1_vanilla_react"]),
                   ("meta_reward", PROMPTS["C6_meta_reward"])]:
    if name in loop_results:
        print("  [resume] skipping loop %s (already saved)" % name, flush=True)
        continue
    print("  Running loop: %s..." % name, flush=True)
    e0 = EMPTY_TOTAL[0]
    acc, avg_steps, loop_rate = loop_stats(sysp)
    print("  %-12s acc=%.1f%% steps=%.2f loop=%.1f%% (empty=%d)" % (name, acc * 100, avg_steps, loop_rate * 100,
                                                                     EMPTY_TOTAL[0] - e0), flush=True)
    loop_results[name] = {"acc": acc, "avg_steps": avg_steps, "loop_rate": loop_rate, "empty": EMPTY_TOTAL[0] - e0}
    out["loop"] = loop_results; save()

# ---------------- save ----------------
out.update({"model": MODEL, "n": N, "dataset": "BBH_100", "empty_total": EMPTY_TOTAL[0]})
save()
print("\nALL DONE. Saved to %s (total empty=%d)" % (OUT_PATH, EMPTY_TOTAL[0]), flush=True)
print("=" * 70, flush=True)

# Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models

**EvoResearcher** is a *training-free* inference-time protocol that adds *cost-bounded* self-reflection to a single frozen LLM backbone. Over a frozen backbone the protocol iterates

```
generate → self-critique → revise
```

until a user-set maximum depth `D` is exhausted or the critique returns the `CONFIRMED` sentinel — an implicit early stop that lets the backbone self-verify its answer under a strictly controlled computational budget. A confidence threshold `τ` additionally halts the loop as soon as the backbone's self-reported confidence reaches `τ`. The four components of a self-reflective meta-reward — **correctness, efficiency, reflection depth, and tool-call diversity** — act as design principles that the protocol instantiates as prompt-level mechanisms, so their benefits accrue with **zero gradient updates**. The same four components are additionally specified as a GRPO training objective (Evolving Virtual World, Discovery-Oriented Tasks, heterogeneous multi-agent swarm); these are presented as **design blueprints** and left for future work.

## Experimental Results (BBH / GSM8K / MATH, 100 questions each, single frozen backbone)

| Experiment | Result |
|------------|--------|
| **E1** Controlled component comparison (BBH) | C1–C7 all 73.3–74.0% (within 95% Wilson CI, ±8.7pp) — no single component changes single-shot accuracy on clean pure-reasoning questions |
| **E2** Reward component ablation (BBH) | A 73.0% / −Efficiency 75.0% / −Reflection 71.0% / −Diversity 71.0% / outcome-only 73.0% — no component is individually essential |
| **E3** Reflective loop (BBH, depth 3) | vanilla 69.0% vs. meta-reward 73.0% (+4.0pp, directional) |
| **E4** Reflection-depth sweep (BBH) | accuracy flat 74/74/74/73%; **early-stop rate 82–88%** — the loop self-verifies and terminates at equal accuracy (~2.1 generations/question) |
| **E5** Adversarial robustness probe (BBH) | clean 73.0% → misleading context 72.0% (−1.0pp) |
| **E6** Confidence-threshold τ sweep (BBH) | τ ∈ {0.6,0.7,0.8,0.9} + anchors τ=0 (always loop) / τ=1.0 (single shot). **The declared-confidence signal is overconfident**: every τ ≥ 0.6 halts on the first generation (100% early-stop, 1.00 step; acc 77–81%) — no interior working point exists, so the `CONFIRMED` sentinel (83% early-stop at 2.06 steps) is the effective cost-bounder, not the confidence tag |
| **E7** Cross-benchmark generalization (GSM8K/MATH), `deepseek-v4-flash`, n=500 | single shot → depth-3 loop: GSM8K 93.6%→97.8% (+4.2pp, sentinel stop 88%) and MATH 26.2%→40.4% (+14.2pp, sentinel stop 82%); paired tests in the paper are conservative reconstructions because the August run kept aggregates only |
| **E7b** Confirmatory re-run with per-item records, `deepseek-v4.1-flash`, n=500 | **exact paired tests**: GSM8K 91.2%→95.4% (+4.2pp, 95% CI [1.8, 6.6], McNemar p=0.0011) and MATH 46.2%→51.6% (+5.4pp, CI [0.0, 11.0], p=0.0596, directional); balanced-vs-plain critique on MATH 51.6% vs 48.6% (+3.0pp, p=0.32). Item-level records in `results/e7_rerun_gsm8k.json` and `results/e7_rerun_math.json` |
| **E8** Compute-cost analysis | mean tokens: BBH 892 (d1) / 1061 (d3), GSM8K 541 / 993, MATH 886 / 2271; the sentinel saves 15–60% vs. the depth-3 worst case |

**Headline:** on clean BBH the loop does not move accuracy beyond the 95% Wilson interval, yet its `CONFIRMED` early stopping intercepts and terminates 82–88% of redundant generations at equal accuracy, bounding inference to ≈2.1 generations per question; on GSM8K and MATH the same loop improves accuracy (+4.2pp and +14.2pp in the August session; +4.2pp, p=0.0011, and +5.4pp, p=0.0596, under exact paired tests in the E7b confirmatory re-run) while still early-stopping 81–93% of items. The confidence threshold τ was found overconfident (no graded cost–accuracy trade-off), making the `CONFIRMED` sentinel the effective cost-bounding mechanism.

Full results: `results/bbh_reviewer_response.json`, `results/reflection_study.json`, `results/crossbench_study.json`, and the E7b confirmatory re-run with per-item records: `results/e7_rerun_gsm8k.json`, `results/e7_rerun_math.json` (recompute the paired statistics with `code/compute_paired.py`).

## Repository Structure

```
EvoResearcher/
├── code/                 # Experiment scripts (self-contained, OpenAI-compatible API only)
│   ├── run_bbh_experiments.py    # E1–E3 (controlled, ablation, loop) + robustness probe
│   ├── run_reflection_study.py   # E4 (depth sweep + shared cells)
│   ├── run_crossbench_study.py   # E6 (τ sweep) + E7 (GSM8K/MATH) + E8 (token cost)
│   ├── run_revision_study.py     # large-sample (n=500) reruns, checkpointed
│   └── parse_revision_results.py # parse large-sample results into paper tables
├── data/                 # Question sets (real, fetched from public sources)
│   ├── bbh_100.jsonl / bbh_500.jsonl
│   ├── gsm8k_100.jsonl / gsm8k_500.jsonl
│   ├── math_100.jsonl  / math_500.jsonl
│   └── fetch_gsm_math.py # dataset download script
├── results/              # Real experimental results (JSON, n=100 per benchmark)
├── figures/              # Figure-drawing script
└── requirements.txt
```

## Usage

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://your_openai_compatible_api/v1"
export MODEL_NAME="deepseek-v4-flash"

# E1–E3 (controlled comparison, ablation, reflective loop) + robustness probe
python code/run_bbh_experiments.py

# E4 reflection-depth sweep
python code/run_reflection_study.py

# E6 τ sweep + E7 GSM8K/MATH cross-benchmark + E8 token cost
python code/run_crossbench_study.py
```

All scripts save results incrementally (`results/*.json`) and support resuming with `RESUME=1`. Each cell runs on the same frozen backbone (temperature 0.3) so results are comparable within the 95% Wilson interval (n=100, ±8.7pp).

## Citation

```bibtex
@article{yu2026evoresearcher,
  title={Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models},
  author={Yu, Wei and Liu, Suxing and Yu, Minjie and Wang, Jiahao and Zheng, Zhijian and Deng, Haocheng and Li, Bing},
  journal={Electronics},
  year={2026}
}
```

## License

MIT License

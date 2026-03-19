# BTP Project Recommendation

## 📌 Recommended Project

### **Causal PNS-Guided Mixture-of-Experts Decoder for Efficient Chain-of-Thought Reasoning**

---

## 🔍 What Did I Read?

### The Paper (2506.09853v3) — NeurIPS 2025
**Title:** *Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning*

**Core Idea:** Current LLM reasoning (Chain-of-Thought / CoT) wastes tokens — it either includes redundant steps (*Sufficient but Unnecessary*) or misses key steps (*Necessary but Insufficient*). This paper introduces a **causal framework using Probability of Necessity and Sufficiency (PNS)** to:
1. **Prune** unnecessary reasoning steps via counterfactual rollouts
2. **Add** missing steps to make reasoning complete
3. Use the cleaned-up CoT for **fine-tuning** smaller models

**Key Results (measured):**
| Task | Token Reduction | Accuracy Change |
|------|---------------|----------------|
| GSM-8k | 113→27 tokens (76% ↓) | 95% → 97.9% (↑) |
| MATH-500 | 387→161 tokens (58% ↓) | 85.9% → 91.6% (↑) |
| CommonsenseQA | 368→167 tokens (54% ↓) | 87.1% → 94.9% (↑) |
| AIME (hard) | 3052→1639 tokens (46% ↓) | 79.2% → 92.6% (↑) |

> **Important:** The paper is published at **NeurIPS 2025** — the #1 ML conference. This gives your project an extremely credible, reproducible baseline. You are *extending* a NeurIPS paper, not copying it.

---

## 🎯 Why This Project? (The Justification)

### What the paper does NOT do
The paper applies PNS to one homogeneous decoder. It does not:
- Assign different types of reasoning to different specialized modules
- Modify the decoder architecture
- Combine PNS-pruning with Mixture-of-Experts routing

### What your BTP adds (the gap you fill)
Your professor mentioned two things: **MoE** and **architectural changes in the decoder**. This project directly addresses both.

> **Your original contribution:** Use PNS scores to *dynamically route* reasoning steps to specialized expert decoders — making MoE routing causally informed, not just load-balanced.

---

## 🏗️ Proposed Architecture

```
Input Question
     ↓
Base LLM generates initial CoT trace (e.g., DeepSeek-R1 or Qwen-2.5)
     ↓
┌─────────────────────────────────────────┐
│         PNS Evaluation Module           │
│  - Counterfactual rollouts per step     │
│  - PNS score assigned to each step      │
│  - Low-PNS steps tagged as redundant    │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│    Step-Type Classifier (gating net)    │
│  Classifies each step as:               │
│  - Mathematical / Arithmetic            │
│  - Logical / Deductive                  │
│  - Commonsense / Background knowledge   │
│  - Verification / Answer generation     │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│        MoE Decoder Experts              │
│  Expert 1 → Math symbolic reasoning     │
│  Expert 2 → Logical inference           │
│  Expert 3 → Commonsense knowledge       │
│  Expert 4 → Verification & answer gen   │
│  (Low PNS steps → dropped entirely)     │
└─────────────────────────────────────────┘
     ↓
Final Compact, Accurate Answer
```

**Key architectural change:** The standard FFN in the decoder is replaced by a **causal-gated MoE FFN**. Gating is not random — it uses PNS scores and step-type labels to route causally.

---

## ✅ Why This Is Better Than Each Other Idea

| Idea | Why it's weaker |
|------|---------|
| **1. Causal MoE for Reasoning** | Same direction, but vague — yours is concrete because PNS provides the routing signal mathematically |
| **2. Dynamic Expert Routing** | Routing on attention entropy is heuristic, not grounded in cause-and-effect |
| **3. Stage-Aware Decoder** | Stage classification is ad-hoc; PNS gives a rigorous criterion for "what is a necessary step" |
| **4. RL-Based MoE** | Needs RL training infrastructure, very hard to stabilize; high risk for BTP scope |
| **5. Graph-of-Experts** | Very high novelty but no existing paper to baseline against; risky |

> **Your project = Idea #1 with a NeurIPS 2025 paper as mathematical backbone.** This is the strongest possible position — you extend proven work with a novel architectural contribution.

---

## 📊 Proof You Can Show Professors

### Existing proof (from the paper — already verified at NeurIPS)
- PNS framework works: GSM-8k tokens reduced 76%, accuracy *improved* by ~3%
- Works on multiple LLMs: Qwen-2.5-72B, DeepSeek-R1, Llama-3.1-8B
- Code is open-source: [github.com/yxn9191/causalmath](https://github.com/yxn9191/causalmath)

### New proof you will generate
Show your MoE extension vs. the paper's baseline:

| Metric | Paper (PNS only) | Your model (PNS + MoE) | Expected direction |
|--------|-----------------|----------------------|------------------|
| Token count | ~27 tokens (GSM-8k) | ~20 tokens | ↓ further |
| Accuracy | 97.9% | ≥ 97.9% | Maintain or improve |
| FLOPs (active params) | Dense decode | Only 1-2 experts active | ↓ (sparsity) |
| Expert specialization | N/A | Routing entropy per expert | Measurable & showable |

### Benchmark datasets (well-known — professors will recognize these)
- **GSM-8k** — Grade school math, 8,500 problems
- **MATH-500** — Intermediate/competition math
- **CommonsenseQA** — Everyday reasoning, 12,100 questions
- *(Optional)* **AIME 2025** — Advanced math, shows generalization

---

## 🗂️ 4-Month Implementation Plan

### Month 1 — Reproduce the paper
- [ ] Clone [github.com/yxn9191/causalmath](https://github.com/yxn9191/causalmath)
- [ ] Run PNS evaluation on GSM-8k with Qwen-2.5-7B (small model, feasible on a single GPU)
- [ ] Verify token reduction numbers match the paper
- **Output:** Reproduced Table 1 of the paper (your baseline)

### Month 2 — Build the Step Classifier & MoE Expert Layer
- [ ] Fine-tune a small classifier (e.g., DistilBERT) to label reasoning step types
- [ ] Replace 1 FFN layer in the decoder with a 4-expert MoE layer (use `DeepSpeed-MoE` or `MegaBlocks`)
- [ ] Wire PNS score into the routing gate (high-PNS → specialized expert; low-PNS → drop)
- **Output:** Modified architecture running on TinyLlama / GPT-2 for sanity checks

### Month 3 — Experiments at Scale
- [ ] Fine-tune DeepSeek-R1-Distill-Qwen-1.5B with your causal MoE CoTs
- [ ] Run on GSM-8k, CommonsenseQA, MATH-500
- [ ] Compare: (a) Original LLM, (b) Paper's PNS SFT, (c) Your PNS + MoE SFT
- **Output:** A completed, comparison results table

### Month 4 — Analysis, Report & Presentation
- [ ] Ablation study: remove MoE (= paper's method) → demonstrates MoE adds value
- [ ] Expert load distribution analysis — show experts learned to specialize
- [ ] Write BTP report: Introduction, Related Work, Method, Experiments, Conclusion
- **Output:** Presentation-ready results + final written report

---

## 💬 What to Tell Your Professors

> *"My BTP extends a NeurIPS 2025 paper on causal reasoning in LLMs. The paper showed that using causal Probability of Necessity and Sufficiency, we can prune redundant reasoning steps and improve both token efficiency and accuracy. My original contribution is architectural: instead of a single homogeneous decoder, I propose routing causally necessary reasoning steps to specialized expert decoders — a MoE architecture where the gating is governed by causal PNS scores, not random load balancing. I measure this against the paper's baseline on GSM-8k, MATH-500, and CommonsenseQA, tracking token count, accuracy, and expert specialization entropy. The existing paper code gives me a provable, reproducible baseline. My hypothesis is that expert specialization will further reduce active compute while maintaining or improving accuracy."*

---

## ⚠️ Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| GPU memory for large models | Use small models (1.5B–7B); QLoRA for fine-tuning |
| MoE doesn't improve over paper | Run ablation — "same accuracy with fewer FLOPs" is still a valid and publishable result |
| PNS computation is slow | Use perplexity-based proxy for fast iteration; full PNS for final experiments |
| Negative results | Even if MoE hurts, you have the reproduced paper baseline as a standalone contribution |

---

## 🔖 Summary Card

| Audience | One sentence |
|----------|-------------|
| **Yourself** | Extend a NeurIPS 2025 paper by replacing the decoder's FFN with a causally-routed MoE, getting sparser and more specialized reasoning at lower token cost. |
| **Your professor** | Architectural modification to the decoder (MoE) + causal grounding (PNS) = novel contribution with measurable proof on standard benchmarks. |
| **For publication (stretch goal)** | *"Causally-Informed Mixture-of-Experts Decoding for Efficient Reasoning in LLMs"* — target ACL 2027 or EMNLP 2026. |

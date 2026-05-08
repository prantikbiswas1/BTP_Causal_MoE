# Research Gap Analysis
## Base Paper: *Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning*
### arXiv: 2506.09853v3 | Yu et al.

**BTP Project:** Causal Mixture-of-Experts: Semantic Step-Wise Expert Routing for Efficient and Interpretable LLM Reasoning  
**Author:** Prantik Biswas, IIT Patna, April 2026

---

## 1. Base Paper — Summary

| Aspect | Details |
|:---|:---|
| **Paper Title** | Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning |
| **Authors** | Xiangning Yu, Zhuohan Wang, Linyi Yang, Haoxuan Li, Anjie Liu, Xiao Xue, Jun Wang, Mengyue Yang |
| **Core Idea** | Apply causal theory (Probability of Necessity and Sufficiency — PNS) to automatically *prune* redundant and *complete* missing steps in Chain-of-Thought reasoning traces |
| **Method** | Monte-Carlo counterfactual rollouts to estimate PNS per step; threshold-based pruning (low PNS = redundant, remove); completeness check to insert missing logical steps |
| **Benchmarks** | GSM-8K, MATH-500, AIME 2025, CommonsenseQA |
| **Key Result** | Up to **78.4% token reduction** on GSM-8K; DeepSeek-R1 accuracy improves from 95.0% → 97.3% |
| **Model Used** | Post-hoc refinement applied to outputs of large frontier models (DeepSeek-R1, GPT-4o class) |

---

## 2. What the BTP Project Shares With the Base Paper

The BTP project was developed *independently*, but the intellectual overlap is significant and serves as strong validation:

| Shared Concept | Base Paper | BTP Project |
|:---|:---|:---|
| **PNS as a quality signal** | Estimates PN per step via counterfactual rollouts | Same — PN used to prune training traces (threshold 0.5) |
| **Causal framing of reasoning** | Treats steps as causal variables toward the final answer | Treats steps as typed causal interventions (`[MATH]`, `[LOGIC]`, etc.) |
| **Redundancy removal** | Prunes steps below PNS threshold | Removes low-PN steps from training data |
| **Token efficiency goal** | Reduces CoT verbosity | V2 achieves 63.7% token reduction on GSM-8K |
| **GSM-8K + CommonsenseQA eval** | Both benchmarks used | Both benchmarks used |

---

## 3. Research Gaps in the Base Paper

### Gap 1: Post-hoc Refinement Only — No Architectural Change
**What the base paper does:** PNS is used purely as a *post-processing* algorithm. After a model generates a CoT trace, the PNS engine prunes/completes it. The underlying model (DeepSeek-R1, GPT-4o) is completely **unchanged**.

**The gap:** This treats the causal framework as an external wrapper. The model itself never *learns* to reason causally. Every inference call still uses the full monolithic model generating a full verbose trace — PNS just trims the output afterwards.

> **BTP Fills This Gap:** Causal-MoE injects PNS-filtered traces *into training*, making the model internalize causal step discipline. The router learns to generate causally necessary steps natively, without external post-processing.

---

### Gap 2: No Domain-Specific Specialization — All Steps Treated Equally
**What the base paper does:** PNS scores individual steps as *necessary* or *redundant*, but makes no distinction between the *type* of reasoning involved. A MATH step and a COMMONSENSE step are evaluated by the same scalar PNS score.

**The gap:** Different reasoning types have fundamentally different causal structures. Mathematical deduction is highly deterministic; commonsense inference is probabilistic. A single PNS threshold cannot capture this heterogeneity.

> **BTP Fills This Gap:** Causal-MoE explicitly assigns domain tags `[MATH]`, `[LOGIC]`, `[COMMONSENSE]`, `[VERIFY]` to each step. A dedicated domain-specific expert (LoRA-based) handles each type — operationalizing *domain-aware causality*.

---

### Gap 3: No Parameter-Level Efficiency Gain — Still Dense Inference
**What the base paper does:** Even after PNS pruning reduces token count, the model used for inference is still the full frontier model (~70B+ params). Every generated token activates all parameters.

**The gap:** The efficiency benefit is *token-count reduction only*. There is **no reduction in per-token compute** (FLOPs per forward pass). For smaller or medium-scale models, this is a significant unaddressed inefficiency.

> **BTP Fills This Gap:** Through sparse Mixture-of-Experts injection, Causal-MoE achieves *both* types of efficiency:
> - **Token efficiency:** 63.7% fewer tokens generated (V2 on GSM-8K)
> - **Per-token FLOP efficiency:** 34.3% FLOP reduction vs base model (2.81T vs 4.28T)

---

### Gap 4: Requires a Powerful External Validator — Not Self-Contained
**What the base paper does:** PNS estimation relies on a "rollout model" that completes reasoning after each intervention. The quality of PNS estimates is stated as dependent on the **validator model's quality** — requiring a large, expensive rollout model.

**The gap:** The system cannot be made fully self-contained at small scale. Good pruning requires a good validator, which requires a large model — expensive and inaccessible.

> **BTP's Approach:** Uses the same Qwen2.5-7B model as both the generator and rollout model during PNS data construction — a closed-loop approach at 7B scale.

---

### Gap 5: Threshold α is Fixed and Non-Adaptive
**What the base paper acknowledges (limitation):** The paper explicitly notes sensitivity to the pruning threshold α. A fixed threshold works for average cases but may be suboptimal for:
- Very complex multi-step math (AIME level)
- Very simple single-step problems
- Mixed-domain problems requiring both MATH and COMMONSENSE

> **BTP's Extension Opportunity:** Domain-typing in Causal-MoE creates a natural basis for *domain-specific thresholds* — a MATH step may need PN ≥ 0.6 while a COMMONSENSE step is retained at PN ≥ 0.4, reflecting domain-specific counterfactual sensitivity.

---

### Gap 6: No Interpretability Mechanism — Black Box Pruning
**What the base paper does:** The output of PNS pruning is a shorter, more accurate trace — but there is no mechanism to *inspect why* a step was retained or removed in a human-understandable way. PNS is a scalar; it does not explain causal structure.

**The gap:** Despite using causal language (necessity, sufficiency), the framework produces no interpretable attribution a practitioner could use to debug reasoning.

> **BTP Fills This Gap Directly:** Every Causal-MoE trace step is labeled with its domain tag. Every error is attributed to a specific expert module. V1's explicit multi-call loop provides step-level attribution with near-perfect interpretability.

---

### Gap 7: Evaluated Only on Large Frontier Models — No Small-Model Validation
**What the base paper does:** Results are reported for DeepSeek-R1 and frontier-scale models with very high baseline accuracy (95%+ on GSM-8K). PNS is only applied as a post-processing layer on already excellent outputs.

**The gap:** It is unclear whether PNS-based pruning benefits smaller models (<10B parameters) where baseline accuracy is lower and reasoning traces are noisier.

> **BTP's Contribution:** All results are on **Qwen2.5-7B-Instruct** — demonstrating that the causal training signal is effective at 7B scale with a meaningful (non-trivial) baseline (88.86% GSM-8K).

---

### Gap 8: Sufficiency-Side (Step Insertion) Is Underexplored
**What the base paper does:** The dual contribution is pruning (necessity) AND completion (sufficiency). However, results are overwhelmingly focused on the pruning/necessity side. The individual impact of step-insertion is not cleanly ablated.

**The gap:** It is unknown how much accuracy gain comes from pruning redundant steps vs. inserting missing steps — the two effects are conflated.

> **BTP's Design Choice:** Causal-MoE focuses entirely on the *necessity* side, providing a clean ablation baseline: causal training benefit comes from *removing redundancy* alone, not from inserting steps.

---

## 4. Summary Table — Research Gap Map

| # | Research Gap in Base Paper | How BTP Addresses It |
|:--|:---|:---|
| 1 | PNS used post-hoc; model unchanged | PNS filters *training data*; model internalizes causal discipline |
| 2 | No domain-awareness; all steps scored as scalar | Domain-typed experts: MATH / LOGIC / COMMONSENSE / VERIFY |
| 3 | Token reduction only; no per-token FLOP saving | Sparse MoE: −34.3% FLOPs + −63.7% tokens |
| 4 | Requires large external validator for quality PNS | Closed-loop: same 7B model family for generation and scoring |
| 5 | Fixed threshold α; noted as a limitation in the paper | Domain-specific thresholds are a natural BTP extension |
| 6 | Black-box pruning with no interpretability mechanism | Explicit domain tags + expert attribution; step-level transparency |
| 7 | Only validated on large frontier models (R1, GPT-4o) | Demonstrated on 7B scale with non-trivial baselines |
| 8 | Sufficiency/insertion effect not cleanly ablated | BTP isolates necessity-only signal; provides a clean baseline |

---

## 5. Positioning Statement for the Thesis

The base paper (Yu et al., 2506.09853) establishes that **PNS is a principled and effective tool for measuring causal necessity of reasoning steps** — a critical theoretical contribution the BTP project borrows and extends.

However, the base paper treats PNS as a **post-hoc efficiency technique** applied to outputs of unchanged black-box models. The Causal-MoE BTP project goes significantly further:

1. **PNS is used to construct training data**, not just filter outputs — making causal discipline a *learned* property of the model.
2. **The model architecture itself is modified** to reflect causal structure through domain-typed expert modules.
3. **Efficiency is achieved at two levels** (tokens *and* FLOPs via sparse routing), not one.
4. **Interpretability is a first-class output**, not an afterthought.

The BTP project can therefore be positioned as the **architectural realization of the theoretical framework** proposed by the base paper — moving from *"score and filter"* to *"internalize and route."*

---

*Generated: April 21, 2026 | Causal MoE BTP — IIT Patna*

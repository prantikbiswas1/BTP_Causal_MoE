# BTP Project: Causal MoE Technical Blueprint & Progress

This document is the **single source of truth** for understanding the technical mechanics, goals, and current progress of the Causal Mixture of Experts project.

---

## 🛠️ Phase 1: Baseline CoT Generation (Script 01)

**Goal:** Create a diverse pool of reasoning traces for every question in the dataset.
**Logic:**
*   **Teacher Model:** `Qwen2.5-7B-Instruct` is used for high-accuracy reasoning.
*   **Sampling:** For every question (GSM8k, MATH, CSQA), we generate **5 independent traces** (`k=5`) at temperature 0.7.
*   **Objective:** To capture multiple valid (and some redundant) paths to a correct answer, which we will later optimize.

---

## 📉 Phase 2: Data Compression & PNS Scoring (Scripts 02-03)

**Goal:** Distill the "Causal Backbone" of a reasoning chain by identifying and removing redundant fluff.
**Logic (PNS Engine):**
1.  **Counterfactual Generation:** For every reasoning step $s_i$ in a trace, we replace it with a **mutated alternative** step $s'_i$ (using a prompt that forces the model to semantically deviate).
2.  **Rollout Verification:** We perform **3 independent rollouts** (`k=3`) starting from the alternative step $s'_i$ forward to the final answer.
3.  **PNS Calculation:** We calculate the **Probability of Necessity (PN)** using the formula:
    $$PN = 1 - E[y]$$
    *   $E[y]$ is the average correctness (Accuracy) of the 3 rollouts.
    *   **Interpretation:** If $PN \approx 1$, removing the step broke the final answer (the step is **Necessary**). If $PN \approx 0$, the model reached the right answer anyway (the step is **Redundant**).
4.  **Pruning:** Any step with a $PN < 0.3$ is removed, creating a **Compact CoT dataset**.

---

## 🏷️ Phase 3: Semantic Taxonomization (Scripts 04-06)

**Goal:** Train a tiny, fast "Expert Router" to understand the cognitive domain of each surviving step.
**Logic:**
*   **Step 04 (Sampling):** We sample **10,000 unique steps** from the compact training data.
*   **Step 05 (Auto-Labeling):** We ask the Qwen 7B model to categorize these 10k steps into four categories:
    1.  **MATH:** Numerical calculations and algebra.
    2.  **LOGIC:** Deductive logic and relational reasoning.
    3.  **COMMONSENSE:** Real-world facts and retrieval.
    4.  **VERIFY:** Double-checking the work or terminal synthesis.
*   **Step 06 (Distillation):** We fine-tune a lightweight **DeBERTa-v3** model (the "Router") on these 10k labels.
    - **Final Performance:** 76% overall accuracy.
    - **Logic Expertise:** 0.85 F1-score (Excellent).
    - **Math Expertise:** 0.69 F1-score (Good).
    - **Class Imbalance:** Identified low recall for `COMMONSENSE` (due to low support in sampled GSM8k/MATH data).

---

## 🚀 Phase 4: MoE Model Layer [COMPLETED ✅]

**Goal:** Integrate the distilled Router directly into the transformer architecture of the student model.
**Logic:**
1.  **Student Model:** `DeepSeek-R1-Distill-Qwen-1.5B`.
2.  **The Router:** The tiny DeBERTa model is embedded inside the model's forward pass.
3.  **Dynamic Routing:** When the model generates a step, the Router tags its type:
    *   If **MATH**, it routes the token only to the **MATH Expert** MLP.
    *   If **LOGIC**, it routes to the
4.  **PNS Dropping**: If a step is tagged as low-PNS (redundant), it **bypasses the experts entirely** (0.0 routing probability), saving massive FLOPs/computation.
    *   **Implementation**: Completed in `moe_experts.py`, `causal_router.py`, and `modified_decoder.py`.

---

## 📈 Detailed Project Log (Completed Items)

### Current Project Stage: Phase 4 (Implemented Causal MoE Architecture)

#### 1. Planning & Architecture (Completed)
- **Blueprint Solidified:** (`BTP_Comprehensive_Implementation_Plan.md`).
- **Theory Base:** Mapped PNS causal theory to MoE structural sparsity.

#### 2. Data & Infrastructure (Completed)
- **PNS Engine:** Fully optimized batched PNS engine (`run_pns_engine_batched.py`) utilizing `vLLM`.
- **Dataset Completion:** Generated Compact CoT Datasets for GSM-8k, MATH-500, and CSQA.
- **Semantic Taxonomy:** Trained DeBERTa-v3 router on 10k context-aware labels. Achieved **76.1% agreement** with the Teacher model in ~5 minutes of training (Microsoft DeBERTa-v3-small).

#### 3. Immediate Next Steps (Phase 4)
1.  Implement **Expert FFNs** in `src/model/moe_experts.py`.
2.  Implement **Causal Routing Logic** in `src/model/causal_router.py`.
3.  Override **Decoder Layers** in `src/model/modified_decoder.py`.

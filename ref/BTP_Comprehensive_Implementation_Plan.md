# BTP Project: Causal Mixture of Experts for Efficient Reasoning
## Comprehensive Deep-Dive Implementation Plan

### 🎯 Objective
This document serves as the **definitive, end-to-end blueprint** for AI agents (LLMs) to fully implement the BTP project: adding a **Causal Mixture-of-Experts (MoE)** routing layer to track and filter Chain-of-Thought (CoT) reasoning steps using **Probability of Necessity and Sufficiency (PNS)**. The project proves that explicitly parsing and routing reasoning step types (Math, Logic, Commonsense) into specialized experts while dropping redundant low-PNS steps results in higher accuracy with significantly fewer compute FLOPs and tokens.

This plan assumes the executor has access to a Linux environment with GPUs and provides exact module structures, logic flows, and benchmarking requirements.

---

## 📂 1. Repository Architecture & Layout

Create the following directory structure at `/home/nlp/Desktop/BTP/BTP_Causal_MoE/`. Agents should strictly follow these file paths.

```text
BTP_Causal_MoE/
├── requirements.txt
├── configs/
│   ├── deepspeed_stage3.json      # ZeRO-3 optimization config
│   └── training_args.yaml         # Hyperparameters for fine-tuning
├── data/
│   ├── raw/                       # HF dataset downloads
│   ├── traces/                    # Unpruned generated CoTs
│   ├── pns_scored/                # Full CoTs with PNS metadata appended
│   └── final_compact/             # Filtered & Tagged CoTs ready for MoE training
├── src/
│   ├── data_prep/
│   │   ├── download_benchmarks.py # Fetch GSM8k, MATH-500, CommonsenseQA
│   │   ├── generator.py           # vLLM script to generate N traces per query
│   ├── pns_engine/
│   │   ├── sufficiency.py         # Evaluates answer correctness
│   │   ├── counterfactuals.py     # Masks steps and generates alternative rollouts
│   │   ├── pns_calculator.py      # Computes PNS score = P(Sufficiency) * P(Necessity)
│   │   └── pruner.py              # Strips steps where PNS < threshold
│   ├── step_classifier/
│   │   ├── teacher_annotator.py   # Prompts large model to label 10k steps
│   │   └── train_deberta.py       # Fine-tunes DeBERTa-v3 on {MATH, LOGIC, COMMONSENSE, VERIFY}
│   ├── model/
│   │   ├── causal_router.py       # Computes gating decisions based on step-type tags
│   │   ├── moe_experts.py         # nn.Module containing the 4 FFN experts
│   │   └── modified_decoder.py    # Overrides standard LLM Decoder layer with MoE
│   └── training/
│       ├── trainer.py             # HuggingFace standard Trainer modified for MoE Aux Loss
│       └── metrics.py             # Computes exact match accuracy and token metrics
└── scripts/
    ├── 1_generate_traces.sh
    ├── 2_compute_pns.sh
    ├── 3_train_classifier.sh
    ├── 4_train_moe.sh
    └── 5_evaluate_all.sh
```

---

## 🛠️ 2. Step-by-Step Implementation Guide

### Phase 1: Environment & Baseline Data Generation (Week 1)
**Goal:** Setup environment and generate raw, unoptimized Chain-of-Thought reasoning traces that we will later prune.

1. **`requirements.txt`**: Define dependencies (`torch`, `transformers`, `vllm`, `deepspeed`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`).
2. **`src/data_prep/download_benchmarks.py`**:
    - Download `openai/gsm8k` (main filter test).
    - Download `Lighteval/MATH-500` (math generalization test).
    - Download `commonsense_qa` (commonsense test).
3. **`src/data_prep/generator.py`**:
    - **Model to use:** `Qwen/Qwen2.5-7B-Instruct` (load via `vLLM` for fast generation).
    - **Action:** For every question in the training sets, prompt the model to "Think step-by-step and provide the final answer".
    - **Output:** Generate `k=5` different reasoning traces per question (temperature 0.7). Save to `data/traces/`.

---

### Phase 2: The PNS Engine (Week 2)
**Goal:** Implement the mathematical logic from paper `2506.09853v3.pdf` to score the usefulness of every reasoning step.

1. **Step Tokenization**: Parse the string traces into list of steps (split by `\n` or `Step X:` markers).
2. **`src/pns_engine/sufficiency.py`**:
    - Does reasoning trace $T$ lead to the correct ground truth answer? (Yes/No). This is the base PS (Probability of Sufficiency).
3. **`src/pns_engine/counterfactuals.py` & `pns_calculator.py`**:
    - **Logic:** For a specific step $S_i$ in trace $T$, generate an *alternative* step $S'_i$ (using a prompt that forces the LLM to output a step with different meaning from $S_i$).
    - Run $k=3$ forward pass rollouts from $S_{i-1} \to S'_i \to \text{Answer}$.
    - Calculate the average correctness $Y$ of these rollout answers.
    - **Probability of Necessity (PN):** `PN = 1 - average_y`.
    - Note: The paper fundamentally calculates `PN` (Probability of Necessity) and uses it to decide whether to intervene (replace/prune) a step. Append `{ "pn_score": 0.XX }` metadata to every step in the JSONL dataset.
5. **`src/pns_engine/pruner.py`**:
    - Discard or replace any reasoning step where `pn_score < threshold` (e.g. 0.3). The casualmath implementation actively updates the chain if PN is below threshold. For our pruned dataset generation, strip these low-PN steps to produce the **Compact CoT dataset**.

---

### Phase 3: The Step-Type Classifier (Week 3)
**Goal:** We need to know *what* a step is to route it to the correct MoE expert.

1. **Taxonomy Definition**:
    - `0: MATH` (Equations, arithmetic, algebra)
    - `1: LOGIC` (Deductions, "therefore", "if-then")
    - `2: COMMONSENSE` (World knowledge, facts)
    - `3: VERIFY` (Final answer generation, checking work)
2. **`src/step_classifier/teacher_annotator.py`**:
    - Sample 10,000 diverse steps from the unpruned dataset.
    - Use an API (like DeepSeek or GPT-4o) with a strict prompt to label them into the 4 buckets.
3. **`src/step_classifier/train_deberta.py`**:
    - Fine-tune `microsoft/deberta-v3-base` (or similar lightweight encoder) on the 10k labeled steps. This model must be fast enough to run continuously during routing.
4. **Data Tagging**: Run the trained DeBERTa model over the entire **Compact CoT dataset** from Phase 2. Now every step has `text`, `pns_score`, and `type_tag`.

---

### Phase 4: Causal MoE Architecture Overlay (Week 4)
**Goal:** Modify a standard LLM to use our new PNS-guided experts.

1. **Base Model**: Use `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. It's a highly capable reasoning model small enough to train easily.
2. **`src/model/moe_experts.py`**:
    - Inherit the existing MLP/FFN layer from Qwen's architecture.
    - Instantiate 4 independent copies of this MLP (`expert_math`, `expert_logic`, `expert_commonsense`, `expert_verify`).
3. **`src/model/causal_router.py`**:
    - Note: In a real training/inference loop, we append control tokens like `[MATH]`, `[LOGIC]` before the step text.
    - The Router reads the token embedding.
    - **Crucial Causal Logic:**
        - If the token sequence is tagged `[MATH]`, routing probability is hard-assigned 1.0 to Expert 1, 0.0 to others.
        - *Architecture Extension (Optional gradient routing):* Pass the embedding through a linear layer, but add a massive bias penalty to non-matching experts based on the token tag.
    - **PNS Dropping:** If a step is dynamically injected but has no tag (or is tagged as a dropped low-PNS step during ablation), the router outputs `0.0` for all experts. The tokens bypass the FFN entirely (saving FLOPs).
4. **`src/model/modified_decoder.py`**:
    - Overwrite `Qwen2ForCausalLM` blocks. Replace `layer.mlp` in the last $N/2$ blocks (e.g., layers 12 to 24) with your `CausalMoELayer`.

---

### Phase 5: Training (Fine-Tuning) (Week 5)
**Goal:** Train the experts to specialize using the curated Compact Tagged CoT dataset.

1. **`configs/deepspeed_stage3.json`**: Enable CPU offloading and ZeRO-3 to ensure the 1.5B model + 4 duplicated FFN layers fit in VRAM.
2. **`src/training/trainer.py`**:
    - Standard Next-Token prediction causal language modeling loss.
    - *Expert Load Balancing:* You must add an auxiliary loss to prevent mode collapse (e.g. all tokens secretly going to Expert 1). Ensure the router adheres to the step tags.
    - Run `scripts/4_train_moe.sh` for 3 epochs on the training data.

---

### Phase 6: System Evaluation & Benchmarking (Week 6)
**Goal:** Generate the absolute proof needed for the University BTP committee.

1. **`src/training/metrics.py`**:
    - Implement a `FlopCounter` hook during inference to explicitly measure saved calculations when routing bypasses experts.
    - Token counting mechanism per generated answer.
2. **`scripts/5_evaluate_all.sh`**:
    - Evaluates the custom DeepSeek-CausalMoE-1.5B against the Base DeepSeek-R1-Distill-Qwen-1.5B model on `GSM8k`, `MATH-500`, and `CommonsenseQA`.
3. **Logs Extraction**: The script MUST dump JSON logs containing `original_accuracy`, `causal_moe_accuracy`, `original_avg_tokens`, `causal_moe_avg_tokens`, and `expert_activation_counts`.

---

## 📊 3. Final Required Deliverables (The "Proof" Tables)

The LLMs executing this plan must produce the data for these EXACT tables for the final paper.

### Table 1: Baseline Reproduction (Proving the paper's claims hold)
| Benchmark | Original Model Tokens | Original Accuracy | Paper PNS-Pruned Tokens | Paper PNS Accuracy | Our PNS Reproduction Tokens | Our PNS Accuracy |
|-----------|-----------------------|-------------------|-------------------------|--------------------|-----------------------------|------------------|
| GSM-8k    | ~113                  | ~90%              | 27 (76% reduction)      | 97.9%              | *(Run eval script)*         | *(Run eval script)*|
| MATH-500  | ~387                  | ~86%              | 161                     | 91.6%              | *(Run eval script)*         | *(Run eval script)*|

### Table 2: Main BTP Contribution Results (PNS + MoE superiority)
| Benchmark   | Architecture | Avg Reasoning Tokens | Avg Inference FLOPs | Expert Specialization Logged? | Final Accuracy |
|-------------|--------------|----------------------|---------------------|-------------------------------|----------------|
| **GSM-8k**  | Original LLM Base | ~113 | Baseline | No | ~90% |
| **GSM-8k**  | Paper's PNS SFT Baseline | ~27 | Medium | No | ~97.9% |
| **GSM-8k**  | **Our PNS + Causal MoE** | **Target: <25**| **Lowest (Sparse FFN)** | **Yes** | **Target: ≥ 98%** |
| **MATH-500**| Paper's PNS SFT Baseline | ~161 | Medium | No | ~91.6% |
| **MATH-500**| **Our PNS + Causal MoE**| **Target: <161**| **Lowest (Sparse FFN)** | **Yes** | **Target: ≥ 92%** |
| **CommonSQA**| Paper's PNS SFT Baseline | ~167 | Medium | No | ~94.9% |
| **CommonSQA**| **Our PNS + Causal MoE**| **Target: <167** | **Lowest (Sparse FFN)** | **Yes** | **Target: ≥ 95%** |

### Table 3: Ablation Study (Justifying the architectural change)
| Component Modified | GSM-8k Accuracy | Token Efficiency | Computational Cost (FLOPs) | Conclusion |
|--------------------|-----------------|------------------|----------------------------|------------|
| **Full PNS + MoE System**| **Best** | **Maximum** | **Lowest** | **Optimal** |
| Run w/o MoE (Dense FFN) | Comparable | High | Medium-High | MoE is strictly required to lower compute overhead |
| Run w/o PNS Pruning | Lower | Poor (Redundant) | Very High | PNS is strictly required to prevent "overthinking" |

---
*End of Blueprint. Agents: Proceed sequentially through Phase 1 to Phase 6.*

# Causal Mixture of Experts for Efficient Reasoning: Guiding Sparse Architecture with Probability of Necessity and Sufficiency

**Abstract**—Chain-of-Thought (CoT) prompting enables Large Language Models (LLMs) to perform complex reasoning but often suffers from computational inefficiencies, commonly known as "overthinking." This manifests as redundant or unnecessary intermediate steps that inflate token counts and computational cost. Recent advancements introduced the Probability of Necessity and Sufficiency (PNS) framework to evaluate and prune these redundant reasoning steps through causal intervention. However, existing methodologies rely on dense, general-purpose decoders that process all reasoning domains (e.g., arithmetic, logic, commonsense) monolithically. In this paper, we propose a novel Causal Mixture of Experts (Causal-MoE) architecture that integrates PNS evaluations directly into the routing mechanisms of sparse transformer layers. During inference, a discrete router flags high-PNS steps and directs them to domain-specific experts, while low-PNS (redundant) steps are bypassed entirely to save Floating Point Operations (FLOPs). We detail our methodology, the scalable batch-generation pipeline using vLLM on the Qwen2.5-7B baseline, and our experimental strategy aiming to demonstrate major reductions in token length and latency without sacrificing reasoning accuracy.

---

## I. INTRODUCTION

Large Language Models (LLMs) have demonstrated exceptional capabilities in complex problem-solving domains, primarily facilitated by Chain-of-Thought (CoT) prompting [1]. By generating intermediate reasoning steps before arriving at a final answer, LLMs mimic human step-by-step logic, significantly boosting performance on arithmetic, deductive, and commonsense tasks. Despite these advancements, standard CoT generation is highly susceptible to generating valid but functionally redundant paths—often referred to as "overthinking."

This overthinking wastes compute. Every generated token requires an expensive forward pass through the entire network. To address this, Yu et al. [2] introduced a causal framework to formalize CoT optimization via the Probability of Necessity and Sufficiency (PNS). By executing counterfactual rollouts—substituting steps and measuring the causal effect on the final answer's correctness—the PNS framework effectively prunes steps that are not simultaneously necessary and sufficient. 

However, a fundamental limitation remains in current optimized CoT applications: they rely on dense transformer decoders. In a dense decoder, mathematical deductions, logical assertions, and factual retrievals are all processed by the exact same multi-layer perceptron (MLP) blocks. This prevents structural specialization.

To bridge this gap, this project proposes **Causal Mixture of Experts (Causal-MoE)**. We introduce an architectural modification that replaces the standard Feed-Forward Network (FFN) layers with specialized experts. Crucially, our system does not rely on standard token-balancing routers. Instead, we propose a causally informed routing mechanism where steps are preemptively scored by their PNS value and categorized by type. Essential steps are processed by their respective specialist experts, while low-PNS steps are dynamically dropped, completely bypassing expert activation. 

## II. RELATED WORK

### A. Chain-of-Thought Reasoning
CoT reasoning [1] has been fundamental in extending the capabilities of LLMs beyond rapid pattern matching. Various extensions, such as Tree-of-Thought and Graph-of-Thought, have expanded reasoning capacity but invariably increased the computational footprint via extensive token generation.

### B. Causal Necessity and Sufficiency
Causal inference principles, originating from Judea Pearl's structural causal models, define the Probability of Necessity and Sufficiency (PNS) to isolate true causal drivers [3]. The "Causalmath" paper [2] successfully applied this to LLMs, computing PNS scores via counterfactual interventions (rollouts) to identify whether an intermediate step was both sufficient (leads to the right answer) and necessary (removing it breaks the answer). This allowed for the construction of "compact CoTs."

### C. Mixture of Experts (MoE)
Mixture of Experts is a prominent technique for scaling model capacity without proportionally increasing inference FLOPs [4]. In a standard MoE transformer, the dense FFN is replaced by an array of $N$ smaller neural networks (experts), governed by a gating layer. Current state-of-the-art models employ MoE primarily for load balancing. Our work uniquely re-purposes the MoE router to act as a causal intervention gate based on PNS scores and reasoning taxonomies.

## III. PROPOSED METHODOLOGY

Our framework consists of three sequential paradigms: Causal Evaluation, Semantic Categorization, and Sparse Routing.

### A. Causal Evaluation via PNS Engine
To generate causally optimized datasets, we employ a teacher model (`Qwen2.5-7B-Instruct`) to generate a baseline set of unpruned CoT traces. We then subject every step $s_i$ in reasoning trace $T$ to a modified PNS engine:
1. **Sufficiency Check:** The model verifies if $T$ correctly derives the ground truth $y$.
2. **Counterfactual Generation:** For step $s_i$, we force the LLM to generate an alternative statement $s'_i$ that intentionally deviates semantically from $s_i$.
3. **Rollout Verification:** The engine completes the logic chain from $s'_i$ forward over $k=3$ iterations. 
4. **PNS Scoring:** The Probability of Necessity is calculated as `PN = 1 - E[y]`, where $E[y]$ is the expectation of the rollout reaching the correct answer. Steps below a specific threshold $\alpha$ are earmarked for deletion.

### B. Semantic Taxonomization 
To prepare the CoT traces for the MoE layer, we structurally categorize the surviving high-PNS steps. We isolate four primary cognitive domains:
- `MATH`: Arithmetic and algebraic calculations.
- `LOGIC`: Deductive and inductive reasoning flows.
- `COMMONSENSE`: Real-world knowledge retrieval.
- `VERIFY`: Terminal verification and answer synthesis.
A lightweight encoder (DeBERTa-v3) is fine-tuned to continuously annotate steps with these domain tags.

### C. Causal-MoE Architecture Overlay
Our primary architectural contribution modifies the standard transformer decoder layer inside a distillation-scale model (e.g., DeepSeek-R1-Distill 1.5B).
1. We instantiate four independent FFN modules aligned with our taxonomy.
2. The router mechanism is intervened upon: rather than projecting token embeddings through a standard Softmax gate, the router interprets the previously attached step-tags and strictly biases routing probability toward the respective expert.
3. If an input vector corresponds to a sequence tagged as low-PNS (redundant), the router explicitly returns `0.0` for all experts. The token embedding inherits only residual connections, allowing the network to completely skip dense computation for that sequence.

## IV. EXPERIMENTAL SETUP & BTP PROGRESS RECORD

### A. Environment and Hardware
To ensure local reproducibility and to avoid high API costs, the entire pipeline is being built natively on Linux utilizing high-end GPUs. Earlier experiments targeted 2x NVIDIA RTX A5000s, with recent efforts scaling to utilizing full A100 80GB PCIe GPUs to maximize tensor-parallel throughput.

### B. Scaled Baselines Integration (Current Progress)
A major component of our BTP progress is the implementation of an accelerated, highly optimized batched PNS engine (`run_pns_engine_batched.py`). Relying on the `vLLM` library for inference acceleration, we instantiated a pipeline that:
- Loads the `Qwen2.5-7B-Instruct` model distributed via Tensor Parallelism (`tensor_parallel_size=2`).
- Orchestrates batched counterfactual prompts and $k$-rollouts utilizing high GPU memory utilization (0.95) and large context windows (8192 tokens).
- Introduces robust checkpointing, enabling distributed, fault-tolerant dataset generation over millions of tokens.

By establishing `Qwen2.5-7B-Instruct` as our "Teacher," we have hit a sweet spot: the 7B perimeter ensures mathematical robust traces lacking in smaller networks while easily fitting under memory constraints, a distinct operational advantage over original research restricted to vast, costly commercial APIs like GPT-4.

### C. Target Datasets
We evaluate on a stringent set of benchmarks representing diverse reasoning challenges:
- **GSM-8k:** Standardized grade-school arithmetic [5].
- **MATH-500:** High-school and competition-level mathematics [6].
- **CommonsenseQA:** Contextual, real-world knowledge deduction [7].

## V. EXPECTED RESULTS AND EVALUATION STRATEGY

We expect the transition from a dense decoder processing unpruned data to a Causal-MoE processing PNS-pruned data to yield empirical improvements across three axes:
1. **Token Efficiency:** A strict reduction in reasoning length vs. conventional baselines. We target matching the $76\%$ token reduction observed in the original Causalmath paper on GSM-8k.
2. **Computational Cost (FLOPs):** By routing low-PNS tokens around the experts, our architecture is expected to exhibit dynamic computational sparsity, theoretically demanding fewer FLOPs than a dense equivalent even when total token length is held equal.
3. **Accuracy Integrity:** Accuracy should remain functionally equivalent or modestly improved ($\sim 98\%$ on GSM-8k) as logical contradictions and hallucinated dead-ends are pruned out of the thought chain.

## VI. CONCLUSION

This project bridges a critical gap in LLM architecture by coupling causal reasoning evaluations with runtime structural routing. By using PNS scoring to direct Mixture of Experts pathways, we aim to demonstrate that large models can perform highly intricate reasoning tasks while expending significantly fewer computations on redundant or erroneous thoughts. The ongoing development of our highly parallel batching infrastructure lays the foundation for full-scale dataset generation and eventual MoE fine-tuning.

---

## REFERENCES

[1] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, and D. Zhou, "Chain-of-thought prompting elicits reasoning in large language models," *Advances in Neural Information Processing Systems*, vol. 35, pp. 24824–24837, 2022.

[2] X. Yu, Z. Wang, L. Yang, H. Li, A. Liu, X. Xue, J. Wang, and M. Yang, "Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning," *arXiv preprint arXiv:2506.09853v3*, 2025. Accepted to NeurIPS 2025.

[3] J. Pearl, *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2000.

[4] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.

[5] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al., "Training verifiers to solve math word problems," *arXiv preprint arXiv:2110.14168*, 2021.

[6] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, "Measuring mathematical problem solving with the MATH dataset," *NeurIPS*, 2021.

[7] A. Talmor, J. Herzig, N. Lourie, and J. Berant, "CommonsenseQA: A question answering challenge targeting commonsense knowledge," *NAACL-HLT*, 2019.

---

## APPENDIX: COMPREHENSIVE BTP PROGRESS RECORD

**Current Project Stage:** Phase 2 (PNS Engine Implementation & Execution)

### 1. Planning & Architecture
- **Plan Established:** The end-to-end blueprint has been solidified (`BTP_Comprehensive_Implementation_Plan.md`) charting 6 phases from data generation to final MoE fine-tuning.
- **Model Selection:** `Qwen2.5-7B-Instruct` was chosen as the teacher model due to its high math accuracy and suitability for 24GB/80GB VRAM distribution. The target architecture for fine-tuning has been identified as DeepSeek-R1-Distill-Qwen-1.5B.
- **Theory Base:** Successfully understood and formally recorded the implications of the "Causalmath" paper (`2506.09853v3.pdf`), bridging its software-level reduction with our proposed hardware-level (MoE) reduction (`simple_explanation.md`).

### 2. Implementation Progress
- **Framework Groundwork:** Initial repository layout scoped under `BTP_Causal_MoE/`
- **PNS Engine:** Wrote and refined the causal mathematical scoring engine. 
- **GPU Scaling:** Refactored the core logic (`run_pns_engine.py` $\implies$ `run_pns_engine_batched.py`) to utilize batch processing and `vLLM`. Overcame sequential GPU under-utilization, allowing the engine to leverage 95% VRAM and multiple GPUs by grouping counterfactual logic statements into concurrent lists before inferencing.
- **Resilience:** Implemented file-based checkpointing via `id` lookup to allow continuous cloud execution without data loss upon preemptions.
- **Dataset Processing (Phase 2):** Successfully and completely processed the GSM-8k training dataset. The batched PNS engine effectively executed batched counterfactual rollouts and attached PNS scores to all 7,477 generated reasoning traces, successfully concluding the GSM-8k generation step of Phase 2.

### 3. Immediate Next Steps
- Execute `run_pns_engine_batched.py` over the remaining dataset trace files (MATH-500 and CommonsenseQA) to conclude PNS metadata generation.
- Run the extraction/pruning step via `pruner.py` over the scored `.jsonl` files to formalize the "Compact CoT Dataset".
- Sample 10,000 steps and utilize an API to label steps into `MATH`, `LOGIC`, `COMMONSENSE`, `VERIFY` to train the DeBERTa-based classifier (Phase 3).

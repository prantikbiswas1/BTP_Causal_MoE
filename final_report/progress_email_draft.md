# BTP Progress Report Email

**To:** Prof. Rajiv Mishra; Dr. Manoranjan (PhD Supervisor)  
**Subject:** BTP Progress Report — Causal Mixture-of-Experts (April 2026)

---

Respected Professor Mishra and Dr. Manoranjan,

I hope this message finds you both in good health. I am writing to provide a progress update on my B.Tech Project: **"Causal Mixture-of-Experts: Semantic Step-Wise Expert Routing for Efficient and Interpretable LLM Reasoning."**

I am pleased to report that the core research, implementation, and experimental evaluation are substantially complete. The paper draft is currently in progress, and while I have not yet finalized it for submission, significant work has been accomplished across all major components of the project.

---

### What Has Been Completed

**1. Architecture Design and Implementation**

I have designed and implemented the **Causal-MoE** architecture, which surgically injects Sparse Gated MoE MLP layers at four strategically selected depths (layers 6, 12, 18, 24) of the **Qwen2.5-7B-Instruct** backbone. The standard Feed-Forward Network layers at these positions are replaced by a **CausalMoEMLP** module containing four specialized LoRA-based domain experts: `[MATH]`, `[LOGIC]`, `[COMMONSENSE]`, and `[VERIFY]`. The attention mechanism is left entirely intact. Expert weights are initialized using an **Interleaved Weight Slicing** strategy derived from the pre-trained FFN weights, avoiding cold-start instability.

**2. Training Pipeline**

The model was trained on curated **Atomic Reasoning Traces** — a structured completion format where every reasoning sub-step is prefixed with a domain tag. The training used Paged AdamW 8-bit with LoRA (rank=32, α=64) applied to all attention projections, and 4-bit NF4 quantization for the base weights, enabling training on the HPC cluster's A100 GPUs.

**3. PNS-Based Trace Pruning**

I developed a **Probabilistic Necessity Score (PNS)** engine (`src/pns_engine/`) to score each reasoning step in a training trace for causal necessity. Steps with PNS ≥ 0.5 are retained for training. This significantly improved training data quality by removing redundant or easily substitutable steps from the traces.

**4. Two Inference Paradigms Evaluated**

I evaluated two distinct inference strategies:
- **V1 (Exploded):** An external Qwen2.5-1.5B model acts as a Causal Router, selecting the next expert tag at each step before the 7B model generates an atomic step.
- **V2 (Collapsed):** A single forward pass generates the full Atomic Reasoning Trace with internalized routing.

**5. Benchmark Evaluation**

Full evaluations were conducted on two standard benchmarks:

| Benchmark | Base Model | Causal-MoE V1 | Causal-MoE V2 |
|-----------|------------|---------------|---------------|
| CommonsenseQA (1,221 examples) | 82.06% | **83.05%** (+0.99 pp) | 78.38% |
| GSM8K (1,319 examples) | 88.86% | 68.08% | **76.80%** (+8.72 pp vs V1) |

Key efficiency results for V2 on GSM8K:
- **−63.7% tokens generated** per query (75.8 vs 208.9 for base)
- **−34.3% estimated inference FLOPs** (2.81T vs 4.28T)
- **+31.7% Reasoning Density** (accuracy per TFLOP) compared to the unmodified base model

**6. Figures and Charts**

Architecture diagrams (V1 and V2 system layouts) and all benchmark charts (accuracy comparison, token efficiency, FLOP vs accuracy trade-off, Pareto frontier) have been generated and incorporated into the draft.

---

### Current Status of the Paper

The research paper draft (`final_report/research_paper.md`) contains complete sections on:
- Abstract, Introduction, Related Work
- Architecture (Sections 3.1–3.4)
- Training Methodology (Section 4)
- Inference Paradigms (Section 5)
- Experiments & Results with all tables and figures (Section 6)
- Qualitative Analysis and Error Profiles (Section 7)
- Discussion, Limitations, and Future Work (Sections 8–10)
- Conclusion and References (Sections 11–12)

The draft is largely complete in structure and content. I am currently reviewing and refining the writing for clarity, academic tone, and consistency before finalizing it for submission. I expect to have a finalized version ready for your review shortly.

---

### Next Steps

- Finalize and polish the research paper draft
- Incorporate feedback from your review
- Prepare the final BTP submission package

I would greatly appreciate your guidance on any areas you feel need strengthening, and I am happy to share the current draft for your early review if that would be helpful.

Thank you for your continued support and mentorship throughout this project.

Warm regards,  
**Prantik Biswas**  
B.Tech (Computer Science and Engineering)  
Indian Institute of Technology Patna  
April 2026

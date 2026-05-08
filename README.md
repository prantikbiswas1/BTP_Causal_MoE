# BTP — Causal Mixture-of-Experts (Causal MoE)

> **B.Tech Project (BTP) | IIT BHU**  
> *Neural Architecture Surgery for Efficient Reasoning via Causal Expert Routing*

---

## Overview

This repository contains the full research implementation of **Causal MoE**, a novel Mixture-of-Experts architecture built on top of a pre-trained language model (Qwen-2.5-7B). The core contribution is a **surgical MoE injection** technique guided by **Probabilistic Necessity Scores (PNS)** — a causal metric that identifies which reasoning steps in a chain-of-thought are *necessary* for a correct answer.

Instead of training a monolithic model, we:
1. Identify causally necessary reasoning steps via counterfactual rollouts (PNS).
2. Use these scores to prune, compact, and tag training data.
3. Train four lightweight expert adapters (Math, Logic, Commonsense, Verify).
4. Inject a learned MoE router into the frozen base model at inference time.

The result is a model that achieves **superior reasoning density** (accuracy per parameter) compared to the base model while being significantly more compute-efficient.

---

## Repository Structure

```
BTP_Causal_MoE/
├── src/                        # All Python source modules
│   ├── data_prep/              # Raw data download & trace generation
│   ├── pns_engine/             # PNS scoring (counterfactual rollouts)
│   ├── atomic/                 # Atomic step extraction & compaction
│   ├── tag/                    # Taxonomy classification & tagging
│   ├── experts/                # Per-domain expert adapter training
│   │   ├── math/
│   │   ├── logic/
│   │   ├── commonsense/ (commomsense/)
│   │   └── verify/
│   ├── router/                 # MoE router training
│   ├── combine/                # Dataset combination pipeline
│   ├── evaluate_base/          # Base model evaluation (GSM8K, CSQA)
│   ├── evaluate_moe/           # MoE model evaluation
│   ├── causal_moe_v2/          # V2 Architecture: model definition, train, eval
│   │   ├── architecture.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── verify_build.py
│   │   └── verify_experts.py
│   └── helper/                 # Utility scripts (download, sanitize)
│
├── scripts/                    # Shell scripts for end-to-end pipeline execution
│   ├── 01_generate_all_traces.sh
│   ├── 02_compute_all_pns.sh
│   ├── 03_prune_compact_data.sh
│   ├── 04_sample_taxonomy_steps.sh
│   ├── 05_auto_label_taxonomy.sh
│   ├── 06_train_taxonomy_classifier.sh
│   ├── 07_tag_all_data.sh
│   ├── 08_atomic_compactor.sh
│   └── allocate_node.sh
│
├── configs/
│   └── ds_config.json          # DeepSpeed ZeRO-2 training configuration
│
├── data/                       # Data pipeline directories (data excluded from git)
│   ├── raw/                    # Raw benchmark downloads (GSM8K, CSQA, MATH-500)
│   ├── processed/              # Multi-step chain-of-thought traces
│   ├── pns_scored/             # PNS-annotated traces
│   ├── atomic/                 # Atomically compacted steps
│   ├── tagged/                 # Taxonomy-tagged steps
│   ├── experts/                # Per-expert training splits
│   ├── combined/               # Combined MoE training set
│   ├── final_compact/          # Final compacted datasets
│   ├── router/                 # Router training data
│   ├── inference_base/         # Base model inference results
│   ├── inference_moe/          # MoE v1 inference results
│   ├── inference_moe_v2/       # MoE v2 inference results
│   └── report/                 # Training & classifier reports
│
├── test/                       # Evaluation result files (excluded from git)
│
├── latex/
│   └── main.tex                # Final LaTeX research paper (IEEE format)
│
├── paper.tex                   # Alternate/draft LaTeX paper
│
├── final_report/               # BTP report documents, architecture diagrams, charts
│   ├── research_paper.md
│   ├── btp_report.md
│   ├── pns_scoring_explainer.md
│   ├── research_gap_analysis.md
│   ├── generate_charts.py
│   ├── Arch1.png / Arch2.png   # Architecture diagrams
│   └── chart_*.png             # Performance charts
│
├── ref/                        # Reference materials & planning documents
│   ├── BTP_Comprehensive_Implementation_Plan.md
│   ├── BTP_IEEE_Research_Paper_Master.md
│   ├── ideas.md
│   ├── simple_explanation.md
│   └── ...
│
├── overview.png                # High-level architecture overview image
├── analysis_output.json        # Expert analysis output
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## Pipeline

The full end-to-end training pipeline is scripted in `scripts/`. Run them in order on a GPU node:

```bash
# 1. Generate multi-step reasoning traces from raw benchmarks
bash scripts/01_generate_all_traces.sh

# 2. Compute PNS scores via counterfactual rollouts
bash scripts/02_compute_all_pns.sh

# 3. Prune low-PNS steps and compact traces
bash scripts/03_prune_compact_data.sh

# 4-6. Taxonomy sampling, labeling, and classifier training
bash scripts/04_sample_taxonomy_steps.sh
bash scripts/05_auto_label_taxonomy.sh
bash scripts/06_train_taxonomy_classifier.sh

# 7. Tag all data with expert categories
bash scripts/07_tag_all_data.sh

# 8. Atomic compaction
bash scripts/08_atomic_compactor.sh

# Then train experts, router, and the full Causal MoE V2 model
# via src/experts/, src/router/, and src/causal_moe_v2/
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `src/pns_engine/` | Computes Probabilistic Necessity Scores using teacher model (Qwen-2.5-72B-Instruct) counterfactual rollouts |
| `src/causal_moe_v2/architecture.py` | Defines the surgical MoE injection: frozen base model + gating network + expert LoRA adapters |
| `src/causal_moe_v2/train.py` | DeepSpeed-accelerated training loop with ZeRO-2 |
| `src/causal_moe_v2/eval.py` | Batched inference & accuracy evaluation |
| `src/router/` | Trains the step-level expert router (classifier head) |
| `src/experts/` | Per-domain LoRA fine-tuning (Math, Logic, Commonsense, Verify) |

---

## Results Summary

| Model | GSM8K Acc | CSQA Acc | Params Active | Reasoning Density |
|-------|-----------|----------|--------------|-------------------|
| Base (Qwen-2.5-7B) | ~72% | ~68% | 7B | 1.0× |
| Causal MoE V1 | ~75% | ~71% | 7B + 4×LoRA | ~1.05× |
| **Causal MoE V2** | **~77%** | **~73%** | **7B + surgical** | **~1.12×** |

> *Data not included in repository. See `data/report/` for training reports.*

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `peft`, `deepspeed`, `datasets`, `accelerate`

---

## Configuration

- **DeepSpeed**: `configs/ds_config.json` — ZeRO-2 configuration for multi-GPU training
- **Model**: Qwen/Qwen2.5-7B-Instruct (base), Qwen/Qwen2.5-72B-Instruct (teacher for PNS)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{biswas2025causalmoe,
  title   = {Causal Mixture-of-Experts: Surgical MoE Injection via PNS-Guided Expert Routing},
  author  = {Prantik Biswas},
  year    = {2025},
  note    = {B.Tech Project, IIT BHU}
}
```

---

## Notes

- **Data is not included** in this repository due to size. The `data/` directory structure is preserved via `.gitkeep` files. Download raw benchmarks using `src/helper/download.py`.
- Reference PDFs (`ref/*.pdf`, `future/*.pdf`) are also excluded due to size.
- All model checkpoints and weights are excluded (`.pt`, `.safetensors`, etc.).

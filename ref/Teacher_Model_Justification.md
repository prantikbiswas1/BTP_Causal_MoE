# Teacher Model Justification & Baseline Generation Strategy

## Why `Qwen2.5-7B-Instruct`?
In our BTP project, we are generating our "baseline" or "gold" dataset of Chain-of-Thought (CoT) reasoning traces. To do this, we are using the `Qwen2.5-7B-Instruct` model as our "Teacher."

This model was specifically chosen because:
1. **The "Small-but-Mighty" Sweet Spot:** `Qwen2.5-7B-Instruct` is widely recognized as one of the best open-source models in its weight class, often outperforming models twice its size in math and logic tasks.
2. **Hardware Constraints:** At 7 Billion parameters, it perfectly fits across the available hardware (2x NVIDIA RTX A5000s with 24GB VRAM each). It leaves enough memory overhead for `vLLM` to generate traces incredibly fast using tensor parallelism. 
    * A smaller model would produce lower-quality reasoning with too many errors (making our "gold" dataset unusable).
    * A larger model (like 70B parameters) would not fit securely or solve the traces fast enough for practical BTP experimentation.

## How this compares to the original PNS Research Paper
The target paper ("Causalmath" / PNS for reasoning) likely utilized a massive, costly API model (like GPT-4) or a massive open-source model (like Llama-3-70B) to generate their baseline traces and compute their probabilistic counterfactuals.

**Why we deviated from the paper:**
1. **Cost & Feasibility:** Using the GPT-4 API to generate 37,000 long CoT traces (and then running the subsequent counterfactual generation on all of them) would cost hundreds of dollars.
2. **Local Reproducibility:** The core of this BTP is building an end-to-end pipeline that can be actively run, tested, and debugged on the available cloud server.

By utilizing `Qwen2.5-7B-Instruct` as a local teacher, we strictly adhere to the fundamental scientific methodology established in the paper (Teacher generates traces -> PNS engine scores/prunes them -> Small model is fine-tuned). The only difference is our execution is purely local and cost-free, rather than relying on a commercial API.

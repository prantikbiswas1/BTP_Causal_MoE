# The Entire BTP Project — Explained in Simple Words

> Written to be understood by anyone. No prior AI knowledge needed.
> Start here if you want to understand what you are building and why.

---

## Part 1: The Problem — AI Models Are Bad at "Thinking Efficiently"

### What is Chain-of-Thought reasoning?

Modern AI models like ChatGPT, DeepSeek, and Qwen don't just spit out answers. For complex questions, they *write out their thinking* before giving a final answer. This is called **Chain-of-Thought (CoT) reasoning**.

**Example — Ask an AI: "What is 99² + 99 + 1?"**

The AI might write:

> Step 1: Let me think. 99 squared...  
> Step 2: I know there's a formula: a² + a + 1 = (a³ − 1)/(a − 1). Let me set a = 99.  
> Step 3: Hmm, but does that actually help me compute this mentally? Maybe not directly.  
> Step 4: Let me try (100 − 1)² = 10000 − 200 + 1 = 9801  
> Step 5: 9801 + 99 + 1 = 9901  
> **Answer: 9901** ✅

The answer is correct — but look at Steps 2 and 3. They went down a dead-end path and contributed nothing to the final answer. The model was **overthinking**, spending extra time and money on useless steps.

### Why does this matter?

Every "token" (roughly a word) an AI generates costs compute time and money. When LLMs are used at scale — millions of queries a day — those wasted tokens add up enormously.

There are two failure modes:
1. **Too many steps (Overthinking)** — includes redundant, unnecessary reasoning → wastes tokens
2. **Too few steps (Underthinking)** — skips a critical step → gets the wrong answer

The ideal is somewhere in the middle: **just enough reasoning, nothing more.**

---

## Part 2: The NeurIPS 2025 Paper — The Mathematical Solution

### What is NeurIPS?

NeurIPS (Neural Information Processing Systems) is the **#1 AI research conference in the world**. Getting a paper published there means your work has been reviewed and validated by the top researchers on Earth. The paper you read was accepted at NeurIPS 2025.

### What did the paper propose?

The paper introduces a concept called **PNS — Probability of Necessity and Sufficiency**.

PNS is a way to mathematically score each reasoning step by asking two questions:

#### Question 1: Is this step *Sufficient*?
"If I use only this reasoning chain, does it lead to the correct answer?"
→ If yes: the chain is **sufficient** (it covers everything needed)

#### Question 2: Is this step *Necessary*?
"If I remove (or replace) this specific step, does the answer break?"
→ If yes: the step is **necessary** (you can't do without it)
→ If no: the step can be safely deleted

#### Combined — PNS Score
PNS = A score from 0 to 1 that measures how much a step is *both* necessary and sufficient at the same time.

- PNS ≈ 1.0 → This step is crucial. Keep it.
- PNS ≈ 0.0 → This step does nothing. Delete it.

### How do they test if a step is necessary?

This is the clever part. They use **counterfactual rollouts** — a technique from causal inference (the same math used in medicine and economics to test "what would have happened if...").

For each reasoning step, they:
1. Replace that step with a different, corrupted/alternative version
2. Let the AI continue from that replaced step
3. Check if the final answer is still correct

- If the answer is still correct after replacing → the original step wasn't needed → **low PNS**
- If replacing that step causes the answer to break → the step was essential → **high PNS**

This is done multiple times per step (called *k rollouts*), and the average gives us the PNS score.

### What they did with PNS

1. Generated full (possibly messy) reasoning chains from a big LLM
2. Scored every reasoning step using PNS
3. Kept only the high-PNS steps, deleted the rest → **compact CoT**
4. Used these compact CoTs to **fine-tune smaller models** to reason efficiently

### Results from the paper

| Dataset | Original tokens | After PNS pruning | Accuracy change |
|---------|----------------|-------------------|----------------|
| GSM-8k (grade math) | 113 tokens | 27 tokens (76% less) | 95% → 97.9% (**improved!**) |
| MATH-500 (harder math) | 387 tokens | 161 tokens (58% less) | 85.9% → 91.6% (**improved!**) |
| CommonsenseQA | 368 tokens | 167 tokens (54% less) | 87.1% → 94.9% (**improved!**) |

The key insight: **fewer reasoning steps, better answers.** Cutting the junk actually helped accuracy because the model was no longer confusing itself with redundant paths.

The paper's code is publicly available: [github.com/yxn9191/causalmath](https://github.com/yxn9191/causalmath)

---

## Part 3: The Gap — What The Paper Didn't Do

The paper improved reasoning **within a standard decoder**. Think of a decoder as the "writer" part of the AI — it generates one word at a time.

The paper's decoder is a single, general-purpose writer. It handles every type of reasoning step — math, logic, commonsense — with the same architecture.

**The analogy:** Imagine a company with one employee who has to do accounting, legal review, customer service, and quality control all by themselves. They'd be mediocre at all of them.

What if you hired specialists — an accountant, a lawyer, a customer service rep, and a QA engineer? Each one would be significantly better at their specific task.

This is what's missing from the paper. And that's exactly what your BTP adds.

---

## Part 4: Mixture of Experts (MoE) — The Architectural Change Your Professor Wants

### What is MoE?

**Mixture of Experts (MoE)** is an architectural technique where, instead of one big network processing everything, you have **multiple smaller "expert" networks**, and a smart **router** decides which expert handles each input.

Inside a transformer (the architecture all modern LLMs use), there's a layer called the **Feed-Forward Network (FFN)**. In a standard transformer, this is one network. In an MoE transformer, this is replaced by **N expert networks + a router**.

**Standard Transformer (old way):**
```
Input → Attention Layer → [ One FFN ] → Output
```

**MoE Transformer (new way):**
```
Input → Attention Layer → [ Router ] → Expert 1 or Expert 2 or Expert 3 or Expert 4 → Output
```

**The key property of MoE:** Only 1 or 2 of the N experts activate for any given input. The rest stay silent. This means:
- The model has more total capacity (more experts = more knowledge)
- But uses less compute per forward pass (only a few experts activate)

This is called **sparse activation** — and it's why GPT-4, Gemini, and Mixtral all use MoE.

### Why does MoE help reasoning?

Different types of reasoning require different cognitive skills:
- Solving 99² requires algebraic manipulation
- "Where do you find a cup?" requires world knowledge
- Checking "does Step 4 follow from Step 3?" requires logical deduction

In a standard model, one FFN tries to do all three. In your MoE model:
- Expert 1 specializes in **mathematical/arithmetic reasoning**
- Expert 2 specializes in **logical/deductive reasoning**
- Expert 3 specializes in **commonsense/world knowledge**
- Expert 4 specializes in **verification and answer generation**

Each expert becomes better at its one job, and the model overall becomes more accurate and efficient.

---

## Part 5: Your BTP — Combining PNS + MoE

### The core innovation

Current MoE models use routers that just **balance the load** — they spread tokens evenly across experts. They don't know *what type of reasoning* is happening or *how important* a given step is.

**Your contribution:** Use **PNS scores to guide the router**.

- High-PNS step that is mathematical → send to **Math Expert**
- High-PNS step that is logical → send to **Logic Expert**
- Low-PNS step (redundant junk) → **drop it entirely, don't process it at all**

So the routing is no longer arbitrary — it's **causally informed**. The router understands the *causal importance* of each step and routes accordingly.

### The full architecture, step by step

```
User asks a question
        ↓
Step 1: A large LLM (e.g., DeepSeek-R1) generates a full reasoning trace
        (This trace may have redundant or missing steps)
        ↓
Step 2: PNS Scorer evaluates each reasoning step
        - Runs k counterfactual rollouts for each step
        - Computes PNS score (0 to 1) for each step
        - Flags low-PNS steps as redundant
        ↓
Step 3: Step-Type Classifier labels surviving steps
        - "This step involves arithmetic" → tag: MATH
        - "This step is a logical deduction" → tag: LOGIC
        - "This step uses world knowledge" → tag: COMMONSENSE
        ↓
Step 4: Causal MoE Router sends each step to the right expert
        - MATH steps → Expert 1 (math specialist)
        - LOGIC steps → Expert 2 (logic specialist)
        - COMMONSENSE steps → Expert 3 (knowledge specialist)
        - VERIFY steps → Expert 4 (answer generator)
        - LOW-PNS steps → DROPPED (not processed by any expert)
        ↓
Step 5: Experts generate their parts, assembled into final answer
        Result: Compact, accurate, efficiently-routed answer
```

### What makes this novel

| What exists | What you add |
|-------------|-------------|
| PNS-based pruning (the paper) | PNS-guided MoE routing |
| Standard MoE routing (load balancing) | Causally-informed routing (PNS scores) |
| Single decoder for all reasoning | Specialized experts per reasoning type |

No existing paper does all three together. This is an original contribution.

---

## Part 6: What Experiments You Run and What You Measure

### The standard benchmarks every AI researcher knows

| Dataset | What it tests | Number of examples |
|---------|--------------|-------------------|
| **GSM-8k** | Grade-school math word problems | 8,500 |
| **MATH-500** | Competition math (harder) | 500 |
| **CommonsenseQA** | Everyday common sense questions | 12,100 |
| **AIME 2025** | Advanced math olympiad (optional) | ~30 |

Your professors will immediately recognize these — they're industry-standard testbeds.

### What you measure

1. **Accuracy** — percentage of questions answered correctly (higher = better)
2. **Token count** — average number of tokens used to reason (lower = better)
3. **Step count** — average number of reasoning steps (lower = better)
4. **FLOPs** — floating point operations = compute cost (lower = better)
5. **Expert specialization** — does each expert actually handle different step types? (measurable via routing distributions)

### Your comparison table (what you fill in after experiments)

| Model | Avg Tokens | Accuracy |
|-------|-----------|---------|
| Original LLM (no pruning) | ~113 | ~90% |
| Paper's PNS SFT (baseline) | ~27 | ~97.9% |
| **Your PNS + MoE (your contribution)** | **~20?** | **≥ 97.9%?** |

Even if accuracy is similar, showing **fewer tokens AND fewer FLOPs** is a strong engineering result.

---

## Part 7: Month-by-Month Plan

### Month 1 — Reproduce the Paper
**Goal:** Get the paper's results yourself. This proves you understand the method.

What to do:
- Download the open-source code from GitHub
- Run PNS evaluation on GSM-8k using Qwen-2.5-7B (small enough for a single GPU)
- Check that your token reduction and accuracy numbers match Table 1 of the paper

**Output:** A table showing: "I reproduced the paper's results."
This alone is an honest and respectable BTP contribution.

### Month 2 — Add the MoE Layer
**Goal:** Modify the decoder architecture.

What to do:
- Train a small step-type classifier (label reasoning steps as Math/Logic/Commonsense/Verify)
- Replace the FFN layer in the decoder with a 4-expert MoE (use `DeepSpeed-MoE` library)
- Connect PNS scores to the routing gate

**Output:** A modified model that runs on a smaller scale (GPT-2 or TinyLlama) — proof the architecture works.

### Month 3 — Run Full Experiments
**Goal:** Compare your method against the paper's baseline.

What to do:
- Fine-tune DeepSeek-R1-Distill-Qwen-1.5B (small model, 1.5B parameters) using your causal MoE CoTs
- Evaluate on GSM-8k, CommonsenseQA, MATH-500
- Compare three systems: (a) Original LLM, (b) Paper's method, (c) Your PNS + MoE method

**Output:** A full results table with all metrics filled in.

### Month 4 — Analysis and Write-Up
**Goal:** Show your method works and explain why.

What to do:
- Ablation study: run your model *without* the MoE to show MoE was contributing
- Plot expert routing distributions (does Expert 1 really handle more math steps?)
- Write the BTP report
- Prepare presentation slides

**Output:** Final BTP report + presentation.

---

## Part 8: The Proof You Show Your Professors

Professors care about **evidence**. Here are 5 specific things you show them:

1. **Table: Reproduced paper results** → "I ran their code. I got the same numbers. The baseline is real."
2. **Table: Your method vs. baseline** → "My method uses 25% fewer tokens at equal accuracy."
3. **Graph: Expert routing distribution** → "Expert 1 handled 78% of math steps. Expert 2 handled 81% of logic steps." This proves specialization happened.
4. **Graph: PNS score histogram before and after pruning** → Shows average PNS per step increased after pruning (junk was removed).
5. **Ablation table: With MoE vs. Without MoE** → Proves MoE was the thing that helped.

These are all standard ML paper figures. Any professor with ML background will immediately understand and be impressed.

---

## Part 9: What to Say to Your Professors (Script)

> *"My BTP extends a NeurIPS 2025 paper on causal reasoning efficiency in large language models. The paper proved that using a concept from causal inference called Probability of Necessity and Sufficiency (PNS), you can identify and remove redundant reasoning steps in Chain-of-Thought prompting — achieving 76% token reduction while actually improving accuracy on standard benchmarks like GSM-8k.*
>
> *My contribution is an architectural extension: I replace the standard Feed-Forward Network in the decoder with a Mixture-of-Experts layer. The router that selects which expert to use is guided by PNS scores and reasoning step type — so math steps go to a math expert, logic steps go to a logic expert, and redundant low-PNS steps are dropped entirely before any expert processes them.*
>
> *I evaluate this on GSM-8k, MATH-500, and CommonsenseQA, measuring accuracy, token count, and computational cost. The paper's published code gives me a solid, reproducible baseline. My hypothesis is that expert specialization will provide additional efficiency gains on top of what the paper already achieved."*

---

## Glossary — Every Term in Plain English

| Term | What it means simply |
|------|---------------------|
| **LLM** | Large Language Model — e.g., ChatGPT, DeepSeek, Qwen. An AI that generates text. |
| **Chain-of-Thought (CoT)** | When the AI writes out step-by-step thinking before giving a final answer |
| **Reasoning step** | One step/sentence in the thinking process |
| **Redundant step** | A step that contributed nothing to the final answer (can be deleted) |
| **Token** | Roughly one word. LLMs work with tokens. More tokens = higher cost. |
| **PNS** | Probability of Necessity and Sufficiency — a score (0 to 1) for how essential a step is |
| **Sufficiency** | "Does this chain alone lead to the correct answer?" |
| **Necessity** | "Does removing this step break the answer?" |
| **Counterfactual rollout** | "What if this step was replaced with something else?" — tests necessity |
| **MoE (Mixture of Experts)** | Architecture with multiple specialist networks; only 1-2 activate per input |
| **Router / Gating network** | The component that decides which expert to use |
| **FFN** | Feed-Forward Network — a layer inside a transformer; MoE upgrades this layer |
| **Causal inference** | A mathematical framework to determine cause-and-effect (not just correlation) |
| **Fine-tuning (SFT)** | Taking a pre-trained LLM and training it further on your specific dataset |
| **In-Context Learning (ICL)** | Giving the model examples inside the prompt, without changing its weights |
| **FLOPs** | Floating Point Operations — a measure of compute used. Fewer FLOPs = more efficient. |
| **Ablation study** | Removing one component of your model to prove that component was helping |
| **Baseline** | The existing best method you compare against to show improvement |
| **NeurIPS** | Neural Information Processing Systems — the top AI research conference in the world |
| **GSM-8k** | A benchmark of 8,500 grade-school math problems — industry standard |
| **MATH-500** | A benchmark of 500 harder math problems spanning algebra, geometry, etc. |
| **CommonsenseQA** | A benchmark testing everyday reasoning ("Where do you find a cup?") |
| **Sparse activation** | In MoE: only a few experts activate per input, saving compute |
| **DeepSeek-R1** | A reasoning LLM by DeepSeek (powerful open-source reasoning model) |
| **Qwen-2.5** | A family of LLMs by Alibaba — used in the paper's experiments |

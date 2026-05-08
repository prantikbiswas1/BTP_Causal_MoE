# PNS Scoring Algorithm — Technical Reference

> **Source files**: `src/pns_engine/generate_pns_gsm8k.py` · `src/pns_engine/generate_pns_csqa.py`  
> **Project**: Causal-MoE BTP Thesis — IIT Patna, April 2026

---

## 1. What is PNS?

**PNS (Probabilistic Necessity Score)** measures how *necessary* a specific reasoning step is for a model to arrive at the correct answer. A high-PNS step is one the model cannot bypass — sabotaging it causes consistent failure. A low-PNS step is redundant or easily substituted.

PNS is used in this project to **prune verbose multi-step traces** down to only the causally essential steps, producing a compact and high-signal training dataset for the Causal-MoE model.

---

## 2. Core Formula

For a reasoning step $s_i$ within a correct trace for question $q$, with ground truth $y^*$:

$$\text{PNS}(s_i) = 1 - \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}\bigl[\hat{y}_k = y^*\bigr]$$

| Symbol | Meaning |
| :--- | :--- |
| $K$ | Number of stochastic sabotage rollouts (= **3** in both scripts) |
| $\hat{y}_k$ | Model's predicted answer on rollout $k$, after being told $s_i$ is **wrong** |
| $y^*$ | Ground truth answer |
| $\mathbf{1}[\cdot]$ | Indicator: 1 if prediction matches ground truth, else 0 |

**Intuition:**

- If the model *still gets it right* after the sabotage → the step wasn't necessary → **PNS is low**.
- If the model *consistently fails* after the sabotage → the step was critical → **PNS is high**.

---

## 3. The Sabotage-Rollout Method

Rather than silently deleting a step (ablation), the implementation **actively injects a contradiction** — it tells the model the step is wrong and forces a re-solve from that point.

### Why sabotage instead of ablation?

Silent ablation leaves an ambiguous gap in the context. The model may reconstruct the missing information from surrounding steps, giving a falsely low PNS. Explicit sabotage poisons the chain of thought with a direct contradiction, creating a much stronger counterfactual test.

---

## 4. Execution Flow

### Step 1 — Load only correct traces

```python
data = [json.loads(line) for line in f if json.loads(line).get("is_correct")]
```

PNS is only meaningful within a *correct* solution. Sabotaging wrong traces scores steps against an already-failing baseline, which is uninformative.

---

### Step 2 — Build sabotage prompts

For each item and each step $s_i$, a prompt is created using all preceding steps as context.

**GSM8K** (`generate_pns_gsm8k.py`, lines 66–71):

```
Question: {query}
Context so far: {steps[0] ... steps[i-1]}
ERROR: The previous thought '{s_i}' is mathematically INCORRECT.
Task: Solve the question again from this point. End with #### [Answer]:
```

**CSQA** (`generate_pns_csqa.py`, lines 64–70):

```
Question: {question_with_choices}
Context so far: {steps[0] ... steps[i-1]}
ERROR: The previous deduction '{s_i}' has been proven WRONG.
Task: Solve the question again from this point without using that wrong logic.
End strictly with #### [Answer Letter]:
```

Each prompt is repeated **K = 3 times** (with `temperature=0.7`) to sample diverse rollouts.

---

### Step 3 — Batch inference via vLLM

```python
outputs = llm.generate(all_sabotage_prompts, sampling_params, use_tqdm=False)
```

All prompts for a batch are dispatched in a single vLLM call for throughput efficiency.

**Sampling configuration:**

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `temperature` | 0.7 | Non-zero → diverse rollout paths across K samples |
| `max_tokens` | 80 | Only an answer is needed, not a full re-trace |
| `top_p` | 0.9 | Nucleus sampling for coherent but varied completions |

---

### Step 4 — Extract answers

**GSM8K** — looks for `#### <number>`, falls back to last number in text:

```python
match = re.search(r"####\s*(-?[\d,.]+)", text)
```

**CSQA** — looks for `#### <letter>`, falls back to last standalone A–E:

```python
match = re.search(r"####\s*[^\d\-A-E]*([A-E])", text, re.IGNORECASE)
```

---

### Step 5 — Compute PNS

```python
pns_val = 1 - (sum(v_scores) / len(v_scores)) if v_scores else 0.0
```

`v_scores` is a list of K binary values. Their mean is the **bypass rate** (fraction of rollouts where the model recovered despite the sabotage).

| Bypass rate | PNS value | Interpretation |
| :---: | :---: | :--- |
| 0 / 3 = 0.00 | **1.00** | Step is indispensable — model always fails without it |
| 1 / 3 ≈ 0.33 | **0.67** | Step is important — model usually cannot recover |
| 2 / 3 ≈ 0.67 | **0.33** | Step is marginal — model often finds another path |
| 3 / 3 = 1.00 | **0.00** | Step is redundant — model always recovers |

---

### Step 6 — Filter with threshold α

```python
ALPHA = 0.5

if pns_val >= ALPHA:
    s_final.append(step_text)
```

Only steps with `PNS ≥ 0.5` are kept in `s_final`. Steps below the threshold are discarded as non-necessary. The final `[VERIFY] #### <answer>` step is **excluded from scoring** (`steps[:-1]`) and always retained in the output.

---

## 5. Output Schema

Each scored item is written to `pns_scored/*.jsonl` with two new fields:

```json
{
  "question":     "Janet's ducks lay 16 eggs per day...",
  "steps":        ["[MATH] 16 - 3 - 4 = 9", "[MATH] 9 * $2 = $18", "[VERIFY] #### 18"],
  "ground_truth": 18.0,
  "is_correct":   true,

  "pns_analysis": [
    { "text": "[MATH] 16 - 3 - 4 = 9", "pns": 1.0  },
    { "text": "[MATH] 9 * $2 = $18",    "pns": 0.67 }
  ],

  "s_final": [
    "[MATH] 16 - 3 - 4 = 9",
    "[MATH] 9 * $2 = $18"
  ]
}
```

---

## 6. Differences Between GSM8K and CSQA Scripts

| Property | `generate_pns_gsm8k.py` | `generate_pns_csqa.py` |
| :--- | :--- | :--- |
| Batch size | 32 | 64 |
| Query field | `question` | `question_with_choices` |
| Sabotage wording | *"mathematically INCORRECT"* | *"has been proven WRONG"* |
| Answer extractor | `#### <number>` | `#### <letter A–E>` |
| Answer comparison | Numeric string equality | Uppercase letter equality |

---

## 7. Scale & Infrastructure

**Prompt volume per batch:**

```
Total prompts = BATCH_SIZE × avg_steps × K
```

- GSM8K: `32 × 4 × 3 = 384` prompts / batch
- CSQA:  `64 × 4 × 3 = 768` prompts / batch

**Hardware & runtime config:**

| Setting | Value |
| :--- | :--- |
| Inference engine | vLLM with Ray distributed backend |
| Tensor parallelism | 2 GPUs (`tensor_parallel_size=2`) |
| GPU memory util. | 90% (`gpu_memory_utilization=0.90`) |
| Compute dtype | BF16 |
| Model | Qwen2.5-7B-Instruct |

Results are flushed to disk after every batch (`out_f.flush()`) to allow live monitoring and prevent data loss on long HPC runs.

---

## 8. Design Decisions

| Decision | Code location | Rationale |
| :--- | :--- | :--- |
| Filter to correct traces only | `if json.loads(line).get("is_correct")` | Necessity is relative to a correct solution |
| Exclude last `[VERIFY]` step | `steps[:-1] if "####" in steps[-1]` | It is an answer marker, not a reasoning step — always PNS ≈ 1.0 trivially |
| Explicit sabotage over ablation | `ERROR:` prefix in prompt | Prevents gap-inference; forces genuine counterfactual reasoning |
| K = 3 stochastic rollouts | `for _ in range(K)` with `temperature=0.7` | Reduces binary score noise at minimal compute cost |
| α = 0.5 threshold | `if pns_val >= ALPHA` | Empirical midpoint — steps bypassed >50% of the time contribute no unique signal |
| Short `max_tokens=80` | `SamplingParams(max_tokens=80)` | Rollouts need only an answer, not a full trace → 4–5× faster than full inference |

---

## 9. Algorithm Pseudocode

```
Input:   correct_traces  ← list of {question, steps[], ground_truth}
Output:  pns_scored.jsonl

for each item in correct_traces:

    scoring_steps ← item.steps[:-1]          # drop final [VERIFY] step

    for each step s_i in scoring_steps:
        prompt ← sabotage_prompt(item.question, context=steps[:i], s_i)
        rollouts ← [model.generate(prompt) for _ in range(K)]
        bypass_rate ← mean([ correct(r, item.ground_truth) for r in rollouts ])
        pns[s_i] ← 1 - bypass_rate

    pns_analysis ← [ {text: s_i, pns: pns[s_i]} for s_i in scoring_steps ]
    s_final      ← [ s_i for s_i in scoring_steps if pns[s_i] >= ALPHA ]

    write_jsonl( {**item, pns_analysis, s_final} )
```

---

*Reference: `src/pns_engine/generate_pns_gsm8k.py` · `src/pns_engine/generate_pns_csqa.py`*

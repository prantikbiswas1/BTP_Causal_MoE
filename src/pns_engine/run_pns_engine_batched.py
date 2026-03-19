import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams

from sufficiency import is_trace_sufficient
from counterfactuals import get_counterfactual_prompt
from pns_calculator import split_into_steps


def run_batched_pns_engine(args):
    print(f"Loading traces from {args.input}...")

    dataset = []
    with open(args.input, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    # -------------------------
    # CHECKPOINTING
    # -------------------------
    processed_ids = set()
    if os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except:
                        pass

    items_to_process = [x for x in dataset if x.get("id") not in processed_ids]

    if not items_to_process:
        print("All items already processed.")
        return

    print("Initializing vLLM...")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )

    BATCH_SIZE = 32  # tune this (16–64)

    print(f"Processing {len(items_to_process)} items...")

    with open(args.output, "a") as out_f:

        for batch_start in tqdm(range(0, len(items_to_process), BATCH_SIZE)):
            batch_items = items_to_process[batch_start:batch_start + BATCH_SIZE]

            results = defaultdict(lambda: defaultdict(list))

            # -------------------------
            # STEP 1: CF PROMPTS
            # -------------------------
            cf_prompts = []
            cf_meta = []

            for item_idx, item in enumerate(batch_items):
                problem = item["question"]
                gt = str(item.get("ground_truth", ""))
                traces = item.get("generated_traces", [])

                for trace_idx, trace in enumerate(traces):
                    if not is_trace_sufficient(trace, gt):
                        steps = split_into_steps(trace)
                        results[item_idx][trace_idx] = [
                            {"text": s, "pns_score": 0.0} for s in steps
                        ]
                        continue

                    steps = split_into_steps(trace)

                    for step_idx, step in enumerate(steps):
                        prev = "\n".join(steps[:step_idx])
                        prompt = get_counterfactual_prompt(problem, prev, step)

                        cf_prompts.append(prompt)
                        cf_meta.append((item_idx, trace_idx, step_idx, steps, gt, problem))

            # -------------------------
            # STEP 2: CF GENERATION
            # -------------------------
            cf_outputs = []
            if cf_prompts:
                cf_outputs = llm.generate(
                    cf_prompts,
                    SamplingParams(temperature=0.7, max_tokens=100),
                    use_tqdm=False
                )

            # -------------------------
            # STEP 3: BUILD ROLLOUTS
            # -------------------------
            rollout_prompts = []
            rollout_meta = []
            step_data = {}  # FIXED storage

            for i, out in enumerate(cf_outputs):
                item_idx, trace_idx, step_idx, steps, gt, problem = cf_meta[i]

                alt_step = out.outputs[0].text.strip()
                prev = "\n".join(steps[:step_idx])

                prompt = (
                    f"<|im_start|>user\nQuestion: {problem}\n"
                    f"Think step-by-step and provide the final answer.<|im_end|>\n"
                    f"<|im_start|>assistant\n{prev}\n{alt_step}\n"
                )

                key = (item_idx, trace_idx, step_idx)

                for _ in range(3):
                    rollout_prompts.append(prompt)
                    rollout_meta.append(key)

                step_data[key] = (steps, alt_step, gt)

            # -------------------------
            # STEP 4: ROLLOUT GENERATION
            # -------------------------
            rollout_outputs = []
            if rollout_prompts:
                rollout_outputs = llm.generate(
                    rollout_prompts,
                    SamplingParams(temperature=0.7, max_tokens=512),
                    use_tqdm=False
                )

            # -------------------------
            # STEP 5: GROUP RESULTS
            # -------------------------
            grouped = defaultdict(list)

            for i, ro in enumerate(rollout_outputs):
                key = rollout_meta[i]
                grouped[key].append(ro.outputs[0].text)

            # -------------------------
            # STEP 6: SCORING
            # -------------------------
            for key, outputs in grouped.items():
                item_idx, trace_idx, step_idx = key
                steps, alt_step, gt = step_data[key]

                prev = "\n".join(steps[:step_idx])

                correct = 0
                for out_text in outputs:
                    full_trace = f"{prev}\n{alt_step}\n{out_text}"
                    if is_trace_sufficient(full_trace, gt):
                        correct += 1

                expected_y = correct / len(outputs)
                pn = 1.0 - expected_y

                while len(results[item_idx][trace_idx]) <= step_idx:
                    results[item_idx][trace_idx].append(None)

                results[item_idx][trace_idx][step_idx] = {
                    "text": steps[step_idx],
                    "pns_score": round(pn, 4)
                }

            # -------------------------
            # STEP 7: SAVE OUTPUT
            # -------------------------
            for item_idx, item in enumerate(batch_items):
                scored_traces = []

                for trace_idx in sorted(results[item_idx].keys()):
                    steps = results[item_idx][trace_idx]
                    steps = [s for s in steps if s is not None]
                    scored_traces.append(steps)

                out_item = {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item.get("ground_truth", ""),
                    "scored_traces": scored_traces
                }

                out_f.write(json.dumps(out_item) + "\n")

            out_f.flush()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    run_batched_pns_engine(args)
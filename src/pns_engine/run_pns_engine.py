import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM

# Import the logic functions we created
from sufficiency import is_trace_sufficient
from counterfactuals import generate_counterfactual_step
from pns_calculator import calculate_pns_for_trace

def main():
    parser = argparse.ArgumentParser(description="Run PNS calculation on generated traces.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use for counterfactuals/rollouts")
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl with traces (e.g. gsm8k_train_traces.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to save PNS-scored traces")
    args = parser.parse_args()

    print(f"Loading traces from {args.input}...")
    dataset = []
    with open(args.input, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Initializing vLLM with model: {args.model}")
    # Same settings as generation: trust remote code, 2 GPUs (if applicable), disable eager/custom to avoid OOM
    llm = LLM(
        model=args.model, 
        trust_remote_code=True, 
        tensor_parallel_size=2, # Use both A5000s
        max_model_len=2048,     # Keep this low so we don't OOM during hundreds of rollouts
        disable_custom_all_reduce=True,
        enforce_eager=True
    )

    print(f"Scoring {len(dataset)} items. This will take a while...")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w') as out_f:
        for item in tqdm(dataset):
            problem = item.get('question')
            ground_truth = str(item.get('ground_truth', ''))
            traces = item.get('generated_traces', [])
            
            scored_traces = []
            
            # Score each of the 5 traces the model generated
            for trace in traces:
                scored_steps = calculate_pns_for_trace(
                    llm_engine=llm,
                    problem=problem,
                    ground_truth=ground_truth,
                    original_trace_text=trace,
                    is_trace_sufficient_fn=is_trace_sufficient,
                    generate_counterfactual_fn=generate_counterfactual_step
                )
                scored_traces.append(scored_steps)
            
            # Save the result for this question
            scored_item = {
                "id": item.get('id'),
                "question": problem,
                "ground_truth": ground_truth,
                "scored_traces": scored_traces
            }
            out_f.write(json.dumps(scored_item) + '\n')

    print(f"Done! Scored dataset saved to: {args.output}")

if __name__ == "__main__":
    main()

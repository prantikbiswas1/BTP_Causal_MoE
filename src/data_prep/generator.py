import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT traces using vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model path or HF ID")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input jsonl file (e.g. gsm8k_train.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl file")
    parser.add_argument("--k", type=int, default=5, help="Number of traces to generate per query")
    parser.add_argument("--temp", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum generation tokens")
    return parser.parse_args()

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    args = parse_args()
    
    print(f"Loading {args.dataset}...")
    dataset = load_data(args.dataset)
    
    # Initialize vLLM
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=1) # Adjust TP size based on GPUs
    
    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        n=args.k, # Generate k sequences per prompt
    )
    
    # Prepare prompts (Dataset format dependent, assuming "question" or "problem" field exists)
    prompts = []
    for item in dataset:
        q = item.get('question', item.get('problem', ''))
        # Using a standard instruction format
        prompt = f"<|im_start|>user\nQuestion: {q}\nThink step-by-step and provide the final answer.<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
    
    print(f"Generating {args.k} traces per query for {len(prompts)} queries...")
    # vLLM handles batching internally
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"Saving outputs to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w') as f:
        for i, (item, output) in enumerate(zip(dataset, outputs)):
            item_traces = []
            for j in range(len(output.outputs)):
                item_traces.append(output.outputs[j].text)
            
            # Save the original question, ground truth answer, and our generated traces
            out_item = {
                "id": i,
                "question": item.get('question', item.get('problem', '')),
                "ground_truth": item.get('answer', item.get('solution', '')),
                "generated_traces": item_traces
            }
            f.write(json.dumps(out_item) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    main()

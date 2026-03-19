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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for checkpointing")
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
    
    # -------------------------
    # CHECKPOINTING
    # -------------------------
    processed_questions = set()
    if os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_questions.add(json.loads(line)["question"])
                    except:
                        pass

    items_to_process = []
    for item in dataset:
        q = item.get('question', item.get('problem', ''))
        if q not in processed_questions:
            items_to_process.append(item)

    if not items_to_process:
        print("All items already processed.")
        return

    print(f"Items remaining to process: {len(items_to_process)}")
    
    # -------------------------
    # VLLM INITIALIZATION
    # -------------------------
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=2,          # Match PNS engine scaling
        gpu_memory_utilization=0.95,     # Match PNS engine memory use
        max_model_len=8192               # Match PNS engine context length
    )
    
    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        n=args.k, # Generate k sequences per prompt
    )
    
    print(f"Generating {args.k} traces per query in batches of {args.batch_size}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "a") as out_f:
        for batch_start in tqdm(range(0, len(items_to_process), args.batch_size)):
            batch_items = items_to_process[batch_start:batch_start + args.batch_size]
            
            prompts = []
            for item in batch_items:
                q = item.get('question', item.get('problem', ''))
                # Using a standard instruction format
                prompt = f"<|im_start|>user\nQuestion: {q}\nThink step-by-step and provide the final answer.<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(prompt)
                
            # vLLM generate for this specific batch
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # Save and flush to disk immediately after generation
            for item, output in zip(batch_items, outputs):
                item_traces = []
                for j in range(len(output.outputs)):
                    item_traces.append(output.outputs[j].text)
                
                out_item = {
                    "question": item.get('question', item.get('problem', '')),
                    "ground_truth": item.get('answer', item.get('solution', '')),
                    "generated_traces": item_traces
                }
                out_f.write(json.dumps(out_item) + '\n')
            
            out_f.flush()

    print("Done!")

if __name__ == "__main__":
    main()

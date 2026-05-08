import json
import re
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
# Works for both CSQA and GSM8K as long as they are Tagged first
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/csqa_tagged_final.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/csqa_atomic.jsonl"
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

def clean_compressed_output(text):
    """
    STRICT CLEANER: Removes instruction leakage and kills loops.
    """
    # 1. Strip common hallucinated prefixes/instruction leaks
    patterns_to_remove = [
        r"Instruction:.*", 
        r"Compress the following.*", 
        r"Atomic Fact[:\s-]*", 
        r"Original[:\s-]*", 
        r"Thought[:\s-]*",
        r"Output[:\s-]*",
        r"Short high-density.*"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 2. Kill Loops: If model repeats itself, keep only the first unique atomic unit
    parts = re.split(r'[.!?\n]', text)
    unique_parts = []
    seen = set()
    for p in parts:
        p_clean = p.strip().lower()
        if p_clean and p_clean not in seen:
            unique_parts.append(p.strip())
            seen.add(p_clean)
            break # We ONLY want the first atomic fact
    
    final_text = unique_parts[0] if unique_parts else text.strip()
    
    # 3. Final sanitization: No brackets or quotes in the middle of logic
    final_text = final_text.replace('"', '').replace('[', '').replace(']', '').strip()
    return final_text

def build_compressed_dataset():
    # 1. Setup vLLM Engine
    print("Initializing vLLM Engine for Final Atomic Compression...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    # CRITICAL: Cap tokens at 15 to prevent the model from having space to loop
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=15, 
        stop=["\n", ".", "Instruction:", "["]
    )
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Compressing {len(data)} samples into high-density tokens...")
    BATCH_SIZE = 64 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_prompts = []
            prompt_map = [] 

            for b_idx, item in enumerate(batch_items):
                # FIXED REGEX: Uses non-greedy matching and positive lookahead to avoid [VERIFY]
                raw_trace = item["atomic_reasoning"]
                steps = re.findall(r"\[(LOGIC|COMMONSENSE|MATH)\]\s*(.*?)(?=\s*\[|$)", raw_trace)
                
                for s_idx, (tag, content) in enumerate(steps):
                    # ChatML Prompt for strict instruction following
                    prompt = (
                        f"<|im_start|>system\n"
                        f"You are a text compressor. Convert thoughts into 5-word facts. No filler.<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"Target: {content.strip()}\n"
                        f"Atomic:<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    all_prompts.append(prompt)
                    prompt_map.append((b_idx, s_idx, tag))

            if not all_prompts:
                continue

            # Batch Inference
            outputs = llm.generate(all_prompts, sampling_params, use_tqdm=False)
            
            # Map back to items
            batch_results = {idx: {} for idx in range(len(batch_items))}
            for out, (b_idx, s_idx, tag) in zip(outputs, prompt_map):
                raw_compressed_text = out.outputs[0].text.strip()
                refined_text = clean_compressed_output(raw_compressed_text)
                batch_results[b_idx][s_idx] = f"[{tag}] {refined_text}"

            # Assemble and Write
            for b_idx, item in enumerate(batch_items):
                sorted_idx = sorted(batch_results[b_idx].keys())
                sorted_steps = [batch_results[b_idx][idx] for idx in sorted_idx]
                
                # Final reconstruction for Causal MoE Experts
                new_trace = " ".join(sorted_steps) + f" [VERIFY] #### {item['answer']}"
                
                # Keep the instruction and answer intact
                f_out.write(json.dumps({
                    "id": item["id"],
                    "instruction": item["instruction"],
                    "atomic_reasoning": new_trace,
                    "answer": item["answer"]
                }) + "\n")
            
            f_out.flush() # Live updates for cluster monitoring

    print(f"Success! Compressed dataset saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Standard cleanup before cluster run
    # os.system("ray stop; pkill -f vllm") 
    build_compressed_dataset()
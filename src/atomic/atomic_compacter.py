import multiprocessing
import os
import re
import json
import argparse

# --- CRITICAL CLUSTER FIX: Same as your previous version ---
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from vllm import LLM, SamplingParams
from tqdm import tqdm

# THE "STRICT SEPARATION" PROMPT 
# This keeps the BTP logic correct while using your same taxonomy tags
SANITIZER_SYSTEM_PROMPT = """You are a Reasoning Distiller. Convert the Trace into a "Strict Atomic Trace".
Rules:
1. Use tags: [LOGIC], [MATH], [COMMONSENSE], [VERIFY].
2. [LOGIC]: Use ONLY for text-to-variable extraction (e.g., coal = 6) or choosing the final answer.
3. [MATH]: MANDATORY for ANY numeric operation, comparison, or formula (e.g., 6/2=3, or max(3,4)=4). 
4. [COMMONSENSE]: Use for definitions (e.g., 1 dozen = 12).
5. STRICTURE: Do NOT put math symbols (+, -, *, /, =) inside [LOGIC].
6. No conversational filler. SHORTEST possible path.

Example Output:
[LOGIC] coal = 6
[LOGIC] per_station = 2
[MATH] stations = 6 / 2 = 3
[VERIFY] ### 3"""

CRITIC_SYSTEM_PROMPT = "Respond with ONLY the final answer (Letter or Number)."

def extract_answer(text):
    text = str(text).upper().strip()
    match = re.search(r"###\s*([A-E]|\d+)", text)
    if match: return match.group(1)
    return ""

def generate_unified_btp_data(model_path, input_file, output_file, batch_size=128):
    if not os.path.exists(input_file): return

    # --- CONFIGURATION: Kept exactly the same as your previous script ---
    print(f"Loading teacher model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.75, 
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16"
    )
    
    # Kept max_tokens same to enforce compaction
    params = SamplingParams(temperature=0, max_tokens=150)

    # Load Data
    with open(input_file, "r") as f:
        all_data = [json.loads(line) for line in f]

    print(f"Generating Strict Atomic Traces for {len(all_data)} items...")
    with open(output_file, "a") as out_f:
        for i in tqdm(range(0, len(all_data), batch_size)):
            batch = all_data[i : i + batch_size]
            
            prompts = []
            for item in batch:
                # Uses the first compact trace as source
                source = item.get("compact_traces", [""])[0]
                prompt = f"<|im_start|>system\n{SANITIZER_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nQuestion: {item['question']}\nTrace: {source}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(prompt)

            outputs = llm.generate(prompts, params)
            
            for idx, output in enumerate(outputs):
                trace = output.outputs[0].text.strip()
                pred = extract_answer(trace)
                
                # Ground Truth Check for both GSM and CSQA
                gt_raw = str(batch[idx].get("ground_truth", "")).strip().upper()
                gt_match = re.search(r"####\s*(\d+)|(^([A-E])$)", gt_raw)
                gt = gt_match.group(1) or gt_match.group(2) if gt_match else gt_raw[0]

                if pred == gt:
                    batch[idx]["atomic_trace"] = trace
                    batch[idx]["is_golden"] = True
                    out_f.write(json.dumps(batch[idx]) + "\n")
            
            out_f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    generate_unified_btp_data(args.model, args.input, args.output, args.batch_size)
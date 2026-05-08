import json
import os
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct" 
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/gsm8k_raw_multistep_traces.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/gsm8k_pns_scored.jsonl"

K = 3 
BATCH_SIZE = 32 
ALPHA = 0.5 

def extract_gsm8k_answer(text):
    """Extracts numeric answer from #### or the final number."""
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?[\d,.]+", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""

def run_gsm8k_sabotage_pns(model_path, input_path, output_path):
    # Initialize vLLM - Ray backend for your cluster
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=2, 
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        # We only process correct traces to find necessary steps
        data = [json.loads(line) for line in f if json.loads(line).get("is_correct")]

    # OPTIMIZED: max_tokens=80 is enough for a math rollout after context
    sampling_params = SamplingParams(temperature=0.7, max_tokens=80, top_p=0.9) 

    print(f"CRITICAL: Generating TRUE Sabotage PNS for {len(data)} items...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as out_f:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_sabotage_prompts = []
            prompt_mapping = [] 

            for item_idx, item in enumerate(batch_items):
                query = item["question"]
                # Prune last step if it's just the answer marker
                steps = item["steps"][:-1] if "####" in item["steps"][-1] else item["steps"]
                
                for step_idx, st in enumerate(steps):
                    prev_context = "\n".join(steps[:step_idx])
                    
                    # --- TRUE SABOTAGE PROMPT ---
                    # Forces model to find a NEW path after being told a step is WRONG.
                    alt_prompt = (
                        f"Question: {query}\n"
                        f"Context so far: {prev_context}\n"
                        f"ERROR: The previous thought '{st}' is mathematically INCORRECT.\n"
                        f"Task: Solve the question again from this point. End with #### [Answer]: "
                    )
                    
                    for _ in range(K):
                        all_sabotage_prompts.append(alt_prompt)
                        prompt_mapping.append((item_idx, step_idx))

            if not all_sabotage_prompts:
                continue
            
            # FIXED: Correct positional argument call (no 'all_prompts=')
            outputs = llm.generate(
                all_sabotage_prompts, 
                sampling_params=sampling_params, 
                use_tqdm=False
            )
            
            # Aggregate Results
            results_agg = {idx: {s_idx: [] for s_idx in range(len(batch_items[idx]["steps"]))} 
                           for idx in range(len(batch_items))}
            
            for out, (item_idx, step_idx) in zip(outputs, prompt_mapping):
                pred = extract_gsm8k_answer(out.outputs[0].text)
                gt = str(batch_items[item_idx]["ground_truth"]).replace(",", "")
                # If model can bypass the "error" and still be right, PNS is low.
                is_correct_path = 1 if pred == gt else 0
                results_agg[item_idx][step_idx].append(is_correct_path)

            # Final Scoring and Saving
            for item_idx, item in enumerate(batch_items):
                s_final = []
                pns_details = []
                current_steps = item["steps"][:-1] if "####" in item["steps"][-1] else item["steps"]
                
                for step_idx, step_text in enumerate(current_steps):
                    v_scores = results_agg[item_idx].get(step_idx, [])
                    # PNS calculation: Necessity = 1 - SuccessRate
                    pns_val = 1 - (sum(v_scores) / len(v_scores)) if v_scores else 0.0
                    
                    pns_details.append({"text": step_text, "pns": round(pns_val, 4)})
                    if pns_val >= ALPHA:
                        s_final.append(step_text)
                
                # Update item with expert signals
                item["pns_analysis"] = pns_details
                item["s_final"] = s_final
                out_f.write(json.dumps(item) + "\n")
            
            # Immediate write for live monitoring
            out_f.flush()

if __name__ == "__main__":
    run_gsm8k_sabotage_pns(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)
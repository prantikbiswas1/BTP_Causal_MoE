import json
import os
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct" 
# USE THE MERGED FILE YOU JUST CREATED
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/csqa_raw_multistep_traces.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/csqa_pns_scored.jsonl"

K = 3
BATCH_SIZE = 64 
ALPHA = 0.5

def extract_letter(text):
    """Robustly extracts A-E, ignoring brackets like [A] or markers like answer: A."""
    match = re.search(r"####\s*[^\d\-A-E]*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b[A-E]\b", text)
    return letters[-1].upper() if letters else ""

def run_expert_pns_sabotage(model_path, input_path, output_path):
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=2, 
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90
    )
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load data: We only sabotage correct traces
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f if json.loads(line).get("is_correct")]

    # Sampling for focused rollouts
    sampling_params = SamplingParams(temperature=0.7, max_tokens=80, top_p=0.9) 

    print(f"Generating Sabotage PNS for {len(data)} items...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as out_f:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_sabotage_prompts = []
            prompt_mapping = [] 

            for item_idx, item in enumerate(batch_items):
                # CHANGE: Use the pre-formatted question_with_choices field
                query_block = item.get("question_with_choices", item.get("question", "No question found."))
                
                # We skip the very last step (the one with ####)
                steps = item["steps"][:-1] if "####" in item["steps"][-1] else item["steps"]
                
                for step_idx, st in enumerate(steps):
                    prev_context = "\n".join(steps[:step_idx])
                    
                    # --- THE SABOTAGE PROMPT ---
                    alt_prompt = (
                        f"Question: {query_block}\n"
                        f"Context so far: {prev_context}\n"
                        f"ERROR: The previous deduction '{st}' has been proven WRONG.\n"
                        f"Task: Solve the question again from this point without using that wrong logic. "
                        f"End strictly with #### [Answer Letter]: "
                    )
                    
                    for _ in range(K):
                        all_sabotage_prompts.append(alt_prompt)
                        prompt_mapping.append((item_idx, step_idx))

            if not all_sabotage_prompts:
                continue

            # FIXED: Positional argument for vLLM
            outputs = llm.generate(all_sabotage_prompts, sampling_params, use_tqdm=False)
            
            # Aggregate Results
            results_agg = {idx: {s_idx: [] for s_idx in range(len(batch_items[idx]["steps"]))} 
                           for idx in range(len(batch_items))}
            
            for out, (item_idx, step_idx) in zip(outputs, prompt_mapping):
                pred = extract_letter(out.outputs[0].text)
                # Use ground_truth field from your merged dataset
                gt = str(batch_items[item_idx]["ground_truth"]).strip().upper()
                
                is_correct_path = 1 if pred == gt else 0
                results_agg[item_idx][step_idx].append(is_correct_path)

            # Scoring and Pruning
            for item_idx, item in enumerate(batch_items):
                s_final = []
                pns_details = []
                current_steps = item["steps"][:-1] if "####" in item["steps"][-1] else item["steps"]
                
                for step_idx, step_text in enumerate(current_steps):
                    v_scores = results_agg[item_idx].get(step_idx, [])
                    pns_val = 1 - (sum(v_scores) / len(v_scores)) if v_scores else 0.0
                    
                    pns_details.append({"text": step_text, "pns": round(pns_val, 4)})
                    if pns_val >= ALPHA:
                        s_final.append(step_text)
                
                item["pns_analysis"] = pns_details
                item["s_final"] = s_final
                out_f.write(json.dumps(item) + "\n")
            
            out_f.flush()

if __name__ == "__main__":
    run_expert_pns_sabotage(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)
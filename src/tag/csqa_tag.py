import json
import re
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
# The input is now your already-merged and PNS-scored file
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/csqa_pns_scored.jsonl"
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/csqa_tagged_final.jsonl"

def build_tagged_dataset():
    # 1. Setup vLLM Engine
    print("Initializing vLLM Engine for Expert Tagging...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    # Precise sampling for tags
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    
    # Load only the items that have valid pruned steps
    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f if json.loads(line).get("s_final")]

    print(f"Tagging {len(data)} samples from merged file...")
    BATCH_SIZE = 64 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_prompts = []
            prompt_map = [] 

            for item in batch_items:
                # Use the pre-merged field directly
                q_context = item["question_with_choices"]
                
                for s_idx, step_text in enumerate(item["s_final"]):
                    prompt = (
                        f"System: You are a logic classifier.\n"
                        f"Definitions:\n"
                        f"- LOGIC: Analyzing the question, planning, or final conclusions.\n"
                        f"- COMMONSENSE: General world facts, definitions, and object properties.\n"
                        f"- MATH: Numbers, counting, or arithmetic calculations.\n\n"
                        f"Question Context: {q_context}\n"
                        f"Thought Step: {step_text}\n"
                        f"Classify the Thought as exactly one: LOGIC, COMMONSENSE, or MATH.\n"
                        f"Classification: "
                    )
                    all_prompts.append(prompt)
                    prompt_map.append((item["id"], s_idx, step_text))

            # Batch Inference
            outputs = llm.generate(all_prompts, sampling_params, use_tqdm=False)
            
            # Map tags back to original text
            batch_results = {item["id"]: {} for item in batch_items}
            for out, (item_id, s_idx, original_text) in zip(outputs, prompt_map):
                choice = out.outputs[0].text.strip().upper()
                
                # Expert Tag selection
                tag = "[LOGIC]"
                if "MATH" in choice: tag = "[MATH]"
                elif "COMMONSENSE" in choice: tag = "[COMMONSENSE]"
                
                # CLEANING: Remove "Step X:", "1.", etc. to save tokens
                clean_text = re.sub(r"^Step\s*\d+[:\-\s]*", "", original_text, flags=re.IGNORECASE)
                clean_text = re.sub(r"^\d+[\.\)]\s*", "", clean_text).strip()
                
                batch_results[item_id][s_idx] = f"{tag} {clean_text}"

            # Assemble into the final BTP expert format
            for item in batch_items:
                item_id = item["id"]
                
                # Re-assemble the expert-tagged steps
                sorted_indices = sorted(batch_results[item_id].keys())
                tagged_steps = [batch_results[item_id][idx] for idx in sorted_indices]
                
                # The format your 1.5B Experts will train on
                atomic_trace = " ".join(tagged_steps) + f" [VERIFY] #### {item['ground_truth']}"
                
                # Save as a clean instruction-following pair
                f_out.write(json.dumps({
                    "id": item_id,
                    "instruction": item["question_with_choices"],
                    "atomic_reasoning": atomic_trace,
                    "answer": item["ground_truth"]
                }) + "\n")
            
            f_out.flush() 

    print(f"Tagging complete. Gold dataset at: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_tagged_dataset()
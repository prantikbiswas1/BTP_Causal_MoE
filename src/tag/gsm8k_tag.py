import json
import re
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/gsm8k_pns_scored.jsonl"
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/gsm8k_tagged_final.jsonl"

def build_gsm8k_tagged_dataset():
    # 1. Setup vLLM Engine
    print("Initializing vLLM Engine for GSM8K Expert Tagging...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    # Greedy sampling for classification
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    
    # Load only items with valid s_final steps
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f if json.loads(line).get("s_final")]

    print(f"Tagging {len(data)} GSM8K samples with Math-Priority Logic...")
    BATCH_SIZE = 64 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_prompts = []
            prompt_map = [] 

            for item in batch_items:
                q_context = item["question"]
                
                for s_idx, step_text in enumerate(item["s_final"]):
                    # ENHANCED PROMPT: Explicit priority for MATH to avoid LOGIC-bias
                    prompt = (
                        f"System: You are an expert math and logic classifier.\n"
                        f"Priority Rule: If a thought contains ANY calculation (e.g., 5+5=10) or numeric derivation, it MUST be MATH.\n\n"
                        f"Definitions:\n"
                        f"- MATH: Equations, arithmetic calculations, or numeric operations.\n"
                        f"- LOGIC: Analyzing the problem, planning a strategy, or stating facts from the prompt without calculations.\n"
                        f"- COMMONSENSE: General world facts (e.g., 'There are 12 months in a year').\n\n"
                        f"Question: {q_context}\n"
                        f"Thought Step: {step_text}\n"
                        f"Classify as exactly one: LOGIC, COMMONSENSE, or MATH.\n"
                        f"Classification: "
                    )
                    all_prompts.append(prompt)
                    prompt_map.append((item["id"], s_idx, step_text))

            # Batch Inference
            outputs = llm.generate(all_prompts, sampling_params, use_tqdm=False)
            
            # Map tags back with Python Fallback Override
            batch_results = {item["id"]: {} for item in batch_items}
            for out, (item_id, s_idx, original_text) in zip(outputs, prompt_map):
                choice = out.outputs[0].text.strip().upper()
                
                # 1. Start with the model's classification
                tag = "[LOGIC]"
                if "MATH" in choice: 
                    tag = "[MATH]"
                elif "COMMONSENSE" in choice: 
                    tag = "[COMMONSENSE]"
                
                # 2. PYTHON FALLBACK: If the model is lazy but the step contains math operators
                # (Check for =, +, *, / and ensure there are actual digits)
                if tag == "[LOGIC]" and any(op in original_text for op in ["=", "+", "/", "*"]):
                    if re.search(r'\d', original_text):
                        tag = "[MATH]"
                
                # CLEANING: Remove "Step X:", "1.", etc.
                clean_text = re.sub(r"^Step\s*\d+[:\-\s]*", "", original_text, flags=re.IGNORECASE)
                clean_text = re.sub(r"^\d+[\.\)]\s*", "", clean_text).strip()
                
                batch_results[item_id][s_idx] = f"{tag} {clean_text}"

            # Final Assembly for the 1.5B Training Loop
            for item in batch_items:
                item_id = item["id"]
                sorted_indices = sorted(batch_results[item_id].keys())
                tagged_steps = [batch_results[item_id][idx] for idx in sorted_indices]
                
                # Atomic reasoning for GSM8K
                atomic_trace = " ".join(tagged_steps) + f" [VERIFY] #### {item['ground_truth']}"
                
                f_out.write(json.dumps({
                    "id": item_id,
                    "instruction": item["question"],
                    "atomic_reasoning": atomic_trace,
                    "answer": item["ground_truth"]
                }) + "\n")
            
            f_out.flush() 

    print(f"Tagging complete. GSM8K Gold dataset saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Standard cluster cleanup
    # os.system("ray stop; pkill -f vllm")
    build_gsm8k_tagged_dataset()
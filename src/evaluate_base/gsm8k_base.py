import json
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ==========================================
# PATHS
# ==========================================
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_base/gsm8k_base_results.jsonl"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ==========================================
# INIT MODEL
# ==========================================
print(f"🚀 Initializing Base Model...")
llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.90, enforce_eager=True)

# ==========================================
# ROBUST UTILS
# ==========================================
def extract_numeric_value(text):
    """Extracts the very last numerical value from a string, handling diverse formats."""
    if not text: return None
    # Prioritize the specific format we asked for
    match = re.findall(r"###\s*\[?(-?[\d,.]+)\]?", str(text))
    if not match:
        # Fallback to standard GSM8K markers
        match = re.findall(r"####\s*(-?[\d,.]+)", str(text))
    if not match:
        # Final fallback: just get the last number in the string
        match = re.findall(r"(-?[\d,.]+)", str(text))
    
    if match:
        try:
            val_str = match[-1].replace(",", "").strip().rstrip('.')
            return float(val_str)
        except:
            return None
    return None

def get_token_count(text):
    return len(str(text).split()) * 1.3

# ==========================================
# RUN EVALUATION
# ==========================================
def run_base():
    data = [json.loads(l) for l in open(DATA_PATH)]
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=512, 
        stop=["### Instruction:", "Question:"] 
    )

    with open(OUT_PATH, "w") as f:
        for i, d in enumerate(tqdm(data)):
            q = d.get("question", "")
            
            # Extract ground truth as a float
            gt_raw = str(d.get("answer", ""))
            gt_val = extract_numeric_value(gt_raw)

            # Modified Prompt for better compliance
            prompt = (
                "### Instruction:\n"
                "Solve this math problem. Show your work.\n"
                "You MUST end your response by stating the final number like this: '### [number]'.\n\n"
                "### Input:\n"
                f"Question: {q}\n\n"
                "### Response:\n"
            )

            # Generate
            output = llm.generate([prompt], sampling_params, use_tqdm=False)
            raw_response = output[0].outputs[0].text.strip()
            
            # Extract prediction as a float
            pred_val = extract_numeric_value(raw_response)

            # MATHEMATICAL TOLERANCE CHECK (Fixed ID 14, 24, etc.)
            is_correct = False
            if pred_val is not None and gt_val is not None:
                is_correct = abs(pred_val - gt_val) < 1e-4

            # Metrics
            in_tokens = get_token_count(prompt)
            out_tokens = get_token_count(raw_response)
            total_flops = 2 * (7.0 * 1e9) * (in_tokens + out_tokens)

            entry = {
                "id": i,
                "is_correct": is_correct,
                "metrics": {
                    "router_tokens": 0,
                    "expert_in_tokens": in_tokens,
                    "expert_out_tokens": out_tokens,
                    "total_flops": total_flops,
                    "steps": 1,
                    "total_tokens": in_tokens + out_tokens
                },
                "prediction_text": raw_response,
                "prediction_val": pred_val,
                "ground_truth_val": gt_val
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    run_base()
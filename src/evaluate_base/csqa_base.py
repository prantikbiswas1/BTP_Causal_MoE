import json
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_TYPE = "csqa" # Options: "gsm8k" or "csqa"
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

if DATASET_TYPE == "gsm8k":
    DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
    OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_base/gsm8k_base_results.jsonl"
else:
    DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_val.jsonl"
    OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_base/csqa_base_results.jsonl"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ==========================================
# INIT MODEL
# ==========================================
print(f"🚀 Initializing Base Model for {DATASET_TYPE.upper()}...")
llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.90, enforce_eager=True)

# ==========================================
# ROBUST UTILS
# ==========================================
def extract_val(text, is_csqa=False):
    if not text: return None
    if is_csqa:
        # CSQA: Look for Letter A-E
        match = re.findall(r"(?:###|####|Answer:|Final Answer:)\s*([A-E])", str(text))
        if not match: match = re.findall(r"\[([A-E])\]", str(text))
        if not match: match = re.findall(r"\b([A-E])\b", str(text))
        return match[-1] if match else None
    else:
        # GSM8K: Look for Number
        match = re.findall(r"###\s*\[?(-?[\d,.]+)\]?", str(text))
        if not match: match = re.findall(r"####\s*(-?[\d,.]+)", str(text))
        if not match: match = re.findall(r"(-?[\d,.]+)", str(text))
        
        if match:
            try:
                return float(match[-1].replace(",", "").strip().rstrip('.'))
            except:
                return None
    return None

def get_token_count(text):
    return len(str(text).split()) * 1.3

def format_options(choices):
    if not choices: return ""
    return " Options: " + ", ".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])

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
            choices = d.get("choices", None)
            is_csqa = (DATASET_TYPE == "csqa")
            
            # Ground Truth Prep
            gt_raw = str(d.get("answerKey", d.get("answer", "")))
            gt_val = extract_val(gt_raw, is_csqa=is_csqa)

            # Unified Prompt
            if is_csqa:
                opt_str = format_options(choices)
                prompt = (
                    "### Instruction:\n"
                    "Select the correct option letter (A, B, C, D, or E) for the following question.\n"
                    "End your response with the final answer exactly in this format: '### [Letter]'.\n\n"
                    "### Input:\n"
                    f"Question: {q}{opt_str}\n\n"
                    "### Response:\n"
                )
            else:
                prompt = (
                    "### Instruction:\n"
                    "Solve this math problem. Show your work.\n"
                    "End your response with the final numerical answer exactly in this format: '### [number]'.\n\n"
                    "### Input:\n"
                    f"Question: {q}\n\n"
                    "### Response:\n"
                )

            # Generate
            output = llm.generate([prompt], sampling_params, use_tqdm=False)
            raw_response = output[0].outputs[0].text.strip()
            
            # Extract prediction
            pred_val = extract_val(raw_response, is_csqa=is_csqa)

            # Comparison
            is_correct = False
            if pred_val is not None and gt_val is not None:
                if is_csqa:
                    is_correct = (str(pred_val) == str(gt_val))
                else:
                    is_correct = abs(pred_val - gt_val) < 1e-4

            # Metrics Calculation
            in_tokens = get_token_count(prompt)
            out_tokens = get_token_count(raw_response)
            # Base Model is 7B
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
import json
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# CONFIG & PATHS
# ==========================================
ROUTER_MODEL = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/router/router_qwen_1.5b_merged_bf16"
EXPERT_BASE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

LORA_PATHS = {
    "LOGIC": "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/logic_expert_qwen_7b",
    "COMMONSENSE": "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/commonsense_expert_qwen_7b",
    "VERIFY": "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/verify_expert_qwen_7b",
    "MATH": "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/math_expert_qwen_7b"
}

DATASET_TYPE = "gsm8k" 
DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_moe/gsm8k_results.jsonl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Hardened Constants
MIN_STEPS = 3      
MAX_STEPS = 12     
DEBUG_SAMPLES = 5

# ==========================================
# INIT
# ==========================================
print(f"🚀 Initializing Precision-Hardened MoE for {DATASET_TYPE}...")
router = LLM(model=ROUTER_MODEL, dtype="bfloat16", gpu_memory_utilization=0.20, enforce_eager=True)
expert_llm = LLM(model=EXPERT_BASE, dtype="bfloat16", enable_lora=True, max_loras=4, gpu_memory_utilization=0.70, enforce_eager=True)

lora_requests = {
    "LOGIC": LoRARequest("logic", 2, LORA_PATHS["LOGIC"]),
    "COMMONSENSE": LoRARequest("common", 3, LORA_PATHS["COMMONSENSE"]),
    "VERIFY": LoRARequest("verify", 4, LORA_PATHS["VERIFY"]),
    "MATH": LoRARequest("math", 5, LORA_PATHS["MATH"])
}

# ==========================================
# HELPERS
# ==========================================
def extract_value(text):
    if text is None: return None
    # Look for #### X first
    match = re.findall(r"####\s*(-?[\d,.]+)", str(text))
    if not match:
        # Fallback to the very last number found in the text
        match = re.findall(r"(-?[\d,.]+)", str(text))
    
    if match:
        try:
            # Clean commas and handle float conversion
            val_str = match[-1].replace(",", "").strip().rstrip('.')
            return float(val_str)
        except ValueError:
            return None
    return None

def build_expert_prompt(tag, question, full_history):
    instrs = {
        "LOGIC": "Define the next required calculation step based on the question. Do NOT solve.",
        "MATH": "Perform the calculation from the previous LOGIC step. Output equation and result. Include units.",
        "COMMONSENSE": "State one specific world fact or number from the question needed now.",
        "VERIFY": "Reasoning is complete. Provide the final numerical answer to the original question. Format: #### X"
    }
    return (
        "### Instruction:\n"
        f"Goal: {question}\n"
        f"Task: {instrs[tag]}\n\n"
        "### Input:\n"
        f"Reasoning History:\n{full_history if full_history else 'None'}\n\n"
        "### Response:\n"
        f"[{tag}]"
    )

# ==========================================
# SOLVER
# ==========================================
def solve(question, debug=False):
    full_history = "" 
    router_trace = "" 
    trace_list = []
    content_list = [] # Used for deduplication check
    
    m = {"router_tokens": 0, "expert_in_tokens": 0, "expert_out_tokens": 0, "steps": 0}

    for step_idx in range(MAX_STEPS):
        # 1. ROUTER - Predict Tag
        r_prompt = (
            "### Instruction:\n"
            f"Question: {question}\n"
            "Predict the NEXT tag: MATH, LOGIC, COMMONSENSE, VERIFY.\n\n"
            f"### Progress:\n{router_trace if router_trace else 'New Problem'}\n\n"
            "### Response:\n"
        )
        r_out = router.generate([r_prompt], SamplingParams(temperature=0.0, max_tokens=10), use_tqdm=False)
        tag_raw = r_out[0].outputs[0].text.strip().upper()
        
        current_tag = "LOGIC"
        for t in ["MATH", "LOGIC", "COMMONSENSE", "VERIFY"]:
            if t in tag_raw:
                current_tag = t
                break
        
        # Override 1: Loop Breaking (If expert repeats exactly, force verify)
        if len(content_list) > 1 and content_list[-1] == content_list[-2]:
            current_tag = "VERIFY"
            
        # Override 2: Anti-Premature Termination
        if current_tag == "VERIFY" and step_idx < MIN_STEPS:
            current_tag = "MATH"

        # 2. EXPERT - Generate Step
        e_prompt = build_expert_prompt(current_tag, question, full_history)
        e_out = expert_llm.generate(
            [e_prompt],
            SamplingParams(temperature=0.0, max_tokens=128, stop=["[", "###"]),
            lora_request=lora_requests[current_tag],
            use_tqdm=False
        )
        
        step_content = e_out[0].outputs[0].text.strip().replace(f"[{current_tag}]", "").strip()

        # Log State
        full_history += f"[{current_tag}] {step_content}\n"
        router_trace += f"[{current_tag}] {step_content}\n"
        trace_list.append(f"[{current_tag}] {step_content}")
        content_list.append(step_content)
        
        # Metrics
        m["router_tokens"] += len(r_prompt.split())
        m["expert_in_tokens"] += len(e_prompt.split())
        m["expert_out_tokens"] += len(step_content.split())

        if debug: print(f"  Step {step_idx+1} | [{current_tag}] {step_content}")
        if current_tag == "VERIFY" or "####" in step_content:
            break

    # 3. ATOMIC EXTRACTION (Cleanup)
    if extract_value(full_history) is None:
        cleanup_prompt = (
            f"### Instruction: Extract the final number from the history: {full_history}\n\n"
            "### Response:\n#### "
        )
        e_out = expert_llm.generate(
            [cleanup_prompt], 
            SamplingParams(temperature=0.0, max_tokens=20),
            lora_request=lora_requests["VERIFY"], 
            use_tqdm=False
        )
        final_val_str = e_out[0].outputs[0].text.strip()
        full_history += f"\n#### {final_val_str}"
        trace_list.append(f"[CLEANUP] #### {final_val_str}")

    m["steps"] = len(trace_list)
    return {"final": full_history, "trace": trace_list, "metrics": m}

# ==========================================
# EVAL LOOP
# ==========================================
def run():
    print(f"📊 Starting Value-Based Evaluation...")
    data = [json.loads(l) for l in open(DATA_PATH)]
    with open(OUT_PATH, "w") as f:
        for i, d in enumerate(tqdm(data)):
            q = d.get("question", "")
            res = solve(q, debug=(i < DEBUG_SAMPLES))
            
            # 🔥 FLOAT COMPARISON LOGIC
            pred_val = extract_value(res["final"])
            gt_text = str(d.get("answer", d.get("answerKey", "")))
            gt_val = extract_value(gt_text)
            
            is_correct = False
            if pred_val is not None and gt_val is not None:
                # Use tolerance for floating point precision issues (like ID 14)
                is_correct = abs(pred_val - gt_val) < 1e-4
            
            entry = {
                "id": i, 
                "question": q,                               
                "ground_truth": gt_val,              
                "prediction": pred_val,            
                "is_correct": is_correct,      
                "metrics": res["metrics"],                 
                "trace": res["trace"]                      
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    run()
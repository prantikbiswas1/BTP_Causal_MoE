import json
import time
import re
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# PATHS
# ==========================================
ROUTER_MODEL = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/router/router_qwen_1.5b_merged_bf16"
EXPERT_BASE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

LOGIC_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/logic_expert_qwen_7b/checkpoint-800"
COMMON_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/commonsense_expert_qwen_7b/checkpoint-800"
VERIFY_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/verify_expert_qwen_7b/checkpoint-800"
MATH_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/math_expert_qwen_7b/checkpoint-800"

GSM8K_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_moe/gsm8k_results.jsonl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Config
# Removed MIN_STEPS constraint for GSM8K to prevent "forced" hallucinated math steps
MAX_STEPS = 8 
DEBUG_SAMPLES = 3

# ==========================================
# INIT MODELS
# ==========================================
print("🚀 Initializing Optimized GSM8K MoE Pipeline...")
router = LLM(model=ROUTER_MODEL, dtype="bfloat16", gpu_memory_utilization=0.20, max_model_len=4096, trust_remote_code=True, enforce_eager=True)
expert_llm = LLM(model=EXPERT_BASE, dtype="bfloat16", enable_lora=True, max_loras=4, gpu_memory_utilization=0.70, max_model_len=4096, trust_remote_code=True, enforce_eager=True)

logic_lora  = LoRARequest("logic", 2, LOGIC_LORA_PATH)
common_lora = LoRARequest("common", 3, COMMON_LORA_PATH)
verify_lora = LoRARequest("verify", 4, VERIFY_LORA_PATH)
math_lora   = LoRARequest("math", 5, MATH_LORA_PATH)

lora_map = {"LOGIC": logic_lora, "COMMONSENSE": common_lora, "VERIFY": verify_lora, "MATH": math_lora}

# ==========================================
# UTILS
# ==========================================
def get_token_count(text):
    return len(text.split()) * 1.3

def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return None

def format_options(choices):
    return " ".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])

# ==========================================
# PROMPT BUILDERS
# ==========================================
def build_router_prompt(question, current_trace_clean):
    return (
        "### Instruction:\n"
        "Predict the NEXT required reasoning step.\n"
        "Choose the MOST appropriate tag:\n"
        "- MATH: numerical calculation\n"
        "- LOGIC: deductive reasoning\n"
        "- COMMONSENSE: world knowledge\n"
        "- VERIFY: final answer is ready\n\n"
        "### Input:\n"
        f"Question: {question}\n"
        f"Context: {current_trace_clean if current_trace_clean else ''}\n\n"
        "### Response:\n"
    )

def build_expert_prompt(tag, question, expert_trace):
    instructions = {
        "LOGIC": "You are a logical reasoning expert.\nProvide ONLY the next logical step.\nUse a short structured format.\nDo NOT explain.",
        "COMMONSENSE": "You are a commonsense reasoning expert.\nState the relevant world fact or axiom.\nDo NOT explain.",
        "MATH": "You are a mathematical reasoning expert.\nPerform the next calculation step.\nOutput ONLY the computation.",
        "VERIFY": "You are a verification expert.\nSelect the correct final answer based on the reasoning.\nOutput ONLY the answer in format: [VERIFY] #### X"
    }
    return (
        "### Instruction:\n"
        f"{instructions[tag]}\n\n"
        "### Input:\n"
        f"Question: {question}\n"
        f"Context: {expert_trace if expert_trace else 'None'}\n\n"
        "### Response:\n"
        f"[{tag}]"
    )

# ==========================================
# SOLVER
# ==========================================
def solve(question, is_csqa=False, debug=False):
    expert_trace = "" 
    router_trace = "" 
    trace_list = []    
    verified = False

    m = {"router_tokens": 0, "expert_in_tokens": 0, "expert_out_tokens": 0, "total_flops": 0, "steps": 0}

    for step_idx in range(MAX_STEPS):
        # 1. ROUTER PHASE
        r_prompt = build_router_prompt(question, router_trace)
        r_out = router.generate([r_prompt], SamplingParams(temperature=0.7, max_tokens=15), use_tqdm=False)
        tag_res = r_out[0].outputs[0].text.strip().upper()

        tag = "VERIFY"
        for t in ["MATH", "LOGIC", "COMMONSENSE", "VERIFY"]:
            if t in tag_res:
                tag = t
                break

        # Removed the forced MIN_STEPS for GSM8K to allow faster verification on simple problems

        m["router_tokens"] += get_token_count(r_prompt)
        m["total_flops"] += 2 * (1.5 * 1e9) * get_token_count(r_prompt)

        # 2. EXPERT PHASE
        e_prompt = build_expert_prompt(tag, question, expert_trace)
        
        # Optimization: Deterministic temp for MATH and VERIFY
        if tag in ["MATH", "VERIFY"]:
            e_temp = 0.0 
        else:
            e_temp = 0.5 # Slightly lower than 0.7 for better stability
        
        e_out = expert_llm.generate(
            [e_prompt], 
            SamplingParams(temperature=e_temp, max_tokens=128), # Increased tokens to allow full calculations
            lora_request=lora_map[tag],
            use_tqdm=False
        )
        
        expert_raw = e_out[0].outputs[0].text.strip().split("\n")[0]
        clean_content = expert_raw.replace("[", "").replace("]", "").replace(tag, "").replace(":", "").strip()

        # Repetition Breaker
        if len(trace_list) > 0:
            last_clean = trace_list[-1].split("]", 1)[-1].strip().lower()
            if clean_content.lower() == last_clean:
                tag = "VERIFY"
                e_prompt = build_expert_prompt(tag, question, expert_trace)
                e_out = expert_llm.generate([e_prompt], SamplingParams(temperature=0.0, max_tokens=64), lora_request=lora_map[tag], use_tqdm=False)
                expert_raw = e_out[0].outputs[0].text.strip()
                clean_content = expert_raw.replace("[", "").replace("]", "").replace(tag, "").replace(":", "").strip()

        if tag == "VERIFY":
            ans = extract_answer(clean_content)
            formatted_expert = f"[{tag}] #### {ans if ans else '0'}"
            formatted_router = f"  {tag} #### {ans if ans else '0'}"
            verified = True
        else:
            formatted_expert = f"[{tag}] {clean_content}"
            formatted_router = f"  {tag} {clean_content}"

        expert_trace += (" " if expert_trace else "") + formatted_expert
        router_trace += formatted_router 
        trace_list.append(formatted_expert)

        m["expert_in_tokens"] += get_token_count(e_prompt)
        m["expert_out_tokens"] += get_token_count(expert_raw)
        m["total_flops"] += 2 * (7.0 * 1e9) * (get_token_count(e_prompt) + get_token_count(expert_raw))

        if debug:
            print(f"  Step {step_idx+1} | {tag} | {clean_content}")

        if verified:
            break

    if not verified:
        tag = "VERIFY"
        e_prompt = build_expert_prompt(tag, question, expert_trace)
        e_out = expert_llm.generate([e_prompt], SamplingParams(temperature=0.0, max_tokens=48), lora_request=lora_map[tag], use_tqdm=False)
        ans = extract_answer(e_out[0].outputs[0].text.strip())
        trace_list.append(f"[{tag}] #### {ans if ans else '0'}")

    m["total_tokens"] = m["router_tokens"] + m["expert_in_tokens"] + m["expert_out_tokens"]
    m["steps"] = len(trace_list)

    return {"final": trace_list[-1], "trace": trace_list, "metrics": m}

# ==========================================
# RUN
# ==========================================
def run():
    print(f"\n📊 Evaluating Optimized GSM8K Loop...")
    data = [json.loads(l) for l in open(GSM8K_PATH)]
    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: continue

    with open(OUT_PATH, "a") as f:
        for i, d in enumerate(tqdm(data)):
            if i in done_ids: continue
            
            q = d["question"]
            gt_answer = extract_answer(d["answer"])
            
            res = solve(q, is_csqa=False, debug=(i < DEBUG_SAMPLES))
            pred_answer = extract_answer(res["final"])
            
            entry = {
                "id": i, 
                "is_correct": (str(pred_answer) == str(gt_answer)), 
                "metrics": res["metrics"], 
                "prediction": res["final"], 
                "ground_truth": gt_answer, 
                "trace": res["trace"]
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    run()
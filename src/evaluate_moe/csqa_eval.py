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

CSQA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_val.jsonl"
OUT_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_moe/csqa_results.jsonl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Config
MIN_STEPS = 2  
MAX_STEPS = 8
DEBUG_SAMPLES = 3

# ==========================================
# INIT MODELS
# ==========================================
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

def extract_choice(text):
    match = re.search(r"####\s*([A-E])", text)
    if match: return match.group(1)
    letters = re.findall(r"\b[A-E]\b", text)
    return letters[-1] if letters else None

def format_options(choices):
    return " ".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])

# ==========================================
# PROMPT BUILDERS
# ==========================================
def build_router_prompt(question, options, current_trace_clean):
    # Matches router_train_exploded.jsonl: Input is Question + Options + Double-spaced Context
    return (
        "### Instruction:\n"
        "Predict the NEXT required reasoning step.\n"
        "Choose the MOST appropriate tag:\n"
        "- MATH: numerical calculation\n"
        "- LOGIC: deductive reasoning\n"
        "- COMMONSENSE: world knowledge\n"
        "- VERIFY: final answer is ready\n\n"
        "### Input:\n"
        f"Question: {question} Options: {options}\n"
        f"Context: {current_trace_clean if current_trace_clean else ''}\n\n"
        "### Response:\n"
    )

def build_expert_prompt(tag, question, options, expert_trace):
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
        f"Question: {question} Options: {options}\n"
        f"Context: {expert_trace if expert_trace else 'None'}\n\n"
        "### Response:\n"
        f"[{tag}]"
    )

# ==========================================
# SOLVER
# ==========================================
def solve(question, choices, is_csqa=True, debug=False):
    expert_trace = "" # For Experts: [TAG] content
    router_trace = "" # For Router: TAG content (Stripped of brackets, DOUBLE SPACED)
    trace_list = []    
    verified = False
    options_str = format_options(choices)

    m = {"router_tokens": 0, "expert_in_tokens": 0, "expert_out_tokens": 0, "total_flops": 0, "steps": 0}

    for step_idx in range(MAX_STEPS):
        # 1. ROUTER PHASE
        r_prompt = build_router_prompt(question, options_str, router_trace)
        r_out = router.generate([r_prompt], SamplingParams(temperature=0.7, max_tokens=15), use_tqdm=False)
        tag_res = r_out[0].outputs[0].text.strip().upper()

        tag = "VERIFY"
        for t in ["MATH", "LOGIC", "COMMONSENSE", "VERIFY"]:
            if t in tag_res:
                tag = t
                break

        # 🔥 FIX 1: Over-ride MATH for CSQA dataset
        if is_csqa and tag == "MATH":
            tag = "LOGIC"

        # 🔥 FIX 2: Over-ride early VERIFY
        if step_idx < MIN_STEPS and tag == "VERIFY":
            tag = "COMMONSENSE" if step_idx == 0 else "LOGIC"

        m["router_tokens"] += get_token_count(r_prompt)
        m["total_flops"] += 2 * (1.5 * 1e9) * get_token_count(r_prompt)

        # 2. EXPERT PHASE
        e_prompt = build_expert_prompt(tag, question, options_str, expert_trace)
        e_temp = 0.2 if tag == "VERIFY" else 0.7
        
        e_out = expert_llm.generate(
            [e_prompt], 
            SamplingParams(temperature=e_temp, max_tokens=80),
            lora_request=lora_map[tag],
            use_tqdm=False
        )
        
        expert_raw = e_out[0].outputs[0].text.strip().split("\n")[0]
        clean_content = expert_raw.replace("[", "").replace("]", "").replace(tag, "").replace(":", "").strip()

        # 🔥 FIX 3: Repetition Breaker
        if len(trace_list) > 0:
            last_clean = trace_list[-1].split("]", 1)[-1].strip().lower()
            if clean_content.lower() == last_clean:
                # If repeating, force verify immediately
                tag = "VERIFY"
                e_prompt = build_expert_prompt(tag, question, options_str, expert_trace)
                e_out = expert_llm.generate([e_prompt], SamplingParams(temperature=0.0, max_tokens=32), lora_request=lora_map[tag], use_tqdm=False)
                expert_raw = e_out[0].outputs[0].text.strip()
                clean_content = expert_raw.replace("[", "").replace("]", "").replace(tag, "").replace(":", "").strip()

        if tag == "VERIFY":
            choice = extract_choice(clean_content)
            formatted_expert = f"[{tag}] #### {choice if choice else 'A'}"
            # 🔥 FIX 4: Double spacing for Router context alignment
            formatted_router = f"  {tag} #### {choice if choice else 'A'}"
            verified = True
        else:
            formatted_expert = f"[{tag}] {clean_content}"
            # 🔥 FIX 4: Double spacing for Router context alignment
            formatted_router = f"  {tag} {clean_content}"

        expert_trace += (" " if expert_trace else "") + formatted_expert
        router_trace += formatted_router # Note: No leading space here, formatted_router has 2 spaces
        trace_list.append(formatted_expert)

        m["expert_in_tokens"] += get_token_count(e_prompt)
        m["expert_out_tokens"] += get_token_count(expert_raw)
        m["total_flops"] += 2 * (7.0 * 1e9) * (get_token_count(e_prompt) + get_token_count(expert_raw))

        if debug:
            print(f"  Step {step_idx+1} | {formatted_expert}")

        if verified:
            break

    # Safety Trigger
    if not verified:
        tag = "VERIFY"
        e_prompt = build_expert_prompt(tag, question, options_str, expert_trace)
        e_out = expert_llm.generate([e_prompt], SamplingParams(temperature=0.0, max_tokens=32), lora_request=lora_map[tag], use_tqdm=False)
        choice = extract_choice(e_out[0].outputs[0].text.strip())
        trace_list.append(f"[{tag}] #### {choice if choice else 'A'}")

    m["total_tokens"] = m["router_tokens"] + m["expert_in_tokens"] + m["expert_out_tokens"]
    m["steps"] = len(trace_list)

    return {"final": trace_list[-1], "trace": trace_list, "metrics": m}

# ==========================================
# RUN
# ==========================================
def run():
    print(f"\n📊 Evaluating CSQA via Causal MoE Loop...")
    data = [json.loads(l) for l in open(CSQA_PATH)]
    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: continue

    with open(OUT_PATH, "a") as f:
        for i, d in enumerate(tqdm(data)):
            if i in done_ids: continue
            # Pass is_csqa=True to trigger the MATH blocking logic
            res = solve(d["question"], d["choices"], is_csqa=True, debug=(i < DEBUG_SAMPLES))
            pred = extract_choice(res["final"])
            entry = {"id": i, "is_correct": (pred == d["answerKey"]), "metrics": res["metrics"], "prediction": res["final"], "ground_truth": d["answerKey"], "trace": res["trace"]}
            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    run()
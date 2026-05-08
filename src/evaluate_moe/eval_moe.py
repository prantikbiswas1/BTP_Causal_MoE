import json
import time
import re
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# PATHS - CONFIGURATION
# ==========================================
ROUTER_MODEL = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/router/router_qwen_1.5b_merged_bf16"
EXPERT_BASE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

MATH_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/math_expert_qwen_7b/checkpoint-800"
LOGIC_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/logic_expert_qwen_7b/checkpoint-800"
COMMON_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/commonsense_expert_qwen_7b/checkpoint-800"
VERIFY_LORA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/verify_expert_qwen_7b/checkpoint-800"

GSM8K_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
CSQA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_val.jsonl"
OUT_DIR = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_moe"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_STEPS = 8
DEBUG_SAMPLES = 3

# ==========================================
# INITIALIZE VLLM
# ==========================================
print("🚀 Initializing Multi-LoRA MoE System...")

router = LLM(
    model=ROUTER_MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    trust_remote_code=True,
    enforce_eager=True
)

expert_llm = LLM(
    model=EXPERT_BASE,
    dtype="bfloat16",
    enable_lora=True,
    max_loras=4,
    gpu_memory_utilization=0.70,
    max_model_len=4096,
    trust_remote_code=True,
    enforce_eager=True
)

math_lora   = LoRARequest("math", 1, MATH_LORA_PATH)
logic_lora  = LoRARequest("logic", 2, LOGIC_LORA_PATH)
common_lora = LoRARequest("common", 3, COMMON_LORA_PATH)
verify_lora = LoRARequest("verify", 4, VERIFY_LORA_PATH)

lora_map = {"MATH": math_lora, "LOGIC": logic_lora, "COMMONSENSE": common_lora, "VERIFY": verify_lora}

# ==========================================
# METRICS & UTILS
# ==========================================
def get_token_count(text):
    return len(text.split()) * 1.3

def extract_number(text):
    if not text or text == "FAILED": return None
    match = re.search(r"####\s*(-?\d+)", text)
    if match: return int(match.group(1))
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def extract_choice(text):
    if not text or text == "FAILED": return None
    match = re.search(r"(?:Choice|####|answer is)\s*[:\s]*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    letters = re.findall(r"\b[A-E]\b", text)
    return letters[-1] if letters else None

# ==========================================
# PROMPT BUILDERS
# ==========================================
def build_router_prompt(context):
    clean_context = context.replace("[", "").replace("]", "")
    return (
        "### Instruction:\nPredict the NEXT required reasoning step.\n"
        "Choose the MOST appropriate tag:\n"
        "- MATH: numerical calculation\n- LOGIC: deductive reasoning\n"
        "- COMMONSENSE: world knowledge\n- VERIFY: final answer is ready\n\n"
        f"### Input:\n{clean_context}\n\n### Response:\n"
    )

def build_expert_prompt(tag, question, context):
    prompts = {
        "MATH": "You are a mathematical reasoning expert.\nPerform the next calculation step.\nOutput ONLY the computation.",
        "LOGIC": "You are a logical reasoning expert.\nProvide ONLY the next logical step.\nUse a short structured format.\nDo NOT explain.",
        "COMMONSENSE": "You are a commonsense reasoning expert.\nState the relevant world fact or axiom.\nDo NOT explain.",
        "VERIFY": "You are a verification expert.\nSelect the correct final answer based on the reasoning.\nOutput ONLY the answer in format: [VERIFY] #### X"
    }
    instr = prompts.get(tag, prompts["VERIFY"])
    return f"### Instruction:\n{instr}\n\n### Input:\nQuestion: {question}\nContext: {context}\n\n### Response:\n"

# ==========================================
# INFERENCE ENGINE
# ==========================================
def solve(question, debug=False):
    context, trace, tag_history = "", [], []
    m = {
        "router_tokens": 0, "expert_in_tokens": 0, "expert_out_tokens": 0,
        "total_tokens": 0, "total_flops": 0, "steps": 0
    }

    for step_idx in range(MAX_STEPS):
        r_prompt = build_router_prompt(context)
        outputs = router.generate([r_prompt], SamplingParams(temperature=0.0, max_tokens=10), use_tqdm=False)
        tag_res = outputs[0].outputs[0].text.strip().upper().replace(" ", "")
        
        tag = "VERIFY"
        for t in ["MATH", "LOGIC", "COMMONSENSE", "VERIFY"]:
            if t in tag_res:
                tag = t
                break

        r_in = get_token_count(r_prompt)
        m["router_tokens"] += (r_in + 2)
        m["total_flops"] += 2 * (1.5 * 1e9) * (r_in + 2)

        tag_history.append(tag)
        if len(tag_history) >= 4 and len(set(tag_history[-4:])) == 1:
            tag = "VERIFY"
        
        e_prompt = build_expert_prompt(tag, question, context)

        # ✅ FIX: increase VERIFY tokens
        max_toks = 32 if tag == "VERIFY" else 64

        e_outputs = expert_llm.generate(
            [e_prompt],
            SamplingParams(temperature=0.0, max_tokens=max_toks),
            lora_request=lora_map[tag],
            use_tqdm=False
        )
        expert_out = e_outputs[0].outputs[0].text.strip().split("\n")[0]

        e_in = get_token_count(e_prompt)
        e_out = get_token_count(expert_out)
        m["expert_in_tokens"] += e_in
        m["expert_out_tokens"] += e_out
        m["total_flops"] += 2 * (7.0 * 1e9) * (e_in + e_out)

        if len(trace) > 0 and expert_out.strip() == trace[-1].strip():
            if tag != "VERIFY":
                e_outputs = expert_llm.generate(
                    [build_expert_prompt("VERIFY", question, context)],
                    SamplingParams(temperature=0.0, max_tokens=32),
                    lora_request=verify_lora,
                    use_tqdm=False
                )
                expert_out = e_outputs[0].outputs[0].text.strip().split("\n")[0]
            break

        # ==========================================
        # ✅ FIX: VERIFY uses latest number reliably
        # ==========================================
        if tag == "VERIFY":
            combined = context + " " + expert_out
            nums = re.findall(r"-?\d+", combined)
            if nums:
                expert_out = f"[VERIFY] #### {nums[-1]}"
            else:
                expert_out = "[VERIFY] #### 0"

        elif f"[{tag}]" not in expert_out:
            expert_out = f"[{tag}] {expert_out}"

        context += " " + expert_out
        trace.append(expert_out)

        if debug:
            print(f"🔀 Step {step_idx+1} [{tag}]: {expert_out}")

        if tag == "VERIFY":
            break

    m["total_tokens"] = m["router_tokens"] + m["expert_in_tokens"] + m["expert_out_tokens"]
    m["steps"] = len(trace)

    return {
        "final": trace[-1] if trace else "FAILED",
        "trace": trace,
        "metrics": m
    }

# ==========================================
# EVALUATION WRAPPER
# ==========================================
def run_evaluation(name, in_path, out_file, extract_fn):
    out_path = os.path.join(OUT_DIR, out_file)
    print(f"\n📊 Evaluating {name}...")
    
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: continue

    data = [json.loads(l) for l in open(in_path)]

    with open(out_path, "a") as f:
        for i, d in enumerate(tqdm(data)):
            item_id = d.get("id", i)
            if item_id in done_ids:
                continue
            
            question = d.get("question", d.get("stem", ""))
            if not question and "question" in d:
                question = d["question"].get("stem", "")

            res = solve(question, debug=(i < DEBUG_SAMPLES))

            pred = extract_fn(res["final"])
            gt = extract_number(str(d.get("answer", ""))) if name == "GSM8K" else str(d.get("answerKey", "")).strip().upper()
            
            entry = {
                "id": item_id,
                "is_correct": (pred == gt and pred is not None),
                "metrics": res["metrics"],
                "prediction": res["final"],
                "ground_truth": gt,
                "trace": res["trace"]
            }

            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    start = time.time()

    # run_evaluation("GSM8K", GSM8K_PATH, "gsm8k_moe_results.jsonl", extract_number)
    run_evaluation("CSQA", CSQA_PATH, "csqa_moe_results.jsonl", extract_choice)

    print(f"\n⏱ Total Time: {time.time() - start:.2f}s")
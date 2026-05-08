import json
import time
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

# ==========================================
# 🚩 GLOBAL TOGGLES (Set to False to skip)
# ==========================================
RUN_CSQA = False
RUN_GSM8K = True

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/causal_moe_v2/integrated_model/checkpoint-100"
GSM8K_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_test.jsonl"
CSQA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_val.jsonl"
OUT_DIR = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/inference_moe_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(script_dir)
if src_root not in sys.path:
    sys.path.append(src_root)

from causal_moe_v2.architecture import convert_qwen_to_causal_moe

# ==========================================
# LOAD INTEGRATED MODEL
# ==========================================
print("🚀 Loading Integrated Causal MoE V2...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Apply architecture surgery (must match train.py)
model = convert_qwen_to_causal_moe(model, num_experts=4, moe_layers=[6, 12, 18, 24], reduction_factor=0.5)

# Load the trained adapter
print(f"🛰️  Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ==========================================
# UTILS & METRICS
# ==========================================
def extract_number(text):
    if not text: return None
    # Use the robust base extraction logic
    match = re.findall(r"###\s*\[?(-?[\d,.]+)\]?", str(text))
    if not match:
        match = re.findall(r"####\s*(-?[\d,.]+)", str(text))
    if not match:
        # Fallback: Find any number that looks like a final answer at the end
        match = re.findall(r"(-?[\d,]+(?:\.\d+)?)", str(text))
    
    if match:
        try:
            # Take the very last numerical value
            val_str = match[-1].replace(",", "").strip().rstrip('.')
            return float(val_str)
        except:
            return None
    return None

def extract_choice(text):
    if not text or text == "FAILED": return None
    match = re.search(r"(?:Choice|####|answer is)\s*[:\s]*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    letters = re.findall(r"\b[A-E]\b", text)
    return letters[-1] if letters else None

def calculate_v2_flops(input_tokens, output_tokens):
    total_tokens = input_tokens + output_tokens
    active_params = 6.67 * 1e9 # Savings due to reduction_factor=0.5
    return 2 * active_params * total_tokens

# ==========================================
# INFERENCE ENGINE
# ==========================================
def solve(question, choices=None):
    input_str = f"Question: {question}"
    if choices:
        opt_str = ", ".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])
        input_str += f" Options: {opt_str}"

    prompt = (
        "### Instruction:\n"
        "Analyze the following problem and provide a solution using ATOMIC reasoning tags: [MATH], [LOGIC], [COMMONSENSE].\n"
        "ALWAYS conclude with the [VERIFY] tag followed by the final answer like this: '[VERIFY] #### [number]'.\n\n"
        "### Input:\n"
        f"{input_str}\n\n"
        "### Response:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    # Nuclear Override
    model.generation_config.max_new_tokens = 512
    model.config.max_new_tokens = 512
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.0, # Atomic format requires repetition of tags
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["### Input:", "### Instruction:", "Human:", "User:"],
            tokenizer=tokenizer
        )
    latency = time.time() - start_time
    
    output_tokens = output_ids[0][input_len:]
    response_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    
    if "####" in response_text:
        parts = response_text.split("####")
        response_text = parts[0] + "#### " + parts[1].strip()[:10]
    
    stops = ["### Input:", "### Instruction:", "Human:", "User:", "Note:"]
    for stop in stops:
        if stop in response_text:
            response_text = response_text.split(stop)[0].strip()
            
    in_tokens = input_len
    out_tokens = len(output_tokens)
    flops = calculate_v2_flops(in_tokens, out_tokens)
    
    raw_trace = response_text.replace("\n", " ").split("[")
    trace = ["[" + t.strip() for t in raw_trace if t.strip()]
    
    return {
        "final": response_text,
        "metrics": {
            "router_tokens": 0,
            "expert_in_tokens": in_tokens,
            "expert_out_tokens": out_tokens,
            "total_flops": flops,
            "latency": latency,
            "steps": 1,
            "total_tokens": in_tokens + out_tokens
        }
    }

def run_evaluation(name, in_path, out_file, extract_fn):
    out_path = os.path.join(OUT_DIR, out_file)
    print(f"\n📊 Evaluating on {name}...")
    
    # Removed SKIP logic to ensure fresh run of truncated items
    data = [json.loads(l) for l in open(in_path)]

    with open(out_path, "w") as f: # Overwrite to clear stale 256-token data
        for i, d in enumerate(tqdm(data)):
            item_id = d.get("id", i)
            
            question = d.get("question", d.get("stem", ""))
            if not question and "question" in d:
                question = d["question"].get("stem", "")
            
            choices = d.get("choices", None)
            res = solve(question, choices=choices)

            pred = extract_fn(res["final"])
            gt = extract_number(str(d.get("answer", ""))) if name == "GSM8K" else str(d.get("answerKey", "")).strip().upper()
            
            is_correct = False
            if name == "GSM8K":
                if pred is not None and gt is not None:
                    is_correct = abs(pred - gt) < 1e-4
            else:
                is_correct = (pred == gt and pred is not None)

            entry = {
                "id": item_id,
                "is_correct": is_correct,
                "metrics": res["metrics"],
                "prediction_text": res["final"],
                "prediction_val": pred,
                "ground_truth_val": gt
            }

            f.write(json.dumps(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    if RUN_CSQA:
        run_evaluation("CSQA", CSQA_PATH, "csqa_vs_results.jsonl", extract_choice)
    
    if RUN_GSM8K:
        run_evaluation("GSM8K", GSM8K_PATH, "gsm8k_v2_results.jsonl", extract_number)

    print(f"\n✅ Evaluation complete. Results in {OUT_DIR}")

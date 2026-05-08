import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# PATHS
# ==========================================
BASE_MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/verify_expert_qwen_7b/checkpoint-50"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD TOKENIZER
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# LOAD MODEL
# ==========================================
USE_MERGED = False  # set True if you saved merged model

if USE_MERGED:
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_PATH + "_merged",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()


# ==========================================
# PROMPT (MATCH TRAINING EXACTLY)
# ==========================================
def build_prompt(question, context):
    return (
        "### Instruction:\n"
        "You are a verification expert.\n"
        "Select the correct final answer based on the reasoning.\n"
        "Output ONLY the answer in format: [VERIFY] #### X\n\n"
        "### Input:\n"
        f"Question: {question}\n"
        f"Context: {context}\n\n"
        "### Response:\n"
    )


# ==========================================
# GENERATE
# ==========================================
@torch.no_grad()
def generate_verify(question, context):
    prompt = build_prompt(question, context)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # 🔥 keep very small
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response
    if "### Response:\n" in decoded:
        response = decoded.split("### Response:\n")[-1]
    else:
        response = decoded

    # 🔥 STRICT CLEANING
    response = response.split("\n")[0]

    # Ensure we only keep verify part
    if "[VERIFY]" in response:
        response = "[VERIFY]" + response.split("[VERIFY]")[-1]

    # Cut after #### answer
    if "####" in response:
        parts = response.split("####")
        if len(parts) > 1:
            answer = parts[1].strip().split()[0]
            response = f"[VERIFY] #### {answer}"

    return response.strip()


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":

    question = "If people are disappointed by something they aren't entitled to, what must they do?"

    context = (
        "[LOGIC] Desires mismatch reality, feel disappointment "
        "[LOGIC] Adjust desires for reality "
        "[LOGIC] Adjusting expectations aligns desires"
    )

    output = generate_verify(question, context)

    print("\n🧠 VERIFY OUTPUT:")
    print(output)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# PATHS
# ==========================================
BASE_MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/math_expert_qwen_7b/checkpoint-800"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD TOKENIZER (FROM OUTPUT DIR)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# LOAD MODEL
# ==========================================
USE_MERGED = False  # set True if using merged model

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
# PROMPT (MATCH TRAINING)
# ==========================================
def build_prompt(question, context):
    return (
        "### Instruction:\n"
        "You are a mathematical reasoning expert.\n"
        "Perform the next calculation step.\n"
        "Output ONLY the computation.\n\n"
        "### Input:\n"
        f"Question: {question}\n"
        f"Context: {context}\n\n"
        "### Response:\n"
    )


# ==========================================
# GENERATE
# ==========================================
@torch.no_grad()
def generate_math(question, context, max_new_tokens=20):
    prompt = build_prompt(question, context)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
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

    # 🔥 Stop at next tag (prevents multi-step spill)
    if "[MATH]" in response:
        parts = response.split("[MATH]")
        if len(parts) > 2:
            response = "[MATH]" + parts[1]

    # Cleanup
    response = response.split("\n")[0].strip()

    # 🔥 Safety: ensure numeric content
    if not any(char.isdigit() for char in response):
        return "[MATH] INVALID"

    return response


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":

    question = "Sandra had 2 different bags of candy. Each had 6 pieces. Roger had 11 and 3. How much more did Roger have?"

    context = (
        "[MATH] 2 * 6 = 12"
    )

    output = generate_math(question, context)

    print("\n🧠 MATH OUTPUT:")
    print(output)


# ==========================================
# OPTIONAL: MULTI-STEP ROLLOUT
# ==========================================
def rollout(question, context, steps=3):
    print("\n🔁 MULTI-STEP MATH ROLLOUT\n")

    for i in range(steps):
        step = generate_math(question, context)
        print(f"Step {i+1}: {step}")
        context += " " + step

    return context
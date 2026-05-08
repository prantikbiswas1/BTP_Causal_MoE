import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# PATHS
# ==========================================
BASE_MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/logic_expert_qwen_7b/checkpoint-100"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token

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
        "You are a logical reasoning expert.\n"
        "Provide ONLY the next logical step.\n"
        "Use a short structured format.\n"
        "Do NOT explain.\n\n"
        "### Input:\n"
        f"Question: {question}\n"
        f"Context: {context}\n\n"
        "### Response:\n"
    )


# ==========================================
# GENERATE
# ==========================================
@torch.no_grad()
def generate_step(question, context):
    prompt = build_prompt(question, context)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=25,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response
    response = decoded.split("### Response:\n")[-1]

    # Stop at next logic tag (important for chain models)
    if "[LOGIC]" in response:
        parts = response.split("[LOGIC]")
        if len(parts) > 2:
            response = "[LOGIC]" + parts[1]

    # Clean
    response = response.split("\n")[0].strip()

    return response


# ==========================================
# TEST (MATCH DATA FORMAT)
# ==========================================
if __name__ == "__main__":

    question = "If people are disappointed by something they aren't entitled to, what must they do?"

    context = (
        "[LOGIC] Desires mismatch reality, feel disappointment "
        "[LOGIC] Adjust desires for reality"
    )

    output = generate_step(question, context)

    print("\n🧠 MODEL OUTPUT:")
    print(output)
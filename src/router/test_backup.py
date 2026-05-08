import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def validate_router():
    BASE_MODEL = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/Qwen2.5-1.5B-Instruct"
    LORA_ADAPTER = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/router/router_qwen_1.5b/final_router"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Ensure pad_token is set correctly
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
    model.eval()

    instruction_text = (
        "Predict the next reasoning tag:\n"
        "- MATH: for numerical calculations\n"
        "- LOGIC: for step-by-step deductive reasoning\n"
        "- COMMONSENSE: for everyday planning and world knowledge\n"
        "- VERIFY: for checking if the final answer is correct"
    )

    test_cases = [
        {"name": "CSQA Start (Step 0)", "input": "Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? Options: A: bank, B: library, C: department store, D: mall, E: new york\nContext:", "expected": ["LOGIC", "COMMONSENSE"]},
        {"name": "CSQA Mid (Step 1)", "input": "Question: A revolving door is convenient... \nContext: LOGIC Revolving doors control flow.", "expected": ["LOGIC", "COMMONSENSE"]},
        {"name": "GSM8K Start (Step 0)", "input": "Question: Janet’s ducks lay 16 eggs per day. She eats 3 and bakes with 4. She sells the rest for $2 each. How much does she make?\nContext:", "expected": ["MATH"]},
        {"name": "GSM8K Mid (Step 1)", "input": "Question: Janet’s ducks lay 16 eggs per day. She eats 3 and bakes with 4. She sells the rest for $2 each. How much does she make?\nContext: MATH 16 - 3 - 4 = 9 eggs sold", "expected": ["MATH"]},
        {"name": "GSM8K End (Step 2)", "input": "Question: Janet’s ducks lay 16 eggs per day. She eats 3 and bakes with 4. She sells the rest for $2 each. How much does she make?\nContext: MATH 16 - 3 - 4 = 9 eggs sold MATH 9 * 2 = 18 dollars", "expected": ["VERIFY"]}
    ]

    print("\n" + "="*80)
    print(f"{'STEP DESCRIPTION':<35} | {'PREDICTED':<12} | {'STATUS'}")
    print("="*80)

    correct = 0
    for case in test_cases:
        prompt = f"### Instruction:\n{instruction_text}\n\n### Input:\n{case['input']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=15, 
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # SLICE the output to only get the new tokens
        input_length = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_length:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
        
        # Cleaning: split by whitespace and take only the FIRST word (the tag)
        clean_pred = prediction.split()[0].strip("[]:.,") if prediction else "EMPTY"

        status = "✅ PASS" if clean_pred in case['expected'] else f"❌ FAIL (Exp: {case['expected'][0]})"
        if clean_pred in case['expected']:
            correct += 1
            
        print(f"{case['name']:<35} | {clean_pred:<12} | {status}")

    print("="*80)
    print(f"Router Test Accuracy: {correct}/{len(test_cases)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    validate_router()
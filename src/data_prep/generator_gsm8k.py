import json
import os
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct" 
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_train.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/gsm8k_raw_multistep_traces.jsonl"

def format_gsm8k_question(item):
    """
    Forces granular mathematical reasoning.
    Ensures every step is an atomic calculation or logic jump.
    """
    question = item["question"]
    
    return (
        f"<|im_start|>system\n"
        f"You are a mathematical reasoning engine. Solve the word problem using atomic, granular steps.\n"
        f"Rules:\n"
        f"1. Each step must be a single calculation or a single logical deduction.\n"
        f"2. Write each step as a complete sentence on a new line.\n"
        f"3. Do not skip intermediate calculations.\n"
        f"4. Focus only on the necessary steps to reach the final number.\n"
        f"5. End strictly with '#### [Number]'.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_gsm8k_answer(text):
    """Extracts the numeric value after ####, removing commas."""
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""

def run_gsm8k_generation(model_path, input_path, output_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=700, 
        stop=["<|im_end|>", "Question:"] 
    )

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Generating GSM8K Traces for {len(data)} items...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as out_f:
        batch_size = 64 
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            prompts = [format_gsm8k_question(item) for item in batch]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            
            for idx, output in enumerate(outputs):
                full_text = output.outputs[0].text.strip()
                steps_list = [s.strip() for s in full_text.split('\n') if s.strip()]
                
                pred_answer = extract_gsm8k_answer(full_text)
                
                # Extract GT answer from original GSM8K 'answer' string
                orig_item = batch[idx]
                gt_match = re.search(r"####\s*(-?[\d,.]+)", orig_item["answer"])
                gt = gt_match.group(1).replace(",", "").strip() if gt_match else ""
                
                is_correct = (pred_answer == gt)

                # CRITICAL: We include 'question' here so we don't need the raw file later
                result = {
                    "id": i + idx, 
                    "question": orig_item["question"], # Kept for Router training context
                    "ground_truth": gt,
                    "predicted": pred_answer,
                    "is_correct": is_correct,
                    "steps": steps_list,
                    "raw_text": full_text
                }
                out_f.write(json.dumps(result) + "\n")
            
            out_f.flush()

if __name__ == "__main__":
    run_gsm8k_generation(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)
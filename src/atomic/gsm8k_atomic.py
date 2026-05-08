import json
import re
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/gsm8k_tagged_final.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/gsm8k_atomic.jsonl"
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"

def clean_extracted_output(text, original_content):
    """STRICT CLEANER: Preserves original math and catches hallucinations."""
    # 1. Strip common hallucinated prefixes
    patterns_to_remove = [
        r"Instruction:.*", r"Extract.*", r"Atomic Fact[:\s-]*", 
        r"Original[:\s-]*", r"Thought[:\s-]*", r"Output[:\s-]*",
        r"Atomic:.*", r"Math:.*"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 2. HALLUCINATION FAIL-SAFES
    bad_outputs = ["0", "0.", "1+1=2", "1 + 1 = 2", "0.0", ""]
    if text in bad_outputs or text.endswith("* 0") or text.endswith("/ 0"):
        fallback = re.sub(r'[A-Za-z]+', '', original_content)
        return original_content.strip() if len(fallback) < 2 else original_content.strip()

    # 3. Clean quotes and brackets
    text = text.replace('"', '').replace('[', '').replace(']', '').strip()
    
    # 4. DECIMAL PRESERVATION FIX: Split on newline only to kill yapping, preserve periods
    text = text.split('\n')[0].strip()
    
    # Safely remove trailing punctuation if the LLM added it at the very end
    if text.endswith('.') or text.endswith('!') or text.endswith('?'):
        text = text[:-1].strip()
        
    return text

def build_gsm8k_atomic_dataset():
    print("Initializing vLLM Engine for GSM8K Atomic Extraction...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=60, 
        stop=["\n", "Target:", "[", "Instruction:", "Input:", "<|im_end|>"]
    )
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Extracting {len(data)} GSM8K samples into atomic units...")
    BATCH_SIZE = 64 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch_items = data[i : i + BATCH_SIZE]
            all_prompts, prompt_map = [], [] 

            for b_idx, item in enumerate(batch_items):
                raw_trace = item["atomic_reasoning"]
                steps = re.findall(r"\[(LOGIC|COMMONSENSE|MATH)\]\s*(.*?)(?=\s*\[|$)", raw_trace)
                
                for s_idx, (tag, content) in enumerate(steps):
                    prompt = (
                        f"<|im_start|>system\n"
                        f"You are a strict Data Extractor. Your job is to extract the core equation or fact from the text.\n"
                        f"CRITICAL RULES:\n"
                        f"1. COPY the exact math equation. DO NOT calculate or solve anything yourself.\n"
                        f"2. DO NOT round decimals. If it says 10/100 = 0.1, output 10/100 = 0.1.\n"
                        f"3. Remove filler words, keep the math.\n"
                        f"Example Input: 'Next, we calculate the profit: 50 * 2.5 = 125 dollars.'\n"
                        f"Example Output: '50 * 2.5 = 125'<|im_end|>\n"
                        f"<|im_start|>user\n"
                        f"Text to extract: {content.strip()}\n"
                        f"Extraction:<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    all_prompts.append(prompt)
                    prompt_map.append((b_idx, s_idx, tag, content.strip()))

            if not all_prompts: continue

            outputs = llm.generate(all_prompts, sampling_params, use_tqdm=False)
            
            batch_results = {idx: {} for idx in range(len(batch_items))}
            for out, (b_idx, s_idx, original_tag, original_content) in zip(outputs, prompt_map):
                raw_extracted = out.outputs[0].text.strip()
                refined_text = clean_extracted_output(raw_extracted, original_content)
                batch_results[b_idx][s_idx] = f"[{original_tag}] {refined_text}"

            for b_idx, item in enumerate(batch_items):
                sorted_idx = sorted(batch_results[b_idx].keys())
                sorted_steps = [batch_results[b_idx][idx] for idx in sorted_idx]
                
                new_trace = " ".join(sorted_steps) + f" [VERIFY] #### {item['answer']}"
                
                f_out.write(json.dumps({
                    "id": item["id"],
                    "instruction": item["instruction"],
                    "atomic_reasoning": new_trace,
                    "answer": item["answer"]
                }) + "\n")
            
            f_out.flush() 

    print(f"Success! Mathematically preserved dataset saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_gsm8k_atomic_dataset()
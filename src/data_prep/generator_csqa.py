import json
import os
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct" 
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_train.jsonl"
OUTPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/csqa_raw_multistep_traces.jsonl"

def format_csqa_question(item):
    """
    Forces verbose, granular, step-by-step reasoning.
    Explicitly forbids analyzing incorrect options (distractors).
    Provides a clean text baseline for future BTP tagging.
    """
    question = item["question"]
    choices = item["choices"]
    options = "\n".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])
    
    return (
        f"<|im_start|>system\n"
        f"You are a reasoning engine. Solve the question using a granular, multi-step causal chain.\n"
        f"Rules:\n"
        f"1. Break the reasoning into the smallest possible atomic logical steps.\n"
        f"2. Provide only the correct reasoning path. Do not mention or analyze incorrect options.\n"
        f"3. Each step must be a single, complete sentence on its own line.\n"
        f"4. Focus on extracting facts first, then applying world knowledge, then concluding.\n"
        f"5. End strictly with '#### <answer> [Letter]'.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {question}\n"
        f"Options:\n{options}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_answer_flexible(text):
    """Catches #### <answer> A, #### A, or the last A-E letter in the text."""
    match = re.search(r"####\s*(?:<answer>|answer:)?\s*\[?([A-E])\]?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text)
    return letters[-1].upper() if letters else ""

def run_raw_generation(model_path, input_path, output_path):
    # Initialize vLLM with Ray Backend (TP=2)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        distributed_executor_backend="ray", 
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    # Setup Sampling
    # stop=["Option A:"] prevents the model from starting an elimination list
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=500, 
        stop=["<|im_end|>", "Option A:", "Option B:", "1.", "A:"] 
    )

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Generating Granular Raw Traces for {len(data)} items...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as out_f:
        batch_size = 64 
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            prompts = [format_csqa_question(item) for item in batch]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            
            for idx, output in enumerate(outputs):
                full_text = output.outputs[0].text.strip()
                
                # Split reasoning by newlines for future PNS-based pruning
                steps_list = [s.strip() for s in full_text.split('\n') if s.strip()]
                
                pred_answer = extract_answer_flexible(full_text)
                gt = batch[idx].get("answerKey")
                is_correct = (pred_answer == gt)

                result = {
                    "id": batch[idx].get("id"),
                    "ground_truth": gt,
                    "predicted": pred_answer,
                    "is_correct": is_correct,
                    "steps": steps_list,
                    "raw_text": full_text
                }
                out_f.write(json.dumps(result) + "\n")
            
            out_f.flush()

if __name__ == "__main__":
    run_raw_generation(MODEL_PATH, INPUT_FILE, OUTPUT_FILE)
import json
import random
import os

# --- CONFIGURATION ---
CSQA_INPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/csqa_atomic.jsonl"
GSM8K_INPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/gsm8k_atomic.jsonl"
FINAL_OUTPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"

def merge_and_shuffle():
    combined_data = []
    
    # 1. Load CSQA (and prefix IDs to avoid collisions)
    if os.path.exists(CSQA_INPUT):
        with open(CSQA_INPUT, 'r') as f:
            for line in f:
                item = json.loads(line)
                item["id"] = f"csqa_{item['id']}"
                item["dataset"] = "csqa" # Useful for metadata tracking
                combined_data.append(item)
    
    # 2. Load GSM8K
    if os.path.exists(GSM8K_INPUT):
        with open(GSM8K_INPUT, 'r') as f:
            for line in f:
                item = json.loads(line)
                item["id"] = f"gsm8k_{item['id']}"
                item["dataset"] = "gsm8k"
                combined_data.append(item)

    print(f"Total samples collected: {len(combined_data)}")
    print(f"CSQA count: {len([x for x in combined_data if x['dataset'] == 'csqa'])}")
    print(f"GSM8K count: {len([x for x in combined_data if x['dataset'] == 'gsm8k'])}")

    # 3. High-Entropy Shuffle
    # We shuffle twice to ensure total distribution across batches
    random.seed(42)
    random.shuffle(combined_data)
    random.shuffle(combined_data)

    # 4. Save to Final Training File
    os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)
    with open(FINAL_OUTPUT, 'w') as f_out:
        for item in combined_data:
            f_out.write(json.dumps(item) + "\n")

    print(f"Success! Final mixed dataset saved at: {FINAL_OUTPUT}")

if __name__ == "__main__":
    merge_and_shuffle()
import json
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"
LOGIC_OUTPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/experts/logic/logic_train_exploded.jsonl"

def prepare_logic_expert_data():
    exploded_dataset = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("🚀 Exploding CLEAN Logic Expert Traces...")
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
            except: continue
                
            instr = item["instruction"]
            steps = re.findall(r"(\[(?:LOGIC|MATH|COMMONSENSE|VERIFY)\])\s*(.*?)(?=\s*\[|$)", item["atomic_reasoning"])
            
            context_list = []
            for i, (tag, content) in enumerate(steps):
                clean_content = content.strip()
                current_full_step = f"{tag} {clean_content}"

                if tag == "[LOGIC]":
                    # FUZZY REDUNDANCY FILTER
                    norm_content = re.sub(r'[\s\(\)]', '', clean_content).lower()
                    norm_history = re.sub(r'[\s\(\)]', '', "".join(context_list)).lower()
                    
                    if norm_content not in norm_history and "Atomic" not in clean_content:
                        exploded_dataset.append({
                            "id": f"{item['id']}_logic_{i}",
                            "instruction": instr,
                            "context_before": " ".join(context_list).strip(),
                            "logic_step": current_full_step 
                        })
                
                context_list.append(current_full_step)

    os.makedirs(os.path.dirname(LOGIC_OUTPUT), exist_ok=True)
    with open(LOGIC_OUTPUT, 'w') as f_out:
        for entry in exploded_dataset:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Success! Logic path: {LOGIC_OUTPUT}")

if __name__ == "__main__":
    prepare_logic_expert_data()
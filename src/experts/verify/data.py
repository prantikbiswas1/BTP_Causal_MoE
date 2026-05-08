import json
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"
VERIFY_OUTPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/experts/verify/verify_train_exploded.jsonl"

def prepare_verify_expert_data():
    exploded_dataset = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("🚀 Extracting CLEAN Terminal Verify steps...")
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
            except: continue
                
            instr = item["instruction"]
            steps = re.findall(r"(\[(?:LOGIC|MATH|COMMONSENSE|VERIFY)\])\s*(.*?)(?=\s*\[|$)", item["atomic_reasoning"])
            
            if not steps or steps[-1][0] != "[VERIFY]":
                continue

            final_tag, final_content = steps[-1]
            
            # Reconstruct context
            context_list = []
            for i in range(len(steps) - 1):
                tag, content = steps[i]
                context_list.append(f"{tag} {content.strip()}")
            
            context_string = " ".join(context_list).strip()

            # For Verify, we don't filter out redundancy as strictly because 
            # its job is to extract the final value into a specific format.
            # But we still clean out Atomic noise.
            if "Atomic" not in final_content:
                exploded_dataset.append({
                    "id": f"{item['id']}_final_verify",
                    "instruction": instr,
                    "context_before": context_string,
                    "verify_step": f"{final_tag} {final_content.strip()}" 
                })

    os.makedirs(os.path.dirname(VERIFY_OUTPUT), exist_ok=True)
    with open(VERIFY_OUTPUT, 'w') as f_out:
        for entry in exploded_dataset:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Success! Verify path: {VERIFY_OUTPUT}")

if __name__ == "__main__":
    prepare_verify_expert_data()
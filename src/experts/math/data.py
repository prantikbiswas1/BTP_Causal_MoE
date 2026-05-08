import json
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"
MATH_EXPERT_OUTPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/experts/math/math_train_exploded.jsonl"

def prepare_math_expert_data():
    exploded_math_dataset = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("🚀 Generating ULTIMATE CLEAN Math Expert Dataset...")
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
            except: continue
                
            instr = item["instruction"]
            # Capture all steps to maintain state, including VERIFY
            steps = re.findall(r"(\[(?:LOGIC|MATH|COMMONSENSE|VERIFY)\])\s*(.*?)(?=\s*\[|$)", item["atomic_reasoning"])
            
            context_list = []
            
            for i, (tag, content) in enumerate(steps):
                clean_content = content.strip()
                
                # 1. BROAD NOISE FILTER: Kill "Atomic" placeholders and generic fillers
                if any(x in clean_content for x in ["Atomic", "1+1=", "10+10=", "10+15="]):
                    continue

                current_full_step = f"{tag} {clean_content}"

                if tag == "[MATH]":
                    # 2. FUZZY REDUNDANCY FILTER
                    # We normalize the content (remove spaces, symbols, lowercase) 
                    # to see if the model is just repeating a previous thought.
                    norm_content = re.sub(r'[\s\(\)\$]', '', clean_content).lower()
                    history_string = "".join(context_list)
                    norm_history = re.sub(r'[\s\(\)\$]', '', history_string).lower()
                    
                    if norm_content not in norm_history:
                        exploded_math_dataset.append({
                            "id": f"{item['id']}_math_{i}",
                            "instruction": instr,
                            "context_before": " ".join(context_list).strip(),
                            "math_step": current_full_step
                        })
                
                # Update context list with the current step
                context_list.append(current_full_step)

    # Save the cleaned dataset
    os.makedirs(os.path.dirname(MATH_EXPERT_OUTPUT), exist_ok=True)
    with open(MATH_EXPERT_OUTPUT, 'w') as f_out:
        for entry in exploded_math_dataset:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Success!")
    print(f"Final Count: {len(exploded_math_dataset)} clean math steps.")
    print(f"Saved to: {MATH_EXPERT_OUTPUT}")

if __name__ == "__main__":
    prepare_math_expert_data()
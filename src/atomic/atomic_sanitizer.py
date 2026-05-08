import json
import os
import random
import re

def sanitize_and_combine(input_files, output_file):
    combined_list = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
        
        print(f"Processing and Verifying: {file_path}")
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue
                
                # VERIFICATION: Only proceed if is_golden is explicitly True
                if not data.get("is_golden") == True:
                    continue
                
                raw_trace = data.get("atomic_trace", "")
                if not raw_trace:
                    continue

                # STRIP BRACKETS: [TAG] content -> TAG content
                # This ensures your Router and Experts see exactly the same format
                sanitized_steps = []
                # Splitting by '[' and handling the content inside/after ']'
                parts = raw_trace.split('[')
                for part in parts:
                    if not part or ']' not in part:
                        continue
                    tag, content = part.split(']', 1)
                    # Resulting format: "LOGIC subject = sanctions"
                    sanitized_steps.append(f"{tag.strip()} {content.strip()}")
                
                # Update the object with the clean, bracket-less trace
                data["atomic_trace"] = "\n".join(sanitized_steps)
                combined_list.append(data)

    # Shuffling is critical so the 1.5B Router doesn't get biased by file order
    print(f"Shuffling {len(combined_list)} verified golden examples...")
    random.seed(42)
    random.shuffle(combined_list)

    with open(output_file, 'w') as out_f:
        for item in combined_list:
            out_f.write(json.dumps(item) + "\n")
    
    print(f"SUCCESS: {len(combined_list)} verified examples saved to {output_file}")

# --- EXECUTION ---

input_paths = [
    "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/commonsense_qa_train_atomic.jsonl",
    "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/gsm8k_train_atomic.jsonl"
]

output_path = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic_sanitized/combined_data.jsonl"

sanitize_and_combine(input_paths, output_path)
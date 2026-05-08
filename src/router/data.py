import json
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"
ROUTER_OUTPUT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/router/router_train_exploded.jsonl"


def prepare_step_level_router_data():
    exploded_dataset = []

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("🚀 Exploding traces into Step-Level ROUTER data...")

    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue

            instr = item["instruction"]

            # ✅ FIX: include VERIFY
            steps = re.findall(
                r"\[(LOGIC|MATH|COMMONSENSE|VERIFY)\]\s*(.*?)(?=\s*\[|$)",
                item["atomic_reasoning"]
            )

            context_so_far = ""

            for i, (tag, content) in enumerate(steps):
                clean_content = content.strip()

                # ----------------------
                # CREATE TRAINING POINT
                # ----------------------
                exploded_entry = {
                    "id": f"{item['id']}_step_{i}",
                    "input_context": f"Question: {instr}\nContext: {context_so_far}".strip(),
                    "next_expert": tag,  # label
                    "step_text": clean_content,
                    "dataset_origin": item.get("dataset", "unknown")
                }

                exploded_dataset.append(exploded_entry)

                # ----------------------
                # UPDATE CONTEXT
                # ----------------------
                context_so_far += f" {tag} {clean_content}"

    # ----------------------
    # SAVE
    # ----------------------
    os.makedirs(os.path.dirname(ROUTER_OUTPUT), exist_ok=True)

    with open(ROUTER_OUTPUT, 'w') as f_out:
        for entry in exploded_dataset:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Success! Generated {len(exploded_dataset)} router samples.")
    print(f"📁 Saved to: {ROUTER_OUTPUT}")


if __name__ == "__main__":
    prepare_step_level_router_data()
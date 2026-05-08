import os
import json
import re
import shutil

def sanitize_traces():
    """
    Reads tagged JSONL files, strips all reasoning tags ([LOGIC], [MATH], etc.),
    and saves them to a fresh 'finetune/' directory for a Pure Neural run.
    """
    # Cloud Absolute Paths
    PROJECT_ROOT = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/"
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data/final_compact")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/finetune")
    
    # CLEAR THE FOLDER AT THE START
    if os.path.exists(OUTPUT_DIR):
        print(f"🗑 Clearing existing files in: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🧹 Starting Sanitization of {INPUT_DIR}...")
    
    tagged_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    
    for filename in tagged_files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"👉 Cleaning: {filename}...")
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                item = json.loads(line)
                question = item.get("question", "") or ""
                
                # UNROLLING LOGIC: We want the model to learn from ALL variations
                # If a questions has 4 traces, we write 4 separate training rows.
                traces = item.get("compact_traces", [])
                if not traces: 
                    traces = item.get("tagged_traces", [])
                
                # Force into a list if it's a single trace
                if isinstance(traces, str):
                    traces = [traces]
                
                if isinstance(traces, list):
                    for idx, raw_trace in enumerate(traces):
                        # STRIP ALL TAGS: [LOGIC], [MATH], [VERIFY], [END]
                        clean_trace = re.sub(r"\[(LOGIC|MATH|VERIFY|END)\]", "", str(raw_trace)).strip()
                        
                        # Save each variation as its own training example
                        sanitized_item = {
                            "id": f"{item.get('id', 'N/A')}_trace_{idx}",
                            "question": question,
                            "text": f"Question: {question}\nAnswer: {clean_trace}"
                        }
                        fout.write(json.dumps(sanitized_item) + "\n")

    print(f"\n✅ Sanitization Complete! Data is ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    sanitize_traces()

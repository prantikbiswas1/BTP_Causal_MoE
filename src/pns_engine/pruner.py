import json
import argparse
from tqdm import tqdm

def prune_traces(input_filepath, output_filepath, threshold=0.3):
    """
    Reads a file containing traces that have been scored with PNS.
    Strips out any step that has a PNS score below the given threshold.
    Saves a dense, 'Compact CoT' dataset ready for MoE training.
    """
    compact_dataset = []
    
    # 1. Load the full, scored dataset
    # Expected format: {"id": X, "scored_traces": [[{"text": step1, "pns_score": 0.5}], ...]}
    with open(input_filepath, "r") as f:
        for line in tqdm(f, desc="Pruning low-PNS steps"):
            item = json.loads(line)
            pruned_traces = []
            
            for scored_trace in item.get('scored_traces', []):
                compact_trace = []
                for step in scored_trace:
                    # 2. Key logic: Is this step probabilistically necessary?
                    if step.get("pns_score", 0.0) >= threshold:
                        compact_trace.append(step["text"])
                
                # Reconstruct trace string if any steps survived
                if compact_trace:
                    pruned_traces.append("\n".join(compact_trace))
            
            # Save the clean versions
            if pruned_traces:
                clean_item = {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "compact_traces": pruned_traces
                }
                compact_dataset.append(clean_item)
                
    # 3. Write out the final, highly-efficient pruned dataset
    with open(output_filepath, "w") as out_f:
        for clean_item in compact_dataset:
            out_f.write(json.dumps(clean_item) + "\n")
    
    print(f"Pruned dataset saved to {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to PNS-scored jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to save compact jsonl")
    parser.add_argument("--thresh", type=float, default=0.3, help="PNS threshold for pruning")
    args = parser.parse_args()
    
    prune_traces(args.input, args.output, args.thresh)

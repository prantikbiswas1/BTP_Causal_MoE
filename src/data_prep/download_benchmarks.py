import os
from datasets import load_dataset

def main():
    print("🚀 Starting benchmark downloads...")
    # Jupyter notebooks do not define __file__. We handle both script execution and Jupyter execution.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.abspath(os.path.join(script_dir, '../../data/raw'))
    except NameError:
        # We are running inside a Jupyter cell. Assume the CWD is the project root or the current cell environment.
        script_dir = os.getcwd()
        # Create a clean folder right where the Jupyter notebook is running
        raw_data_dir = os.path.join(script_dir, 'BTP_Causal_MoE/data/raw')
    
    os.makedirs(raw_data_dir, exist_ok=True)

    # 1. Download GSM8k (main math benchmark)
    print("Downloading openai/gsm8k (main)...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
        gsm8k['train'].to_json(os.path.join(raw_data_dir, 'gsm8k_train.jsonl'))
        gsm8k['test'].to_json(os.path.join(raw_data_dir, 'gsm8k_test.jsonl'))
        print("✅ GSM8k downloaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to download GSM8k: {e}\n")

    # 2. Download MATH-500 (hard math benchmark)
    print("Downloading HuggingFaceH4/MATH-500...")
    try:
        math500 = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)
        # usually math500 has a test split primarily
        if 'test' in math500:
            math500['test'].to_json(os.path.join(raw_data_dir, 'math500_test.jsonl'))
        else:
            print("MATH-500 'test' split not found, saving available splits.")
            for split in math500.keys():
                math500[split].to_json(os.path.join(raw_data_dir, f'math500_{split}.jsonl'))
        print("✅ MATH-500 downloaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to download MATH-500 from HuggingFaceH4, trying fallback to custom pull...")
        # Fallback to taking 500 from hendrycks math test set if H4 doesn't exist either
        try:
            full_math = load_dataset("hendrycks/competition_math", trust_remote_code=True)
            math500_test = full_math['test'].select(range(500))
            math500_test.to_json(os.path.join(raw_data_dir, 'math500_test.jsonl'))
            print("✅ Fallback MATH-500 (sampled from competition_math) downloaded successfully.\n")
        except Exception as fallback_e:
            print(f"❌ Both HuggingFaceH4/MATH-500 and Fallback failed: {fallback_e}\n")

    # 3. Download CommonsenseQA
    print("Downloading commonsense_qa...")
    try:
        csqa = load_dataset("commonsense_qa", trust_remote_code=True)
        csqa['train'].to_json(os.path.join(raw_data_dir, 'commonsense_qa_train.jsonl'))
        csqa['validation'].to_json(os.path.join(raw_data_dir, 'commonsense_qa_val.jsonl'))
        # Usually CSQA test split is blind (no answers), validation is used for testing.
        print("✅ CommonsenseQA downloaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to download CommonsenseQA: {e}\n")

    print(f"🎉 Download complete! All files saved to: {os.path.abspath(raw_data_dir)}")

if __name__ == "__main__":
    main()

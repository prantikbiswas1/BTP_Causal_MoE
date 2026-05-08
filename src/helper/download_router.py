# /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/helper/download_router.py
import os
from huggingface_hub import snapshot_download

# Switching to Qwen 2.5 1.5B - No Gating/Restrictions
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
SAVE_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/Qwen2.5-1.5B-Instruct"

def download_router_base():
    print(f"Downloading {MODEL_ID} to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=SAVE_PATH,
        local_dir_use_symlinks=False,
        revision="main"
    )
    print("\nDownload Complete! You can now run the training script.")

if __name__ == "__main__":
    download_router_base()
import os
import shutil
from huggingface_hub import snapshot_download

def download_qwen():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    local_dir = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
    
    # 🧹 FORCE CLEAN: Metadata corruption detected in previous run
    if os.path.exists(local_dir):
        print(f"🗑️  Cleaning existing directory to resolve metadata corruption...")
        shutil.rmtree(local_dir)
    
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"🚀 Starting fresh download of {model_id}...")
    
    # Download with robust settings
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # Important for cloud storage stability
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    
    print("\n✅ Download Complete!")


download_qwen()
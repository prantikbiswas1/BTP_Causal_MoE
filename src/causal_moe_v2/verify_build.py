import torch
import sys
import os

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(script_dir)
if src_root not in sys.path:
    sys.path.append(src_root)

from causal_moe_v2.architecture import convert_qwen_to_causal_moe
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_build():
    print("="*60)
    print("🧪  MICRO-TEST: Building Integrated Causal MoE V2")
    print("="*60)
    
    # Using your PRODUCTION Qwen-7B model for the test
    MODEL_ID = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
    
    # Fallback to HF hub if local path is not accessible for any reason
    if not os.path.exists(MODEL_ID):
        print(f"⚠️  Local path {MODEL_ID} not found, falling back to HF Hub")
        MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    print(f"📦 Loading base model: {MODEL_ID}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        print("🛠️  Swapping layers to Causal MoE...")
        # Injecting into a few middle-reach layers as a test
        model = convert_qwen_to_causal_moe(model, num_experts=4, moe_layers=[6, 12, 18])
        
        print(f"✅  Architecture swap successful! (Device: {model.device})")
        
        # Test forward pass
        test_text = "What is 2+2?"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        print("🔄  Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅  Forward pass successful! Output logits shape: {outputs.logits.shape}")
        print("\n🎉  Architecture is VALID and ready for training.")
        
    except Exception as e:
        print(f"\n❌  TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_build()

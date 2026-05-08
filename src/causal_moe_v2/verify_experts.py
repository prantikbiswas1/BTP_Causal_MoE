import torch
import torch.nn as nn
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(script_dir)
if src_root not in sys.path:
    sys.path.append(src_root)

from causal_moe_v2.architecture import convert_qwen_to_causal_moe

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/causal_moe_v2/integrated_model"

# Storage for hook data
expert_usage = {} # {layer_idx: [list of expert IDs]}

def expert_hook(module, input, output):
    # This hook captures the gating decision from our CausalMoEMLP
    hidden_states = input[0]
    
    # We re-run the gating logic to see what was chosen
    with torch.no_grad():
        shared_features = module.act_fn(module.router_stem(hidden_states))
        gating_logits = module.gating(shared_features)
        weights = torch.softmax(gating_logits, dim=-1)
        _, top_indices = torch.topk(weights, 1, dim=-1)
        
        # Save the indices (flattened)
        layer_idx = getattr(module, "layer_idx", "unknown")
        if layer_idx not in expert_usage:
            expert_usage[layer_idx] = []
        expert_usage[layer_idx].extend(top_indices.cpu().view(-1).tolist())

def run_verification():
    print("="*60)
    print("🧠  CAUSAL MOE EXPERT VITALITY TEST")
    print("="*60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Surgery
    moe_layers = [6, 12, 18, 24]
    model = convert_qwen_to_causal_moe(model, num_experts=4, moe_layers=moe_layers, reduction_factor=0.5)
    
    # Load Adapter
    print(f"🛰️  Loading Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # Attach Hooks
    for idx in moe_layers:
        layer_module = model.base_model.model.model.layers[idx].mlp
        layer_module.layer_idx = idx
        layer_module.register_forward_hook(expert_hook)

    # --- TEST CASES ---
    test_prompts = {
        "MATH": "Question: If I have 3 apples and buy 4 more, how many do I have? Response: [MATH]",
        "LOGIC/CS": "Question: Where would you find a bank? Response: [COMMONSENSE]"
    }

    for category, prompt in test_prompts.items():
        print(f"\n📝 Testing Category: {category}")
        expert_usage.clear()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
        # Report
        print(f"📊 Expert Selection (Top-1) per MoE Layer:")
        for layer_idx in moe_layers:
            indices = expert_usage.get(layer_idx, [])
            if not indices:
                print(f"  Layer {layer_idx}: No data captured")
                continue
            
            # Count distribution
            counts = [indices.count(i) for i in range(4)]
            dist_str = " | ".join([f"Exp {i}: {counts[i]}" for i in range(4)])
            print(f"  Layer {layer_idx:2d}: {dist_str}")

    print("\n✅ Verification complete. If the 'Exp' counts vary across categories, the Router is ALIVE.")

if __name__ == "__main__":
    run_verification()

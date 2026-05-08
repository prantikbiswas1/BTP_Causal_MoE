import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2PreTrainedModel, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from typing import Optional, List, Union, Tuple

class CausalMoEConfig(Qwen2Config):
    model_type = "causal_moe"
    def __init__(
        self,
        num_experts=4,
        top_k=2,
        reduction_factor=0.5,
        moe_layers=[6, 12, 18, 24],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.reduction_factor = reduction_factor
        self.moe_layers = moe_layers

class CausalMoEMLP(nn.Module):
    """
    Efficient Sparse Causal MoE Layer with FLATTENED Experts (to support PEFT saving)
    """
    def __init__(self, config, num_experts: int, top_k: int = 2, reduction_factor: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = config.hidden_size
        self.expert_intermediate_size = int(config.intermediate_size * reduction_factor)

        # --- ROUTER GATING (Small, efficient) ---
        self.gating = nn.Linear(self.hidden_size, num_experts, bias=False)
        
        # --- FLATTENED EXPERTS (PEFT Compatibility) ---
        for i in range(num_experts):
            setattr(self, f"gate_expert_{i}", nn.Linear(self.hidden_size, self.expert_intermediate_size, bias=False))
            setattr(self, f"up_expert_{i}", nn.Linear(self.hidden_size, self.expert_intermediate_size, bias=False))
            setattr(self, f"down_expert_{i}", nn.Linear(self.expert_intermediate_size, self.hidden_size, bias=False))

    def get_expert_gate(self, idx):
        return getattr(self, f"gate_expert_{idx}")

    def get_expert_up(self, idx):
        return getattr(self, f"up_expert_{idx}")

    def get_expert_down(self, idx):
        return getattr(self, f"down_expert_{idx}")

    def forward(self, hidden_states):
        gating_logits = self.gating(hidden_states)
        
        weights = F.softmax(gating_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        final_output = torch.zeros_like(hidden_states)
        
        for k in range(self.top_k):
            expert_idx = top_indices[:, :, k]
            current_weight = top_weights[:, :, k].unsqueeze(-1)
            
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if not mask.any():
                    continue
                
                token_inputs = hidden_states[mask]
                
                # RECONSTRUCT PAIR: act(gate(x)) * up(x)
                gate_out = F.silu(self.get_expert_gate(i)(token_inputs))
                up_out = self.get_expert_up(i)(token_inputs)
                expert_out = gate_out * up_out
                
                down_out = self.get_expert_down(i)(expert_out)
                
                # Sum partitions (Standard for our Interleaved Architecture)
                final_output[mask] += down_out
                
        return final_output

def extract_number(text):
    if not text: return None
    # Use the robust base extraction logic
    match = re.findall(r"###\s*\[?(-?[\d,.]+)\]?", str(text))
    if not match:
        match = re.findall(r"####\s*(-?[\d,.]+)", str(text))
    if not match:
        # Fallback: Find any number that looks like a final answer
        # We look for the last number that isn't part of a date or unit
        match = re.findall(r"(-?[\d,]+(?:\.\d+)?)", str(text))
    
    if match:
        try:
            # Take the very last numerical value found in the text
            val_str = match[-1].replace(",", "").strip().rstrip('.')
            return float(val_str)
        except:
            return None
    return None

def dequantize_weight(weight_param):
    if hasattr(weight_param, "quant_state"):
        try:
            import bitsandbytes as bnb
            return bnb.functional.dequantize_4bit(weight_param.data, weight_param.quant_state)
        except Exception as e:
            return weight_param.data
    return weight_param.data

def convert_qwen_to_causal_moe(model, num_experts=4, moe_layers=None, reduction_factor=0.5):
    if moe_layers is None:
        total_layers = len(model.model.layers)
        moe_layers = list(range(total_layers // 4, 3 * total_layers // 4))

    print(f"⚡  Injecting SPARSE MoE (experts={num_experts}, reduction={reduction_factor})...")
    
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        orig_gate = dequantize_weight(layer.mlp.gate_proj.weight)
        orig_up = dequantize_weight(layer.mlp.up_proj.weight)
        orig_down = dequantize_weight(layer.mlp.down_proj.weight)
        
        new_mlp = CausalMoEMLP(model.config, num_experts=num_experts, top_k=2, reduction_factor=reduction_factor)
        
        with torch.no_grad():
            even_indices = torch.arange(0, model.config.intermediate_size, 2)
            odd_indices = torch.arange(1, model.config.intermediate_size, 2)
            
            # Simple routing init (can be improved by training)
            new_mlp.gating.weight.zero_()
            
            for i in range(num_experts):
                indices = even_indices if i % 2 == 0 else odd_indices
                indices = indices[:new_mlp.expert_intermediate_size]
                
                # CRITICAL: Pair the Gate and Up projections
                new_mlp.get_expert_gate(i).weight.copy_(orig_gate[indices, :])
                new_mlp.get_expert_up(i).weight.copy_(orig_up[indices, :])
                new_mlp.get_expert_down(i).weight.copy_(orig_down[:, indices])
        
        new_mlp.to(model.device).to(model.dtype)
        layer.mlp = new_mlp
        
    return model

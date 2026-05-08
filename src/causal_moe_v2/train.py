import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
import sys

# Add src to path so we can import our architecture
script_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.dirname(script_dir)
if src_root not in sys.path:
    sys.path.append(src_root)

from causal_moe_v2.architecture import convert_qwen_to_causal_moe

# ==========================================
# CONFIG (Matching src/experts/math/train.py)
# ==========================================
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/combined/final_moe_train.jsonl"
OUTPUT_DIR = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/causal_moe_v2/integrated_model"

MAX_LENGTH = 512
TARGET_STEPS = 400

# ==========================================
# CLEAN MATH STEP
# ==========================================
def clean_math(step: str):
    step = step.strip()

    remove_phrases = [
        "therefore", "thus", "hence", "this means",
        "so", "final answer", "Therefore", "Thus", "Hence"
    ]
    for p in remove_phrases:
        step = step.replace(p, "")
        step = step.replace(p.lower(), "")

    # normalize spacing
    step = " ".join(step.split())

    return step

# ==========================================
# DATA LOADING (Adapted for Integrated MoE)
# ==========================================
def tokenize_func(example, tokenizer):
    try:
        # The combined dataset (final_moe_train.jsonl) uses 'instruction' and 'atomic_reasoning'
        question = example.get("instruction", "").strip()
        answer = example.get("atomic_reasoning", "").strip()
        
        if not question or not answer:
            return None

        prompt = (
            "### Instruction:\n"
            "Analyze the following problem and provide a solution using ATOMIC reasoning tags: [MATH], [LOGIC], [COMMONSENSE].\n"
            "ALWAYS conclude with the [VERIFY] tag followed by the final answer like this: '[VERIFY] #### [number]'.\n\n"
            "### Input:\n"
            f"{question}\n\n"
            "### Response:\n"
        )
        
        full_text = prompt + answer + tokenizer.eos_token
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        
        if len(prompt_ids) >= len(input_ids):
            return None
            
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        tokenized["labels"] = labels
        
        return tokenized
    except:
        return None

# ==========================================
# TRAIN
# ==========================================
def train():
    set_seed(42)

    # ----------------------
    # TOKENIZER
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------
    # MODEL (With bitsandbytes 4-bit)
    # ----------------------
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
    )

    # --- ARCHITECTURE SWAP ---
    # Convert standard Qwen MLP layers to our new CausalMoEMLP
    # We target layers 6, 12, 18, 24 as a representative sample
    model = convert_qwen_to_causal_moe(model, num_experts=4, moe_layers=[6, 12, 18, 24])

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- PEFT CONFIG ---
    # We use LoRA for existing Qwen weights to preserve intelligence.
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
            # 🔥 PEFT COMPATIBLE TARGETS: Flattened names match architecture.py
            modules_to_save=[
                "gating",
                "gate_expert_0", "gate_expert_1", "gate_expert_2", "gate_expert_3",
                "up_expert_0", "up_expert_1", "up_expert_2", "up_expert_3",
                "down_expert_0", "down_expert_1", "down_expert_2", "down_expert_3"
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        ),
    )

    # PEFT with modules_to_save automatically handles unfreezing.
    # No manual unfreeze loop needed.

    model.config.pad_token_id = tokenizer.pad_token_id

    # ----------------------
    # DATASET
    # ----------------------
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_func(x, tokenizer),
        remove_columns=dataset.column_names,
    ).filter(lambda x: x is not None)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # ----------------------
    # RUN TRAINING
    # ----------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_steps=100,
        weight_decay=0.01,
        max_steps=TARGET_STEPS,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"🚀 Training Integrated Causal MoE (V2 - Layers [6, 12, 18, 24])")
    trainer.train()

    # ----------------------
    # SAVE
    # ----------------------
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()

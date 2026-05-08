import os
import torch
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

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/Qwen2.5-1.5B-Instruct"
DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/router/router_train_exploded.jsonl"
OUTPUT_DIR = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/router/router_qwen_1.5b"

MAX_LENGTH = 512
TARGET_STEPS = 400

# ==========================================
# TRAIN
# ==========================================
def train():
    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        ),
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_func(example):
        try:
            context = example["input_context"].strip()
            # Match evaluation/test preprocessing
            context = context.replace("[", "").replace("]", "")

            label = example["next_expert"].strip().upper()
            if label not in ["MATH", "LOGIC", "COMMONSENSE", "VERIFY"]:
                return None

            step = label + tokenizer.eos_token

            # 🔥 EXACT INFERENCE MATCH PROMPT
            prompt = (
                "### Instruction:\n"
                "Predict the NEXT required reasoning step.\n"
                "Choose the MOST appropriate tag:\n"
                "- MATH: numerical calculation\n"
                "- LOGIC: deductive reasoning\n"
                "- COMMONSENSE: world knowledge\n"
                "- VERIFY: final answer is ready\n\n"
                "### Input:\n"
                f"{context}\n\n"
                "### Response:\n"
            )

            full_text = prompt + step

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

            labels = input_ids.copy()
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

            tokenized["labels"] = labels
            return tokenized
        except:
            return None

    tokenized_dataset = dataset.map(
        tokenize_func,
        remove_columns=dataset.column_names,
    )

    tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_steps=TARGET_STEPS,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 🔥 FIX: SAFE CHECKPOINT RESUME LOGIC
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if last_checkpoint is not None and not os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
        print("⚠️ Incomplete checkpoint detected, starting fresh")
        last_checkpoint = None

    if last_checkpoint:
        print(f"🔁 Resuming ROUTER training from {last_checkpoint}")
    else:
        print(f"🆕 Starting fresh ROUTER training (target_steps={TARGET_STEPS})")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"💾 Saving final LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete. Use your separate merge script to create the BF16 model for vLLM.")

if __name__ == "__main__":
    train()
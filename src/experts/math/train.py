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
MODEL_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct"
DATA_PATH = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/experts/math/math_train_exploded.jsonl"
OUTPUT_DIR = "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/experts/math_expert_qwen_7b"

MAX_LENGTH = 256
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
# TRAIN
# ==========================================
def train():
    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # ----------------------
    # TOKENIZER
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------
    # MODEL
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

    # ----------------------
    # DATASET
    # ----------------------
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_func(example):
        try:
            instruction = example["instruction"].strip()
            context = example["context_before"].strip()

            cleaned_step = clean_math(example["math_step"])

            # enforce tag
            if not cleaned_step.startswith("[MATH]"):
                cleaned_step = f"[MATH] {cleaned_step}"

            # 🔥 STRICT FILTER: must contain numbers
            if not any(char.isdigit() for char in cleaned_step):
                return None

            # 🔥 keep short
            if len(cleaned_step.split()) > 12:
                return None

            step = cleaned_step + tokenizer.eos_token

            prompt = (
                "### Instruction:\n"
                "You are a mathematical reasoning expert.\n"
                "Perform the next calculation step.\n"
                "Output ONLY the computation.\n\n"
                "### Input:\n"
                f"Question: {instruction}\n"
                f"Context: {context}\n\n"
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

    # ----------------------
    # COLLATOR
    # ----------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # ----------------------
    # TRAINING
    # ----------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
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

    # ----------------------
    # CHECKPOINT RESUME
    # ----------------------
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if last_checkpoint is not None and not os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
        print("⚠️ Incomplete checkpoint detected, starting fresh")
        last_checkpoint = None

    if last_checkpoint:
        print(f"🔁 Resuming from {last_checkpoint}")
    else:
        print("🆕 Starting fresh training")

    print(f"🚀 Training MATH Expert (steps={TARGET_STEPS})")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ----------------------
    # SAVE
    # ----------------------
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR + "_merged")
        tokenizer.save_pretrained(OUTPUT_DIR + "_merged")
        print("✅ Saved merged model")
    except:
        print("⚠️ Merge skipped")

    print("💾 Math expert saved correctly.")


if __name__ == "__main__":
    train()
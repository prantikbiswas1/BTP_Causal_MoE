#!/bin/bash
# 08_atomic_compactor.sh
# Phase 4: Atomic Trace Distillation
# Uses the teacher model (Qwen-7B) to sanitize reasoning steps into Atomic format.

echo "--- Starting Atomic Compaction (Training Data) ---"

# GSM-8k (Train)
echo "Compacting GSM-8k (Train)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/atomic/atomic_compacter.py \
    --model /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/gsm8k_train_compact.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/gsm8k_train_atomic_v2.jsonl \
    --batch_size 1024

# CommonsenseQA (Train)
echo "Compacting CommonsenseQA (Train)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/atomic/atomic_compacter.py \
    --model /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/qwen2.5-7b-instruct \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/commonsense_qa_train_compact.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/atomic/commonsense_qa_train_atomic_v2.jsonl \
    --batch_size 1024

echo "Atomic compaction complete for all training datasets!"



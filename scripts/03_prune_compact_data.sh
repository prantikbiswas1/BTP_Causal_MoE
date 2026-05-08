#!/bin/bash
# prune_compact_data.sh
# Converts PNS-scored traces into compact CoT datasets for MoE training.
# Updated: Re-added test/val sets to have compact versions for all files.
# Path Sync: Consistent with /home/rupeshk_iitp/Prantik/BTP/ pattern.

echo "Pruning GSM-8k scored traces (Train)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/pruner.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/gsm8k_train_scored.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/gsm8k_train_compact.jsonl \
    --thresh 0.3

echo "--------------------------------------------------------"
echo "Pruning MATH-500 scored traces (Test)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/pruner.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/math500_test_scored.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/math500_test_compact.jsonl \
    --thresh 0.3

echo "--------------------------------------------------------"
echo "Pruning CommonsenseQA scored traces (Train)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/pruner.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/commonsense_qa_train_scored.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/commonsense_qa_train_compact.jsonl \
    --thresh 0.3

echo "--------------------------------------------------------"
echo "Pruning GSM-8k scored traces (Test)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/pruner.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/gsm8k_test_scored.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/gsm8k_test_compact.jsonl \
    --thresh 0.3

echo "--------------------------------------------------------"
echo "Pruning CommonsenseQA scored traces (Val)..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/pruner.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/commonsense_qa_val_scored.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/commonsense_qa_val_compact.jsonl \
    --thresh 0.3

echo "All compact dataset generation complete (Train, Test, and Validation)!"
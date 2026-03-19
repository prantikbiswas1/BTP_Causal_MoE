#!/bin/bash
# compute_all_pns.sh
# Runs the batched PNS evaluation engine over all generated CoT datasets.
# Use these exact commands to run directly in your terminal.

echo "Scoring PNS for GSM-8k..."
echo "Note: The checkpoint system will skip already processed IDs, so this is safe to rerun."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/run_pns_engine_batched.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/gsm8k_train_traces.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/gsm8k_train_scored.jsonl

echo "--------------------------------------------------------"
echo "Scoring PNS for MATH-500..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/run_pns_engine_batched.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/math500_test_traces.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/math500_test_scored.jsonl

echo "--------------------------------------------------------"
echo "Scoring PNS for CommonsenseQA..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/pns_engine/run_pns_engine_batched.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/commonsense_qa_train_traces.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/pns_scored/commonsense_qa_train_scored.jsonl

echo "All PNS scoring processes complete! The 'pns_scored' folder is fully equipped."

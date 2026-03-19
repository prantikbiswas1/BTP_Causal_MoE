#!/bin/bash
# generate_all_traces.sh
# Generates the initial CoT traces for all datasets (GSM-8k, MATH-500, CommonsenseQA).
# Use these exact commands to run directly in your terminal.

echo "Generating traces for GSM-8k..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/data_prep/generator.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/gsm8k_train.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/gsm8k_train_traces.jsonl \
    --k 5 --temp 0.7

echo "--------------------------------------------------------"
echo "Generating traces for MATH-500..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/data_prep/generator.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/math500_test.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/math500_test_traces.jsonl \
    --k 5 --temp 0.7

echo "--------------------------------------------------------"
echo "Generating traces for CommonsenseQA..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/data_prep/generator.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/raw/commonsense_qa_train.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/processed/commonsense_qa_train_traces.jsonl \
    --k 5 --temp 0.7

echo "All trace generation complete!"

#!/bin/bash
# 05_auto_label_taxonomy.sh
# Phase 3: Semantic Taxonomization (Auto-Labeling)
# Uses the teacher model (Qwen-7B) to categorize reasoning steps.

echo "Running auto-labeling using Qwen-7B Teacher..."
# CUDA_VISIBLE_DEVICES=0,1
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/auto_labeler.py \
    --model /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/Qwen2.5-7B-Instruct \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/sampled_compact/sampled_steps.jsonl \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/taxonomy/labeled_steps.jsonl \
    --batch_size 1024

echo "Auto-labeling complete: data/taxonomy/labeled_steps.jsonl"

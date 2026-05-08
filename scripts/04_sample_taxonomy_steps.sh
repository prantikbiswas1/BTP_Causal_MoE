#!/bin/bash
# 04_sample_taxonomy_steps.sh
# Phase 3: Semantic Taxonomization (Sampling)
# Samples 10,000 steps from compact datasets for labeling.

echo "Sampling steps for taxonomy labeling..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/step_sampler.py \
    --input_dir /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact \
    --output /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/sampled_compact/sampled_steps.jsonl \
    --n 10000 \
    --include_test

echo "Sampling complete: data/taxonomy/sampled_steps.jsonl"

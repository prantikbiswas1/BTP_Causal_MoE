#!/bin/bash
# 06_train_taxonomy_classifier.sh
# Phase 3: Semantic Taxonomization (Training)
# Fine-tunes DeBERTa-v3 on the auto-labeled data.

echo "Training the distilled DeBERTa-v3 taxonomy classifier..."
python /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/train_classifier.py \
    --input /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/taxonomy/labeled_steps.jsonl \
    --output_dir /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier \
    --base_model microsoft/deberta-v3-small

echo "Training complete! Model saved to: models/step_classifier/distilled_classifier"

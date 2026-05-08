#!/bin/bash
# 07_tag_all_data.sh
# Uses the trained DeBERTa classifier to tag reasoning steps for MoE training.

echo "--- Tagging TRAINING Data ---"
echo "Tagging GSM-8k (Train)..."
python "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/tag_dataset.py" \
    --input "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/gsm8k_train_compact.jsonl" \
    --output "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/gsm8k_train_tagged.jsonl" \
    --model "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier/distilled_classifier"
    
echo "Tagging CommonsenseQA (Train)..."
python "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/tag_dataset.py" \
    --input "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/commonsense_qa_train_compact.jsonl" \
    --output "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/commonsense_qa_train_tagged.jsonl" \
    --model "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier/distilled_classifier"

echo "--- Tagging EVALUATION Data ---"
echo "Tagging MATH-500 (Test)..."
python "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/tag_dataset.py" \
    --input "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/math500_test_compact.jsonl" \
    --output "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/math500_test_tagged.jsonl" \
    --model "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier/distilled_classifier"

echo "Tagging GSM-8k (Test)..."
python "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/tag_dataset.py" \
    --input "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/gsm8k_test_compact.jsonl" \
    --output "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/gsm8k_test_tagged.jsonl" \
    --model "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier/distilled_classifier"

echo "Tagging CommonsenseQA (Val)..."
python "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/step_classifier/tag_dataset.py" \
    --input "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/final_compact/commonsense_qa_val_compact.jsonl" \
    --output "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/data/tagged/commonsense_qa_val_tagged.jsonl" \
    --model "/home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/models/step_classifier/distilled_classifier"

echo "All compact files (Train, Test, Val) tagged successfully!"

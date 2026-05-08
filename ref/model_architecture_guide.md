# 🧠 Causal-MoE Architecture & Integration Guide

This guide explains the "Scientific Plumbing" of the Causal Mixture of Experts (Causal-MoE) system implemented for the BTP project.

## 1. The Model Hierarchy
| Role | Identity | Purpose |
| :--- | :--- | :--- |
| **Teacher** | `Qwen2.5-7B-Instruct` | The source of truth for reasoning traces and taxonomic labels. |
| **Router** | `DeBERTa-v3-small` (125M) | The high-speed classifier that predicts the `[TAG]` for each reasoning step. |
| **Student** | `DeepSeek-R1-1.5B-Qwen` | The core MoE model being trained to specialize. |

---

## 2. Component Logic Breakdown

### **A. Experts ([src/model/moe_experts.py](file:///home/nlp/Desktop/BTP/BTP_Causal_MoE/src/model/moe_experts.py))**
- **Structure**: We took the standard MLP/FFN layer and cloned it into 4 parallel "Experts".
- **Specialization**:
  - `Expert 0`: **MATH** Specialist.
  - `Expert 1`: **LOGIC** Specialist (The heavy-lifter).
  - `Expert 2`: **COMMONSENSE** Specialist.
  - `Expert 3`: **VERIFY** Specialist (Checks work).

### **B. Causal Router ([src/model/causal_router.py](file:///home/nlp/Desktop/BTP/BTP_Causal_MoE/src/model/causal_router.py))**
- **Mechanism**: Unlike standard "Softmax" routers that guess paths, our router uses **Explicit Causal Signals**.
- **Logic**: It scans the `input_ids` for the special tokens added by the Router model. If it sees `[MATH]`, it flips a binary switch to activate `Expert 0`.
- **Latency**: $O(1)$ lookup time. No extra neural passes needed inside the student model.

### **C. Custom MoE Layer ([src/model/modified_decoder.py](file:///home/nlp/Desktop/BTP/BTP_Causal_MoE/src/model/modified_decoder.py))**
- **Integration**: Replaces the standard `Qwen2MLP` block in the target layers (12-26).
- **PNS Bypassing**: A critical efficiency feature. If the router identifies a "Low-PNS" (redundant) step, it returns a zero mask. This causes the signals to skip the experts entirely, saving billions of FLOPs while preserving the residual path.

### **D. Integration Surgery ([src/model/integration.py](file:///home/nlp/Desktop/BTP/BTP_Causal_MoE/src/model/integration.py))**
- **Surgery**: This script iterates through the 1.5B model's layers and "hot-swaps" standard weights with our MoE modules.
- **Vocabulary**: It resizes the `embedding` and `lm_head` matrices to support the 4 new taxonomy tokens.

---

## 3. High-Level Inference Flow
1. **Input**: Question + Reasoning Steps.
2. **Tagging**: The **DeBERTa Router** injects tags: `[MATH] 2+2=4`.
3. **Execution**:
   - The **Student** reads `[MATH]`.
   - The **Causal Router** detects the ID.
   - The **Math Expert** is switched ON.
   - The calculation is processed by the specialist.
4. **Bypass**: If a filler step like "Hmm, let me think" appears without a tag, the experts stay OFF, saving compute.

---
**Status**: Ready for Phase 5 Fine-Tuning.
**Last Updated**: 2026-03-30

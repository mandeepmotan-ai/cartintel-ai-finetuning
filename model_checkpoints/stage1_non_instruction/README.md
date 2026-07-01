---
base_model: unsloth/llama-3.2-1b-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/llama-3.2-1b-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- domain-adaptation
- e-commerce
---

# CartIntel-AI (Stage 1: Domain Pre-Trained)

This model represents **Stage 1 (Continued Pre-Training)** of the CartIntel-AI project, an enterprise-grade multi-tenant e-commerce support assistant. This specific adapter has been explicitly trained to ingest and memorize internal corporate policies, operational guidelines, and compliance terminology for two simulated e-commerce infrastructures (AeroCart and LuminaCart).

## Model Details

### Model Description

CartIntel-AI is built using a Tri-Stage Fine-Tuning pipeline. This Stage 1 model utilizes Low-Rank Adaptation (LoRA) on the Llama-3.2-1B architecture to absorb domain-specific unstructured text. By undergoing this non-instruction fine-tuning phase, the model updates its internal weights to understand specific business logic (e.g., 60-minute cancellation grace periods, First Contact Resolution (FCR) targets, and LuminaPlus loyalty tiers) prior to learning Q&A behaviors.

- **Developed by:** Mandeep Motan
- **Model type:** Causal Language Model (LoRA Adapter)
- **Language(s) (NLP):** English
- **License:** Apache-2.0 (Inherited from Llama 3.2)
- **Finetuned from model:** `unsloth/llama-3.2-1b-unsloth-bnb-4bit`

## Uses

### Direct Use
This model is **not** intended for direct conversational use. Because it has only undergone non-instruction fine-tuning on raw policy text, it is optimized for text-completion tasks based on corporate guidelines. If prompted with a question, it may attempt to autocomplete the question rather than answer it.

### Downstream Use 
These Stage 1 weights serve as the foundational domain-adapted base for **Stage 2: Supervised Fine-Tuning (SFT)** and **Stage 3: Direct Preference Optimization (DPO)**.

## How to Get Started with the Model

To load this model for inference testing or continued training, utilize the `unsloth` library for native 2x faster inference:

```python
from unsloth import FastLanguageModel
import torch

model_path = "mandeep-motan/cartintel-ai-stage1" # Update with your HF repo path if hosted

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Test text completion on internal policy
prompt = "Customers can self-cancel orders through the dashboard within"
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 50, use_cache = True)
print(tokenizer.batch_decode(outputs, skip_special_tokens = True)[0])
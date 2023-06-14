#!/bin/bash
python utils/merge_llama_with_chinese_lora.py \
     --base_model /disc1/models/llama_hf/7B \
     --lora_model /disc1/models/chinese-llama-plus-lora-7b \
     --output_type huggingface \
     --output_dir /disc1/models_output/chinese-llama-plus
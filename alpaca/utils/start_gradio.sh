OUTPUT=/disc1/models_output/llama
python utils/gradio_demo.py \
	--base_model /disc1/models/llama_hf/7B \
	--lora_model ${OUTPUT}
lr=1e-4

pretrained_model=/disc1/models/llama_hf/7B
chinese_tokenizer_path=utils/merged_tokenizer_hf
dataset_dir=/disc1/rasa_data
per_device_train_batch_size=64
per_device_eval_batch_size=64
training_steps=3000
gradient_accumulation_steps=1
output_dir=/disc1/models_output/llama
validation_file=/disc1/rasa_data/val/alpaca_rasa_dev.json
peft_model=/disc1/models/chinese-alpaca-lora-7b

deepspeed_config_file=train/ds_zero2_no_offload.json

CUDA_VISIBLE_DIVICES=0,1,2,3 torchrun --nnodes 1 \
    --nproc_per_node 4 \
    train/run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 18000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --peft_path ${peft_model} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end

LR=5e-4
CUDA_VISIBLE_DEVICES=3 python3 lora.py \
    --do_train \
    --train_file /home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/train.json \
    --validation_file /home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /disc1/models/chatglm-6b \
    --output_dir /disc1/chatglm_output/adgen-chatglm-6b-lora-$LR \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --remove_unused_columns False \
    --max_steps 30 \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR 


PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=3 python3 ptuning_v2.py \
    --do_train \
    --train_file /home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/train.json \
    --validation_file /home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /disc1/models/chatglm-6b \
    --output_dir /disc1/chatglm_output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 30 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN


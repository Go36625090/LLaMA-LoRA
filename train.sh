#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py \
    --do_train \
    --train_file ../dataset/sampled_train.csv \
    --validation_file ../dataset/sampled_validation.csv \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path ../THUDM/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-lora_version \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --num_train_epochs 30 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --lora_r 32 \
    --model_parallel_mode True
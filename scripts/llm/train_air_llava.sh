#!/bin/bash
base_dir=/home/fit/qiuhan/WORK/wmq/AirVLN_ws
root_dir=$base_dir/AirVLN
model_dir=$root_dir/Model/LLaMA-UAV
deepspeed \
    --include localhost:0 \
    --master_port 29101 \
    $root_dir/src/vlnce_src/train_uav_notice.py \
    --data_path $base_dir/DATA/data/aerialvln/train.json \
    --dataset_path $base_dir/DATA/img_features/collect/AirVLN-seq2seq/train \
    --output_dir $root_dir/work_dirs/llava-7b-pretrain-336-uav-full-data-lora64_bs128 \
    --deepspeed /home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/scripts/zero2.json \
    --model_name_or_path /home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b \
    --version imgsp_uav \
    --is_multimodal True \
    --vision_tower /home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_waypoint_predictor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --compress_type "mean" \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True \
    
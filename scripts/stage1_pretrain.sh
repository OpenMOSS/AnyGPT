#!/bin/bash
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3

METAROOT="Path-to-pretrained-model" 
# https://huggingface.co/fnlp/AnyGPT-base
METAROOT="/mnt/petrelfs/share_data/llama2_hf/llama-2-7b-hf" 
DATAROOT="./data"

OUTROOT="output_models/pretrain"
CACHEROOT="${DATAROOT}/cache"

create_cache_directories() {
    local dataset_path=$1
    local modality=$2
    local dataset_name=$(echo $dataset_path | tr " " "\n" | awk -F "/" '{print $NF}' | awk -F "." '{print $1}' | tr "\n" " ")
    echo "datasets name: ${dataset_name}"
    for dataset in $dataset_name
    do
        mkdir -p ${CACHEROOT}/${modality}/${dataset}/tokenized/train/
        mkdir -p ${CACHEROOT}/${modality}/${dataset}/tokenized/test/
        mkdir -p ${CACHEROOT}/${modality}/${dataset}/group/train/
    done
}

music_datasets="${DATAROOT}/music/music-1m.jsonl"
speech_datasets="${DATAROOT}/speech/mls.jsonl ${DATAROOT}/speech/commonvoice.jsonl ${DATAROOT}/speech/gigaspeech.jsonl"
image_datasets="${DATAROOT}/image/laion-coco-caption.jsonl ${DATAROOT}/image/journeydb.jsonl ${DATAROOT}/image/laion_aesthetics_6plus_8m.jsonl"
mls_data_path="${DATAROOT}/speech/mls.jsonl"

create_cache_directories "$music_datasets" "music"
create_cache_directories "$speech_datasets" "speech"
create_cache_directories "$mls_data_path" "speech"
create_cache_directories "$image_datasets" "image"

#ddp realted
NNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NODE_RANK=$(($(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -m 1 -Fn $(hostname) | cut -d ":" -f1)-1))

echo "stage1: multimodel pretraining"
torchrun \
    --nnode $NNODE \
    --nproc_per_node 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port 29501  \
    anygpt/src/train/stage1_pretrain.py \
    --run_name "mm_pretrain" \
    --model_name_or_path "${METAROOT}" \
    --image_pair_data_path "${image_datasets}" \
    --music_data_path "${music_datasets}" \
    --speech_data_path "${speech_datasets}" \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 100 \
    --bf16 True \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --model_max_length 4500 \
    --block_size 4500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --max_steps 4096 \
    --report_to "wandb" \
    --num_train_epochs 1 \
    --val_set_size 5 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir False\
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --use_flash_attn True \
    --ddp_timeout 7200 \
    --save_total_limit 8
    # --max_grad_norm 0.6
    # --max_steps 50000 \
    # --dispatch_batches False \
    # --streaming True \



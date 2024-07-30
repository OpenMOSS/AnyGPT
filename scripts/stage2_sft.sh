#!/bin/bash
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3

# path-to-pretrain-model
METAROOT="output_models/mm_pretrain2/checkpoint-4000"

DATAROOT="./data"
OUTROOT="output_models/sft"
CACHEROOT="${DATAROOT}/cache/sft"

create_cache_directories() {
    local dataset_path=$1
    local dataset_name=$(echo $dataset_path | tr " " "\n" | awk -F "/" '{print $NF}' | awk -F "." '{print $1}' | tr "\n" " ")
    echo "datasets name: ${dataset_name}"
    for dataset in $dataset_name
    do
        mkdir -p ${CACHEROOT}/${dataset}/tokenized/train/
        mkdir -p ${CACHEROOT}/${dataset}/tokenized/test/
        mkdir -p ${CACHEROOT}/${dataset}/group/train/
    done
}

mmi_datasets="${DATAROOT}/instruction/anyinstruct_speech.jsonl
${DATAROOT}/instruction/anyinstruct_text.jsonl"
for dataset in $mmi_datasets
do 
    create_cache_directories "$dataset"
done

#ddp realted
NNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NODE_RANK=$(($(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -m 1 -Fn $(hostname) | cut -d ":" -f1)-1))

echo "stage2: multimodal interleaved surpervised fineturning"

torchrun \
    --nnode $NNODE \
    --nproc_per_node 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port 29501  \
    anygpt/src/train/stage2_sft.py \
    --model_name_or_path "${METAROOT}" \
    --run_name "mm_sft" \
    --cache_dir ${CACHEROOT} \
    --report_to "wandb" \
    --mmi_datasets "$mmi_datasets" \
    --preprocessing_num_workers 100 \
    --bf16 True \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --model_max_length 4096 \
    --save_strategy "steps" \
    --save_steps 500 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --max_steps 5000 \
    --concatenating False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --val_set_size 10 \
    --learning_rate 2e-5 \
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
    --save_total_limit 10

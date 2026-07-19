#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do 
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python generate_images.py \
    --savedir "/COCO2017-val" \
    --expdir "/exp_log" \
    --dataset_name "custom" \
    --dataset_anno_file "/annotations/partiprompt.json" \
    --gpu_id ${GPULIST[$IDX]} \
    --gpu_ids ${gpu_list//,/ } \
    --node_id 0 \
    --node_ids 0 &
done
wait


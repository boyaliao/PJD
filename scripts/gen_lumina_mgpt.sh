#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAVE_DIR="${SAVE_DIR:-./outputs/coco2017-val}"
EXP_DIR="${EXP_DIR:-./exp_log}"
ANNOTATION_FILE="${ANNOTATION_FILE:-./annotations/partiprompt.json}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"

for gpu_id in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python generate_images.py \
            --savedir "${SAVE_DIR}" \
            --expdir "${EXP_DIR}" \
            --dataset_name "custom" \
            --dataset_anno_file "${ANNOTATION_FILE}" \
            --gpu_id "${gpu_id}" \
            --gpu_ids "${GPUS[@]}" \
            --node_id 0 \
            --node_ids 0 &
done
wait
#!/bin/sh  在多个卡上同时跑，通过设置CUDA_VISIBLE_DEVICES，每个程序都写的是cuda:0,只要设置gpu_id不同就可以分配不同的数据集
export CUDA_VISIBLE_DEVICES=6

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# for IDX in $(seq 0 $((CHUNKS-1))); do 
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python generate_images.py \
#     --savedir "/home/liaoboya/Datasets/nateraw/parti-prompts" \
#     --expdir "/home/liaoboya/Projects/Accelerating-T2I-AR-with-SJD/exp_log/" \
#     --dataset_name "parti_cocoformat" \
#     --dataset_anno_file "/home/liaoboya/Datasets/nateraw/parti-prompts/PartiPrompts.tsv" \
#     --gpu_id ${GPULIST[$IDX]} \
#     --gpu_ids ${gpu_list//,/ } \
#     --node_id 0 \
#     --node_ids 0 &
# done
# wait

for IDX in $(seq 0 $((CHUNKS-1))); do 
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python generate_images.py \
    --savedir "/home/liaoboya/Datasets/MS-COCO/COCO2017-val" \
    --expdir "/home/liaoboya/Projects/Accelerating-T2I-AR-with-SJD/exp_log/" \
    --dataset_name "custom" \
    --dataset_anno_file "/home/liaoboya/Datasets/MS-COCO/COCO2017-val/annotations/lumin_pp.json" \
    --gpu_id ${GPULIST[$IDX]} \
    --gpu_ids ${gpu_list//,/ } \
    --node_id 0 \
    --node_ids 0 &
done
wait


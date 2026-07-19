export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# CUDA_LAUNCH_BLOCKING=1 
python3 test_llamagen.py \
        --vq_ckpt /home/liaoboya/Datasets/LlamaGen/vq_ds16_t2i.pt \
        --gpt_ckpt /home/liaoboya/Datasets/LlamaGen/t2i_XL_stage2_512.pt \
        --gpt_model GPT-XL \
        --image_size 512 \
        --t5_path /home/liaoboya/Datasets/LlamaGen/t5-ckpt \
        --top_p 1.0 \
        --cfg_scale 3.5
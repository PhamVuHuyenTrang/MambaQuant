export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_OFFLINE=1

# RESUME="./saved_checkpoint/vim_t+_midclstok_ft_78p3acc.pth"
# MODEL="vim_tinyplus_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

# RESUME="./saved_checkpoint/vim_s+_midclstok_ft_81p6acc.pth"
# MODEL="vim_smallplus_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

# RESUME="./saved_checkpoint/vim_t_midclstok_76p1acc.pth"
# MODEL="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

# RESUME="./saved_checkpoint/vim_s_midclstok_80p5acc.pth"
# MODEL="vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

RESUME="./saved_checkpoint/vim_b_midclstok_81p9acc.pth"
MODEL="vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"

# echo "w8a8 naive quantization **********************************************************************"
# /data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
#     --eval \
#     --resume "$RESUME" \
#     --model "$MODEL" \
#     --data-path "/data01/datasets/imagenet" \
#     --use_vim_torch True \
#     --batch-size 32 \
#     --static_quant \
#     --quant_weight \
#     --quant_act \
#     --w_bit 8 \
#     --a_bit 8 \
#     --w_perchannel
    
# echo "w8a8 smoothquant **********************************************************************"
# /data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
#     --eval \
#     --resume "$RESUME"\
#     --model "$MODEL" \
#     --data-path "/data01/datasets/imagenet" \
#     --use_vim_torch True \
#     --batch-size 32 \
#     --static_quant \
#     --quant_weight \
#     --quant_act \
#     --use_smoothquant True\
#     --w_bit 8 \
#     --a_bit 8 \
#     --w_perchannel

# echo "w8a8 gptq **********************************************************************"
# /data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
#     --eval \
#     --resume "$RESUME"\
#     --model "$MODEL" \
#     --data-path "/data01/datasets/imagenet" \
#     --use_vim_torch True \
#     --batch-size 32 \
#     --static_quant \
#     --quant_weight \
#     --quant_act \
#     --use_gptq \
#     --w_bit 8 \
#     --a_bit 8 \
#     --w_perchannel

echo "w8a8 method_1A **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
    --eval \
    --resume "$RESUME"\
    --model "$MODEL" \
    --data-path "/data01/datasets/imagenet" \
    --use_vim_torch True \
    --batch-size 32 \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 8 \
    --a_bit 8 \
    --w_perchannel \
    --use_hadmard \
    --use_S2 \
    --use_S4 \
    --fake_online_hadamard 

echo "w8a8 method_1B **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
    --eval \
    --resume "$RESUME"\
    --model "$MODEL" \
    --data-path "/data01/datasets/imagenet" \
    --use_vim_torch True \
    --batch-size 32 \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 8 \
    --a_bit 8 \
    --w_perchannel \
    --use_hadmard \
    --use_S2 \
    --use_S4 \
    --use_hadmard_R1 \
    --use_hadmard_R5 \
    --fake_online_hadamard

echo "w8a8 method_1C **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
    --eval \
    --resume "$RESUME"\
    --model "$MODEL" \
    --data-path "/data01/datasets/imagenet" \
    --use_vim_torch True \
    --batch-size 32 \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 8 \
    --a_bit 8 \
    --w_perchannel \
    --use_hadmard \
    --use_S2 \
    --use_S4 \
    --use_hadmard_R1 \
    --use_hadmard_R2 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --fake_online_hadamard  

echo "w8a8 method_2 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
    --eval \
    --resume "$RESUME"\
    --model "$MODEL" \
    --data-path "/data01/datasets/imagenet" \
    --use_vim_torch True \
    --batch-size 32 \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 8 \
    --a_bit 8 \
    --w_perchannel \
    --use_hadmard \
    --use_S2 \
    --use_S4 \
    --use_S5 \
    --use_S7 \
    --use_hadmard_R1 \
    --use_hadmard_R2 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --use_klt \
    --observe percentile

echo "w8a8 method_2+S3 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba/bin/python tools/main_quant_naive.py \
    --eval \
    --resume "$RESUME"\
    --model "$MODEL" \
    --data-path "/data01/datasets/imagenet" \
    --use_vim_torch True \
    --batch-size 32 \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 8 \
    --a_bit 8 \
    --w_perchannel \
    --use_hadmard \
    --use_S2 \
    --use_S3 \
    --use_S4 \
    --use_S5 \
    --use_S7 \
    --use_hadmard_R1 \
    --use_hadmard_R2 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --use_klt \
    --observe percentile
                
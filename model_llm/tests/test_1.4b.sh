#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_OFFLINE=1

tasks=("arc_easy" "arc_challenge" "piqa" "winogrande")
for task in "${tasks[@]}"; do
    /data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tests/quant_naive.py \
        --model mamba \
        --model_args pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-1.4b-hf \
        --tasks "$task" \
        --device cuda \
        --batch_size 64 \


done
        # "arc_easy" "arc_challenge" "piqa" "winogrande" "hellaswag" "lambada_openai" "rte" "copa"
        # --quant_weight \
        # --quant_act \
        # --use_smoothquant \
        # --use_gptq \
        # --use_hadmard \
        # --use_hadmard_R1 \
        # --use_hadmard_R3S \
        # --use_perkernel \
        # --use_gptq \
        # --static_quant 